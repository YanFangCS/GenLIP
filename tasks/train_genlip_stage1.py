import json
import os
import time
import datetime
from dataclasses import asdict, dataclass, field
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Literal

import numpy as np
import torch
import torch.distributed as dist
import wandb
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

from tqdm import trange

try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
    )
except ImportError:
    print("To enable flexattention, please install torch>=2.5.0")

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    OmniDataCollatorWithPacking,
    OmniDataCollatorWithPadding,
    OmniSequenceShardCollator,
    build_dataloader,
    build_mapping_dataset,
    build_iterative_dataset,
    build_iterative_webdataset,
    build_mixed_iterative_wdsapi,
    build_multimodal_chat_template,
)
from veomni.data.constants import IMAGE_INPUT_INDEX
from veomni.data.multimodal.preprocess import conv_preprocess
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_processor, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.dist_utils import all_reduce
from veomni.utils.flex_attn_utils import create_fast_flex_mask_padding, create_flex_mask_padding

from transformers import AutoModelForCausalLM

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from veomni.data.chat_template import ChatTemplate

logger = helper.create_logger(__name__)

def _create_single_modality_attn_mask(
    mask: torch.Tensor
):
    bs, seq_len = mask.shape
    device = mask.device
    
    is_modality = mask.int()
    prepended_input = torch.cat(
        (torch.zeros(bs, 1, device=device, dtype=is_modality.dtype), is_modality), dim=-1
    )
    diffs = torch.diff(prepended_input, dim=-1)
    
    segment_starts = (diffs != 0).int()
    segment_ids = torch.cumsum(segment_starts, dim=-1)
    pair_ids = (segment_ids + 1) // 2

    q_pair_ids = pair_ids.unsqueeze(2)  # (bs, seq_len, 1)
    k_pair_ids = pair_ids.unsqueeze(1)  # (bs, 1, seq_len)
    
    same_pair_mask = q_pair_ids == k_pair_ids
    
    k_is_modality = mask.unsqueeze(1)
    
    hybrid_mask = same_pair_mask & k_is_modality
    
    q_is_text = (~mask).unsqueeze(2)
    k_is_text = (~mask).unsqueeze(1)
    
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)).unsqueeze(0)
    
    text_to_text_mask = same_pair_mask & q_is_text & k_is_text & causal_mask
    
    hybrid_mask = hybrid_mask | text_to_text_mask
    
    last_token_pair_ids = pair_ids[:, -1].unsqueeze(1)
    is_in_last_pair = pair_ids == last_token_pair_ids
    hybrid_mask[:, -1, :] = hybrid_mask[:, -1, :] | is_in_last_pair
    
    return hybrid_mask

def get_hybrid_attn_mask(
    image_mask=None,
    video_mask=None,
):
    if image_mask is None and video_mask is None:
        return None

    if image_mask is not None:
        bs, seq_len = image_mask.shape
        device = image_mask.device
    else:
        bs, seq_len = video_mask.shape
        device = video_mask.device

    final_mask = torch.zeros(bs, seq_len, seq_len, dtype=torch.bool, device=device)

    if image_mask is not None:
        final_mask = final_mask | _create_single_modality_attn_mask(image_mask)
    
    if video_mask is not None:
        final_mask = final_mask | _create_single_modality_attn_mask(video_mask)
        
    return final_mask

def constraint_max_len(
    tokenized: Dict[str, Any],
    max_len: int,
    **kwargs,
):
    """
    Truncates the input sequence to the maximum length.
    """
    sample_len = len(tokenized["input_ids"])
    if sample_len <= max_len:
        return tokenized
    
    # truncate if longer than max_len
    for k in ["input_ids", "attention_mask", "labels", "flex_indicators", "position_ids"]:
        if k in tokenized:
            tokenized[k] = tokenized[k][:max_len]
    
    for k in ["image_mask", "video_mask"]:
        if k in tokenized:
            if tokenized[k][max_len:].sum() > 0:
                return None
            tokenized[k] = tokenized[k][:max_len]

    return tokenized

# it is a simplified version for computing position ids,
# just suitable for sequence without special token around vision tokens
def get_position_ids(
    image_mask,
    image_grid_thw
):
    # compute position ids for mixed sequence
    # image_mask [1, seq_len]
    # image_grid_thw [1, 3]
    grid_t, grid_h, grid_w = (
        image_grid_thw[0][0],
        image_grid_thw[0][1],
        image_grid_thw[0][2],
    )
    # assign time axis
    range_tensor = torch.arange(grid_t).view(-1, 1)
    expanded_range = range_tensor.expand(-1, grid_h * grid_w)
    time_tensor_long = expanded_range.long()

    t_index = time_tensor_long.flatten()
    h_index = torch.arange(grid_h).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()

    vis_pos_ids = torch.stack([t_index, h_index, w_index]) 

    text_start_id = vis_pos_ids.max() + 1
    text_len = image_mask.shape[-1] - grid_t * grid_h * grid_w
    txt_pos_ids = torch.arange(text_len).view(1, -1).expand(3, -1) + text_start_id
    
    position_ids = torch.cat([vis_pos_ids, txt_pos_ids], dim=-1)

    return position_ids

# you should modify this sample processing function based on your dataset structure,
# our implementation is based on our dataset processing pipeline
def process_sample(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    flex_indicator_func: "Callable", 
    max_len: int=4096,
    **kwargs,
):
    """
    Processes multimodal example with qwen2vl's pre-processor.
    """
    source = (
        kwargs["source_name"] if "source_name" in kwargs else sample["source"]
    )
    image = None
    if "jpg" in sample and len(sample['jpg']) > 0:
        try:
            image = Image.open(BytesIO(sample["jpg"]["bytes"]))
        except:
            image = None
    elif 'image' in sample and len(sample['image']) > 0:
        try:
            image = Image.open(BytesIO(sample["image"]["bytes"]))
        except:
            image = None

    if image is not None:
        if image.mode == 'P':
            image = image.convert('RGBA').convert('RGB')
        else:
            image = image.convert('RGB')
    else:
        print(f"Sample with no image.")
        return None  
    
    if image.size == (1, 1):
        print("image size:", image.size, flush=True)
        return None
    
    meta = sample["json"]
    conversations = conv_preprocess(source, meta, **kwargs)

    token_num_inputs, image_inputs = {}, {}
    try:
        image_inputs = processor.image_processor(images=[image], input_data_format="channels_last", return_tensors="pt")
    except:
        print(f"Sample with no image.")
        return None
    token_w, token_h = image_inputs['pixel_values'].shape[-2] // 16, image_inputs['pixel_values'].shape[-1] // 16
    image_token_num = token_w * token_h
    token_num_inputs["image"] = [image_token_num]
    # image_grid_thw = torch.tensor([(1, token_h, token_w)])
    image_grid_thw = [(1, token_h, token_w)]

    try:
        # it is a simplified verision to mitigate attention sink problem
        tokenized_example = chat_template.encode_messages(conversations, token_num_inputs, no_special_tokens=True)
    except:
        print(f"Sample with no conversation data field.")
        return None
    tokenized_example = {k: v.clone().detach() if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in tokenized_example.items()}
    input_ids = tokenized_example["input_ids"]

    if tokenized_example['image_mask'].sum() != image_token_num:
        print(f"Sample with image mask error.")
        return None

    position_ids = get_position_ids(
        image_mask = tokenized_example['image_mask'].unsqueeze(0),
        image_grid_thw = image_grid_thw
    ) # position_ids [3, L]

    tokenized_example["position_ids"] = position_ids.squeeze()   # (dim, l) -> [3, l]
    tokenized_example["image_grid_thw"] = torch.tensor(image_grid_thw)

    tokenized_example["flex_indicators"] = torch.zeros_like(tokenized_example["input_ids"])
    tokenized_example["flex_indicators"][tokenized_example["image_mask"]] = 1

    tokenized_example["image_mask"] = tokenized_example["input_ids"] == IMAGE_INPUT_INDEX 
    tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
    tokenized_example.update(image_inputs)
    tokenized_example = constraint_max_len(tokenized_example, max_len=max_len)
    return [tokenized_example]

def get_param_groups(model: "torch.nn.Module", default_lr: float, vit_lr: float):
    vit_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info_rank0(f"Trainable parameter - name {name}")
            if "visual" in name:
                vit_params.append(param)
            else:
                other_params.append(param)

    return [{"params": vit_params, "lr": vit_lr}, {"params": other_params, "lr": default_lr}]

@dataclass
class MyModelArguments(ModelArguments):
    image_processor: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
    tokenizer: Optional[str] = field(default=None) # default Qwen2-VL tokenizer, which is a subset of Qwen3 tokenizer
    use_flex_attn: bool = field(default=False)

@dataclass
class MyDataArguments(DataArguments):
    source_name: str = field(
        default=None,
        metadata={"help": "Source name of dataset."},
    )
    enable_multisource: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable multi-source training."},
    )
    datasets_type: Literal["mapping", "iterable_wds", "iterable"] = field(
        default="mapping",
        metadata={"help": "Type of the datasets."},
    )
    pure_mm: bool = field(
        default=True,
        metadata={"help": "Whether or not to enable pure-mm data loading for training."},
    )

@dataclass
class MyTrainingArguments(TrainingArguments):
    freeze_encoder: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vit parameters."},
    )
    freeze_decoder: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the decoder parameters."},
    )
    freeze_emb_tok: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the lm head parameters."},
    )
    freeze_lm_head: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the lm head parameters."},
    )
    step2token_path: str = field(
        default=None,
        metadata={"help":"custom step2token path usage."}
    )

@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=MyModelArguments)
    data: "DataArguments" = field(default_factory=MyDataArguments)
    train: "TrainingArguments" = field(default_factory=MyTrainingArguments)

def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    torch.cuda.set_device(f"cuda:{args.train.local_rank}")
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=1500), # timeout threshold 15 mins
    )
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)
    
    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32"
        if args.train.data_parallel_mode == "fsdp1" and args.train.enable_mixed_precision
        else "bfloat16",
        # torch_dtype="bfloat16",
        init_device=args.train.init_device,
        attn_implementation=args.model.attn_implementation,
    )
    model_config = model.config
    if args.model.model_path is None:
        model.initialize_weights()
        logger.info_rank0("Call model initialization function.")
    if model_config.llm_decoder is not None:
        llm = AutoModelForCausalLM.from_pretrained(model_config.llm_decoder)
        lm_head_weight = llm.lm_head.weight.to(dtype=model.lm_head.weight.dtype, device=model.device)
        if torch.equal(lm_head_weight, model.lm_head.weight):
            logger.info_rank0(f"Check lm_head weight correctly initialized from {model_config.llm_decoder}!")
        else:
            logger.info_rank0(f"lm_head weight not correctly initialized, get lm_head in model {model.lm_head.weight}\n, but in llm {lm_head_weight}")

        lm_emb_weight = llm.model.embed_tokens.weight.to(dtype=model.embeddings.weight.dtype, device=model.device)
        if torch.equal(lm_emb_weight, model.embeddings.weight):
            logger.info_rank0(f"Check embedding layers weight correctly initialized from {model_config.llm_decoder}!")
        else:
            logger.info_rank0(f"embedding layers weight not correctly initialized, get embeddings in model {model.embeddings.weight}\n, but in llm {lm_emb_weight}")
    else:
        logger.info_rank0(f"Model language modules are random initialized!")

    helper.print_device_mem_info("VRAM usage after building model")
    
    logger.info_rank0(f"Model Config:\n {model_config}")
    logger.info_rank0(f"Model Arch:\n {model}")
    logger.info_rank0("Prepare data")
    processor = build_processor(args.model.tokenizer_path)
    if args.model.image_processor is not None:
        processor.image_processor = build_processor(args.model.image_processor).image_processor
        if processor.image_processor.size["height"] != model_config.image_size:
            logger.info_rank0(f"The image resize params in image_processor {processor.image_processor.size} not consistent with model {model_config.image_size}")
            processor.image_processor.size["height"] = model_config.image_size
            processor.image_processor.size["width"]  = model_config.image_size
            processor.image_processor.crop_size = processor.image_processor.size
    if args.model.tokenizer is not None:
        processor.tokenizer = build_tokenizer(args.model.tokenizer) # replace with required tokenizer
    
    chat_template = build_multimodal_chat_template(args.data.chat_template, processor.tokenizer)
    position_id_func = model.get_position_id_func()
    transform = partial(
        process_sample,
        processor=processor,
        chat_template=chat_template,
        source_name=args.data.source_name,
        position_id_func=position_id_func,
        flex_indicator_func=model.get_flex_indicators_func() if args.model.use_flex_attn else None,
        max_len=args.data.max_seq_len,
    )

    if args.train.rmpad:
        raise ValueError("Qwen2-VL does not support rmpad. Use `rmpad_with_pos_ids` instead.")

    data_collate_fn = []
    if args.train.rmpad_with_pos_ids:
        data_collate_fn.append(OmniDataCollatorWithPacking())
    else:
        data_collate_fn.append(OmniDataCollatorWithPadding())
    if get_parallel_state().sp_enabled:
        data_collate_fn.append(
            OmniSequenceShardCollator(
                padding_scale={
                    "pixel_values": processor.image_processor.merge_size**2,
                },
                rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            )
        )
    if args.data.dataloader_type == "native":
        if args.data.datasets_type == "iterable_wds":
            logger.info_rank0("Start building iterative webdataset")
            train_dataset = build_iterative_webdataset(args.data.train_path, transform=transform, seed=args.train.seed)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size)
        elif args.data.datasets_type == "iterable_wds_api":
            logger.info_rank0("Start building iterative webdataset")
            if ';' in args.data.train_path:
                train_pathes = args.data.train_path.strip().split(';')
                train_dataset = build_mixed_iterative_wdsapi(train_pathes, transform=transform, seed=args.train.seed, pure_mm=args.data.pure_mm)
                logger.info_rank0(f"Start building iterative webdataset with mixed data from {len(train_pathes)} sources, using build_mixed_iterative_wdsapi api.")
            else:
                train_dataset = build_mixed_iterative_webdataset(args.data.train_path, transform=transform, seed=args.train.seed, pure_mm=args.data.pure_mm)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size)
        elif args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size)
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))
            
        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            collate_fn=data_collate_fn,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
            stateful=True,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")
    
    fsdp_kwargs = {}
    if args.train.freeze_encoder:
        model.visual.requires_grad_(False)
        if args.train.data_parallel_mode == "fsdp1":
            fsdp_kwargs["use_orig_params"] = True
    if args.train.freeze_decoder:
        if model.use_llm_decoder:
            model.decoder.requires_grad_(False)
            model.decoder.eval()
            if args.train.data_parallel_mode == "fsdp1":
                fsdp_kwargs["use_orig_params"] = True
    if args.train.freeze_emb_tok:
        model.embeddings.requires_grad_(False)
        if args.train.data_parallel_mode == "fsdp1":
            fsdp_kwargs["use_orig_params"] = True
    if args.train.freeze_lm_head:
        model.lm_head.requires_grad_(False)
        if args.train.data_parallel_mode == "fsdp1":
            fsdp_kwargs["use_orig_params"] = True

    print(f"train_init_device, {args.train.init_device}")
    model = build_parallelize_model(
        model,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        init_device=args.train.init_device,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        fsdp_kwargs=fsdp_kwargs,
        basic_modules=model._no_split_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )
    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=False,
        optimizer_type=args.train.optimizer,
        param_groups=get_param_groups(model, args.train.lr, args.train.lr),
    )
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        if args.train.enable_profiling:
            profiler = helper.create_profiler(
                start_step=args.train.profile_start_step,
                end_step=args.train.profile_end_step,
                trace_dir=args.train.profile_trace_dir,
                record_shapes=args.train.profile_record_shapes,
                profile_memory=args.train.profile_profile_memory,
                with_stack=args.train.profile_with_stack,
            )
            profiler.start()

        model_assets = [model_config, processor]
        save_model_assets(args.train.model_assets_dir, model_assets)

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        # dataloader=train_dataloader,
        # data_path=args.data.train_path,
    )

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        # extra_state = state.get("extra_state") or {}
        # torch_rng_state = extra_state.get("torch_rng_state", None)
        # if torch_rng_state is None:
        #     logger.warning_rank0(
        #         "Checkpoint missing extra_state['torch_rng_state']; continue with current RNG state."
        #     )
        # else:
        #     try:
        #         torch.set_rng_state(torch_rng_state)
        #     except Exception as e:
        #         logger.warning_rank0(
        #             f"Failed to restore torch_rng_state from checkpoint ({type(e).__name__}: {e}); "
        #             "continue with current RNG state."
        #         )


        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data
        
        if args.train.global_rank == 0:
            helper.load_step2token(args.train.load_checkpoint_path)
        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()      
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1
            try:
                # batch data_format
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            torch.cuda.synchronize()
            start_time = time.time()
            for micro_batch in micro_batches:
                # logger.info(f"micro_batch position ids, {type(micro_batch['position_ids'])}, {micro_batch['position_ids'].shape}")
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("cur_token_num", None)

                micro_batch = {
                    k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()
                }
                logger.info_rank0(f"micro_batch input_ids:{micro_batch['input_ids'].shape}")
                if args.model.use_flex_attn:
                    flex_mask=create_fast_flex_mask_padding(
                        micro_batch["sample_ids"], 
                        micro_batch["flex_indicators"], 
                        args.data.max_seq_len,
                    )
                    flex_args = dict(
                        flex_mask=flex_mask,
                        # max_seqlen=div_num,
                        max_seqlen=args.data.max_seq_len,
                    )
                    micro_batch["flex_attn_args"] = flex_args
                    if global_step == 1:
                        logger.info_rank0(f"input_ids {micro_batch['input_ids'].shape}, {micro_batch['input_ids']}")
                        logger.info_rank0(f"labels {micro_batch['labels'].shape}, {micro_batch['labels']}")
                        logger.info_rank0(f"position_ids  {micro_batch['position_ids'].shape}, {micro_batch['position_ids'][0][0][-1000:]}, {micro_batch['position_ids'][0][1][-1000:]}, {micro_batch['position_ids'][0][2][-1000:]}")
                        logger.info_rank0(f"flex_attn_args {flex_mask.to_string()}")
                    micro_batch.pop("flex_indicators")
                    micro_batch.pop("sample_ids")
                
                with model_fwd_context:
                    loss: "torch.Tensor" = model(**micro_batch, use_cache=False).loss / len(micro_batches)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = model.clip_grad_norm_(args.train.max_grad_norm).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm, foreach=True)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # empty cache in the first few steps to save memory
            if global_step <= 3:
                torch.cuda.empty_cache()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            torch.cuda.synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}")
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                    )
                    wandb.log(train_metrics, step=global_step)

                if args.train.enable_profiling and global_step <= args.train.profile_end_step:
                    profiler.step()
                    if global_step == args.train.profile_end_step:
                        profiler.stop()
                        helper.upload_trace(
                            args.train.wandb_project, args.train.wandb_name, args.train.profile_trace_dir
                        )

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                
                if args.train.global_rank == 0:
                    helper.save_step2token(
                        os.path.join(args.train.save_checkpoint_path, 'step2token.txt'),
                        consumed_tokens=train_metrics["consume_tokens(B)"],
                        global_step=global_step,
                        avg_effective_len=train_metrics["training/avg_effective_len"],
                        avg_sample_seq_len=train_metrics["training/avg_sample_seq_len"],
                        save_checkpoint_path=save_checkpoint_path,
                    )
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch+1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            if args.train.global_rank == 0:
                helper.save_step2token(
                    os.path.join(args.train.save_checkpoint_path, 'step2token.txt'),
                    consumed_tokens=train_metrics["consume_tokens(B)"],
                    global_step=global_step,
                    avg_effective_len=train_metrics["training/avg_effective_len"],
                    avg_sample_seq_len=train_metrics["training/avg_sample_seq_len"],
                    save_checkpoint_path=save_checkpoint_path,
                )
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    torch.cuda.synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0:
        if args.train.save_hf_weights and save_checkpoint_path is not None:
            hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
            model_state_dict = ckpt_to_state_dict(
                save_checkpoint_path=save_checkpoint_path,
                output_dir=args.train.output_dir,
                ckpt_manager=args.train.ckpt_manager,
            )
            save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
            logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
