<div align="center">

<h1>Let ViT Speak: Generative Language-Image Pre-training</h1>

<b>Yan Fang</b><sup>1,2,&#42;</sup> · <b><a href="https://mc-lan.github.io">Mengcheng Lan</a></b><sup>2,3,&#42;</sup> · <b><a href="https://speedinghzl.github.io">Zilong Huang</a></b><sup>2,&dagger;</sup> · <b>Weixian Lei</b><sup>2</sup> · <b><a href="https://yunqing-me.github.io">Yunqing Zhao</a></b><sup>2</sup> · <b><a href="https://y-zhong.info">Yujie Zhong</a></b><sup>2</sup> · <b><a href="https://yingchen001.github.io">Yingchen Yu</a></b><sup>2</sup> · <b><a href="https://qi-she.net/">Qi She</a></b><sup>2</sup> · <b>Yao Zhao</b><sup>1</sup> · <b><a href="https://weiyc.github.io">Yunchao Wei</a></b><sup>1,&dagger;</sup>

Beijing Jiaotong University<sup>1</sup> & ByteDance<sup>2</sup> & Nanyang Technological University<sup>3</sup>

<a href="https://yanfangcs.github.io/vitspeak"><img src="https://img.shields.io/badge/Github-Page-blue" alt="Home Page"></a>
<a href="https://arxiv.org/abs/2605.00809"><img src="https://img.shields.io/badge/Paper-Arxiv-red" alt="Paper Arxiv"></a>
<a href="https://huggingface.co/collections/YanFang/genlip"><img src="https://img.shields.io/badge/Model-HuggingFace-orange" alt="Model HuggingFace"></a>
</div>

**TL;DR:** **GenLIP -- lets ViT speak.** We show that a strong MLLM vision encoder can be pretrained with just **one Transformer** and **one autoregressive language modeling objective** -- no contrastive loss, no dual-tower architecture, and no extra text decoder. Despite its simplicity, GenLIP scales effectively and performs well as a vision encoder in MLLMs, with particularly strong gains on Doc & OCR tasks.

<div align='center'>
  <img src="assets/teaser.png" alt="teaser" style="height: 200px; width: auto;">
</div>

---

## Table of Contents
- [News](#news)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Datasets](#datasets)
  - [Configuration](#configuration)
- [Training](#training)
- [Model Checkpoints](#model-checkpoints)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## News
- 2025-05-03: Code released. [✔]

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/YanFangCS/GenLIP
cd GenLIP

# Install dependencies
pip install -r requirements.txt
pip install -e .   # install veomni from this repo
```

> **Note:** If you are using PyTorch >= 2.6.0, you need to install ByteCheckpoint manually:
>
> ```bash
> git clone https://github.com/ByteDance-Seed/ByteCheckpoint.git
> cd ByteCheckpoint
> # Modify the torch version assert statement in bytecheckpoint/checkpointer/fsdp_checkpointer.py#L232-L234 to support torch >= 2.6.0
> # assert "2.1.0" <= torch.__version__.strip()
> pip install -e .
> ```

<!-- ### Datasets

#### Data Source

We use several caption datasets during pretraining:

**Stage 1:**
- [Recap-DataComp-1B](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B)

**Stage 2:**
- [Infinity-MM](https://huggingface.co/datasets/BAAI/Infinity-MM) (stage1 subset)
- [BLIP3o-Pretrain-Long-Caption](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption)

**Optional for Stage 2:**
- [CapRL-2M](https://huggingface.co/datasets/internlm/CapRL-2M)
- [PLM-Image-Auto](https://huggingface.co/datasets/facebook/PLM-Image-Auto) (caption subset only)

For Stage 1, training GenLIP with 1B seen samples is sufficient to obtain a strong vision encoder.
For Stage 2, training GenLIP with Infinity-MM and BLIP3o-Long-Caption using NaViT is sufficient.
Training with the two additional datasets (CapRL and PLM-Image-Auto) does not bring further performance gains, but we list them here as potential alternatives. -->

### Datasets

#### Data Format

<!-- All datasets need to be downloaded and processed into suitable formats for pretraining. Please ensure your preprocessing function can correctly consume your data. -->
Please ensure your preprocessing function can correctly consume your data. 

Below are example data formats in our pretraining:

```python
# Stage 1 caption data
# sample keys: ['__key__', '__url__', 'jpg', '__local_path__', 'json']
json_content = {
  'caption': 'A modern coffee machine with a digital display and two white coffee cups filled with coffee is shown. The machine has a stainless steel finish and is accompanied by a milk frothing pitcher with a white liquid inside. The coffee machine is placed on a surface with a white background.'
}

# Stage 2 caption data
# sample keys: ['__key__', '__url__', 'jpg', '__local_path__', 'json']
json_content = {
  'conversation': [
    {
      'from': 'user',
      'value': '<image>Describe this image in detail.'
    },
    {
      'from': 'assistant',
      'value': 'The image depicts a serene waterfront scene with calm, slightly rippled water in the foreground...'
    }
  ]
}
```

You can also process the datasets into other formats as needed. To ensure training runs smoothly, check and modify the `process_sample` function implementation to match your data format.

### Configuration

We provide three model configurations in `configs/model_configs/genlip/`:
- `genlip_l16_224.json`
- `genlip_so16_224.json`
- `genlip_g16_224.json`

Along with corresponding training configurations in `configs/pretrain/genlip/`:
- `stage1/train_genlip_*_recap.yaml`
- `stage2/train_genlip_*_navit.yaml`

You may need to modify `model.config_path` in the YAML config files to point to the correct model configuration.

**Remember to update the dataset paths in the config files before starting training.**

## Training

A training script is provided in `jobs/train.sh`. You can start training with:

```bash
bash jobs/train.sh <main_func> <train_config>

# Stage 1 example:
bash jobs/train.sh tasks/train_genlip_stage1.py configs/pretrain/genlip/stage1/train_genlip_so16_224_recap.yaml

# Stage 2 example:
bash jobs/train.sh tasks/train_genlip_navit.py configs/pretrain/genlip/stage2/train_genlip_so16_navit.yaml
```

- `<main_func>`: the training script to execute (e.g., `tasks/train_genlip_stage1.py` for Stage 1, `tasks/train_genlip_navit.py` for Stage 2).
- `<train_config>`: the training configuration file to use.

All you need to do is set the paths and appropriate hyperparameters in the config files, then launch the script and wait for training to complete.

For **multi-node training**, we also provide `jobs/train_multinode.sh` and `jobs/train_slurm_multinode.sh`. You can modify them to fit your cluster setup and launch distributed training across multiple nodes.


## Model Checkpoints

The pretrained models are available on [HuggingFace](https://huggingface.co/collections/YanFang/genlip).

## Acknowledgments

Our codebase is built upon:
- [VeOmni](https://github.com/ByteDance-Seed/VeOmni): A simple and high-performance multi-modal model training framework developed by the ByteDance Seed team.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this project helpful, please give us a star and cite our [paper](https://arxiv.org/abs/2605.00809):

```bibtex
@article{fang2026letvitspeakgenerative,
  title={Let ViT Speak: Generative Language-Image Pre-training}, 
  author={Yan Fang and Mengcheng Lan and Zilong Huang and Weixian Lei and Yunqing Zhao and Yujie Zhong and Yingchen Yu and Qi She and Yao Zhao and Yunchao Wei},
  journal={arXiv preprint arXiv:2605.00809},
  year={2026}
}
```