# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Callable, Dict, List, Literal, Optional

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import Dataset, IterableDataset

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.dist_utils import main_process_first

import glob
import json
import webdataset as wds
import itertools
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # dropout truncated images to avoid inconsistency

logger = logging.get_logger(__name__)


class DummyDataset(Dataset):
    def __init__(self, size: int, seq_length: int):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 32768

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        labels = input_ids.clone()
        return [{"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}]


class MappingDataset(Dataset):
    """
    Mapping dataset.
    Args:
        data (Dataset): Dataset
        transform (Optional[Callable]): transform function
    """

    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        if self._transform is not None:
            return self._transform(self._data[index])
        else:
            return self._data[index]


class IterativeDataset_wds(IterableDataset):
    """
    Iterative dataset.
    Args:
        data (Dataset): Dataset
        transform (Optional[Callable]): transform function
    """

    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __iter__(self):
        for sample in self._data:
            # drop out samples without images
            if 'jpg' in sample.keys():
                if type(sample['jpg']) is dict:
                    if len(sample['jpg']['bytes']) == 0:
                        continue
                else:
                    if len(sample['jpg']) == 0:
                        continue
            elif 'image' in sample.keys():
                if type(sample['image']) is dict:
                    if len(sample['image']['bytes']) == 0:
                        continue
                else:
                    if len(sample['image']) == 0:
                        continue
                sample['jpg'] = sample['image']
                # sample.pop('image')
            else:
                continue
            if self._transform is not None:
                try:
                    yield self._transform(sample)
                except Exception as e:
                    logger.warning(f"Failed to transform sample: {sample.get('__url__')} - {sample.get('__key__')}, skipping.")
                    continue
            else:
                yield sample

    def load_state_dict(self, state_dict):
        self._data.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {"dataset": self._data.state_dict()}

    def set_epoch(self, epoch: int):
        if hasattr(self._data, "set_epoch"):
            self._data.set_epoch(epoch)

class IterativeDataset(IterableDataset):
    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __iter__(self):
        for sample in self._data:
            if self._transform is not None:
                try:
                    yield self._transform(sample)
                except Exception as e:
                    logger.warning(f"Failed to transform sample: {sample.get('__url__')} - {sample.get('__key__')}, skipping.")
                    continue
            else:
                yield sample

    def load_state_dict(self, state_dict):
        self._data.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {"dataset": self._data.state_dict()}

    def set_epoch(self, epoch: int):
        self._data.set_epoch(epoch)

def build_dummy_dataset(size: int, max_seq_len: int) -> "Dataset":
    return DummyDataset(size=size, seq_length=max_seq_len)


def build_mapping_dataset(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
) -> "Dataset":
    """
    Build mapping dataset.
    Args:
        data_path (str): data path
        transform (Optional[Callable]): transform function
        namespace (Literal["train", "test"]): dataset namespace
    Returns:
        Dataset: mapping dataset
    """
    data_files = []
    data_paths = data_path.split(",")
    for data_path in data_paths:
        if os.path.isdir(data_path):
            data_files.extend([os.path.join(data_path, fn) for fn in os.listdir(data_path)])
        elif os.path.isfile(data_path):
            data_files.append(data_files)
        else:
            raise FileNotFoundError(f"Dataset {data_path} not exists.")

    file_extenstion = os.path.splitext(data_files[0])[-1][1:]
    if file_extenstion not in ["parquet", "jsonl", "json", "csv", "arrow"]:
        raise ValueError(f"{file_extenstion} files are not supported.")

    file_extenstion = "json" if file_extenstion == "jsonl" else file_extenstion
    with main_process_first():
        dataset = load_dataset(file_extenstion, data_files=data_files, split=namespace)

    return MappingDataset(data=dataset, transform=transform)


def build_iterative_dataset(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    seed: int = 42,
) -> "IterableDataset":
    """ "
    Build iterative dataset.
    Args:
        data_path (str): data path
        transform (Optional[Callable]): transform function
        namespace (Literal["train", "test"]): dataset namespace
        seed (int): random seed
    Returns:
        IterableDataset: iterative dataset
    """

    data_files = []
    data_paths = data_path.split(",")
    for data_path in data_paths:
        if os.path.isdir(data_path):
            data_files.extend([os.path.join(data_path, fn) for fn in os.listdir(data_path)])
        elif os.path.isfile(data_path):
            data_files.append(data_files)
        else:
            raise FileNotFoundError(f"Dataset {data_path} not exists.")

    parallel_state = get_parallel_state()
    file_extenstion = os.path.splitext(data_files[0])[-1][1:]
    if file_extenstion not in ["parquet", "jsonl", "json", "csv", "arrow"]:
        raise ValueError(f"{file_extenstion} files are not supported.")

    file_extenstion = "json" if file_extenstion == "jsonl" else file_extenstion
    dataset = load_dataset(file_extenstion, data_files=data_files, split=namespace, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    dataset = split_dataset_by_node(dataset, parallel_state.dp_rank, parallel_state.dp_size)

    return IterativeDataset(dataset, transform)

def build_iterative_webdataset(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    seed: int = 42,
) -> "IterableDataset":
    """
    Build iterative dataset.
    Args:
        data_path (str): data path
        transform (Optional[Callable]): transform function
        namespace (Literal["train", "test"]): dataset namespace
        seed (int): random seed
    Returns:
        IterableDataset: iterative dataset
    """
    data_files = wds.shardlists.expand_urls(data_path) # expand input wds str
    # data_files = glob.glob(data_path)
    parallel_state = get_parallel_state()
    dataset = load_dataset("webdataset", data_files={"train": data_files}, split=namespace, streaming=True).decode(False) # consider using webdataset pkg for stable implementation
    dataset = dataset.shuffle(seed=seed, buffer_size=20_000)
    dataset = split_dataset_by_node(dataset, parallel_state.dp_rank, parallel_state.dp_size)

    return IterativeDataset(dataset, transform)

def build_iterative_webdataset_wdsapi(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    pure_mm: bool = True,
    seed: int = 42,
) -> "IterableDataset":
    # todo: support load all tar files in parent/grandparent dir 
    data_path = os.path.join(data_path, "**/*.tar")
    data_files = glob.glob(data_path, recursive=True)
    
    logger.info(f"Loading {len(data_files)} full multimodal webdataset tars.")
    parallel_state = get_parallel_state()
    dataset = load_dataset("webdataset", data_files={"train": data_files}, split=namespace, streaming=True).decode(False)
    dataset = dataset.shuffle(seed=seed, buffer_size=100_000)
    dataset = split_dataset_by_node(dataset, parallel_state.dp_rank, parallel_state.dp_size)

    return IterativeVisionDataset(dataset, transform)

def build_mixed_iterative_wdsapi(
    mix_data_path: list[str],
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    pure_mm: bool = True,
    seed: int = 42,
):
    total_data_files = []
    for data_path in mix_data_path:
        data_path = os.path.join(data_path.strip(), "**/*.tar")
        data_files = glob.glob(data_path, recursive=True)
        total_data_files = total_data_files + data_files
    logger.info(f"Loading {len(total_data_files)} webdataset tars from {len(mix_data_path)} data sources.")

    random.Random(seed).shuffle(total_data_files)

    dataset = wds.WebDataset(total_data_files, seed = seed, resampled=False, shardshuffle=5_000, 
                             nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker).shuffle(size=20_000).repeat()
    return IterativeDataset_wds(dataset, transform)
