# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Implement base data transfer protocol between any two functions, modules.
We can subclass Protocol to define more detailed batch info with specific keys
"""

import contextlib
import copy
import logging
import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import ray
import tensordict
import torch
import torch.distributed
from packaging import version
from packaging.version import parse as parse_version
from tensordict import TensorDict
from torch.utils.data import DataLoader

from verl.utils.device import get_device_id, get_torch_device
from verl.utils.py_functional import union_two_dict
from verl.utils.torch_functional import allgather_dict_tensors

__all__ = ["DataProto", "union_tensor_dict"]

with contextlib.suppress(Exception):
    tensordict.set_lazy_legacy(False).set()
    if parse_version(tensordict.__version__) < parse_version("0.10.0"):
        tensordict.set_list_to_stack(True).set()


class _DataProtoConfigMeta(type):
    _config = {}

    auto_padding_key = "_verl_auto_padding"

    @property
    def auto_padding(cls):
        enabled_by_env = os.getenv("VERL_AUTO_PADDING", "FALSE").upper() in ["TRUE", "1"]
        return enabled_by_env or cls._config.get(cls.auto_padding_key, False)

    @auto_padding.setter
    def auto_padding(cls, enabled: bool):
        assert isinstance(enabled, bool), f"enabled must be a boolean, got {enabled} as {type(enabled)}"
        cls._config[cls.auto_padding_key] = enabled


class DataProtoConfig(metaclass=_DataProtoConfigMeta):
    pass


_padding_size_key = "_padding_size_key_x123d"


def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int):
    """Pad a DataProto to size divisible by size_divisor

    Args:
        size_divisor (int): size divisor

    Returns:
        data: (DataProto): the padded DataProto
        pad_size (int)
    """
    assert isinstance(data, DataProto), "data must be a DataProto"
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        data_padded = DataProto.concat([data] + padding_protos)
    else:
        if len(data) == 0:
            logging.warning("padding a DataProto with no item, no changed made")
        pad_size = 0
        data_padded = data
    return data_padded, pad_size


def unpad_dataproto(data: "DataProto", pad_size):
    """Unpad the data proto with pad_size. i.e. `data[:-pad_size]`"""
    if pad_size != 0:
        data = data[:-pad_size]
    return data


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """Union two tensordicts."""
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, (
        f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
    )
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2[key]
        else:
            assert tensor_dict1[key].equal(tensor_dict2[key]), (
                f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
            )

    return tensor_dict1


def _array_equal(array1: np.ndarray, array2: np.ndarray, visited: set[int]) -> bool:
    """
    Recursively compares two NumPy arrays for strict equality, with special
    handling for object-dtype arrays, NaN values, and circular references.
    This function assumes that the two arguments provided are NumPy arrays.

    Args:
        array1: The first NumPy array.
        array2: The second NumPy array.

    Returns:
        True if the arrays' dtypes, shapes, and all elements are equal.
    """
    # Check dtype and shape first, as this is the fastest failure path.
    if array1.dtype != array2.dtype or array1.shape != array2.shape:
        return False

    # For non-object dtypes, use NumPy's implementation with equal_nan=True.
    if array1.dtype != "object":
        return np.array_equal(array1, array2, equal_nan=True)

    # For object-dtype arrays, we must recursively compare each element.
    # We delegate to _deep_equal to handle elements, as they could be any
    # type, including other nested arrays or NaNs.
    return all(_deep_equal(x, y, visited) for x, y in zip(array1.flat, array2.flat, strict=False))


def _deep_equal(a: Any, b: Any, visited: set[int]) -> bool:
    """
    Recursively performs a deep comparison between two Python objects.
    - Handles NaN values correctly (NaN == NaN evaluates to True).
    - Handling circular references.
    - Dispatches to _array_equal if both objects are NumPy arrays.
    - Otherwise, uses standard '==' comparison.
    """
    if type(a) is not type(b):
        return False

    # If we have seen this object ID before on this path, it's a cycle.
    # Since we already know the types match, we can safely assume this part
    # of the structure is equal.
    obj_id = id(a)
    if obj_id in visited:
        return True

    visited.add(obj_id)

    # Perform the specific comparison based on type
    result = False
    if isinstance(a, float) and math.isnan(a) and math.isnan(b):
        result = True
    elif isinstance(a, np.ndarray):
        # We know b is also an ndarray due to the initial type check
        result = _array_equal(a, b, visited)
    else:
        # Standard equality for all other types
        result = a == b

    # Clean up the visited set on the way out of the recursion
    visited.remove(obj_id)
    return result


def union_numpy_dict(tensor_dict1: dict[str, np.ndarray], tensor_dict2: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    for key, val in tensor_dict2.items():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            # to properly deal with nan and object type
            assert _deep_equal(tensor_dict1[key], tensor_dict2[key], visited=set()), (
                f"`{key}` in tensor_dict1 and tensor_dict2 are not the same object."
            )
        tensor_dict1[key] = val

    return tensor_dict1


def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            assert key in output
            output[key].append(item)
    return output


def fold_batch_dim(data: "DataProto", new_batch_size):
    """
    Fold a batch dim from [bsz, xxx] into [new_bsz, bsz // new_bsz, xxx]
    """
    batch_size = data.batch.batch_size[0]

    assert batch_size % new_batch_size == 0

    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch

    tensor = tensor.view(new_batch_size, -1)
    tensor.auto_batch_size_(batch_dims=1)

    for key, val in non_tensor.items():
        non_tensor[key] = np.reshape(val, newshape=(new_batch_size, -1, *val.shape[1:]))

    return type(data)(batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info)


def unfold_batch_dim(data: "DataProto", batch_dims=2):
    """
    Unfold the first n dims as new batch dim
    """
    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch
    tensor.auto_batch_size_(batch_dims=batch_dims)
    tensor = tensor.view(-1)

    batch_size = tensor.batch_size[0]

    non_tensor_new = {}

    for key, val in non_tensor.items():
        non_tensor_new[key] = np.reshape(val, newshape=(batch_size, *val.shape[batch_dims:]))

    return type(data)(batch=tensor, non_tensor_batch=non_tensor_new, meta_info=data.meta_info)


def serialize_single_tensor(obj: torch.Tensor) -> tuple[str, tuple[int, ...], int | memoryview]:
    data = obj.flatten().contiguous().view(torch.uint8).numpy()
    dtype = str(obj.dtype).removeprefix("torch.")
    return dtype, obj.shape, data


def serialize_tensordict(batch: TensorDict) -> tuple[tuple[int, ...], Optional[str], dict[str, tuple[str, Any]]]:
    encoded_items: dict[str, tuple[Any]] = {}
    for k, v in batch.items():
        if not v.is_nested:
            encoded_items[k] = serialize_single_tensor(v)
        else:
            layout = str(v.layout).removeprefix("torch.")
            data = [serialize_single_tensor(tensor) for tensor in v.unbind()]
            encoded_items[k] = (layout, data)

    batch_size = tuple(batch.batch_size)
    device = str(batch.device) if batch.device is not None else None
    return batch_size, device, encoded_items


def deserialize_single_tensor(arr: Any) -> torch.Tensor:
    dtype, shape, data = arr

    torch_dtype = getattr(torch, dtype)
    assert isinstance(torch_dtype, torch.dtype)

    buffer = bytearray(data)
    # Create uint8 array
    arr = torch.frombuffer(buffer, dtype=torch.uint8)
    # Convert back to proper shape & type
    return arr.view(torch_dtype).view(shape)


def deserialize_tensordict(arr: Any) -> TensorDict:
    batch_size, device, encoded_items = arr
    decoded_items: dict[str, Any] = {}

    for k, v in encoded_items.items():
        if len(v) == 3:
            # decode single tensor
            decoded_items[k] = deserialize_single_tensor(v)
        elif len(v) == 2:
            # decode nested tensor
            layout, data = v
            torch_layout = getattr(torch, layout)
            decoded_items[k] = torch.nested.as_nested_tensor(
                [deserialize_single_tensor(tensor) for tensor in data], layout=torch_layout
            )
        else:
            raise ValueError(f"Invalid tensor encoding format, expected length 2 or 3, got {len(v)}")

    return TensorDict(source=decoded_items, batch_size=batch_size, device=device)


def collate_fn(x: list["DataProtoItem"]):
    batch = []
    non_tensor_batch = []
    for data in x:
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)
    batch = torch.stack(batch).contiguous()
    non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
    for key, val in non_tensor_batch.items():
        non_tensor_batch[key] = np.array(val, dtype=object)
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


@dataclass
class DataProtoItem:
    # TODO(zhangchi.usc1992) add consistency check
    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)


@dataclass
class DataProto:
    """
    A DataProto is a data structure that aims to provide a standard protocol for data exchange between functions.
    It contains a batch (TensorDict) and a meta_info (Dict). The batch is a TensorDict https://pytorch.org/tensordict/.
    TensorDict allows you to manipulate a dictionary of Tensors like a single Tensor. Ideally, the tensors with the
    same batch size should be put inside batch.
    """

    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)

    def __post_init__(self):
        # perform necessary checking
        self.check_consistency()

    def __len__(self):
        if self.batch is not None:
            return self.batch.batch_size[0]
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            random_key = list(self.non_tensor_batch.keys())[0]
            return self.non_tensor_batch[random_key].shape[0]
        else:
            return 0

    def __getitem__(self, item):
        """
        Enhanced indexing for DataProto objects.

        Args:
            item: Can be one of:
                - int: A single index
                - slice: A slice object (start:stop:step)
                - list: A list of indices
                - numpy.ndarray: An array of indices
                - torch.Tensor: A tensor of indices

        Returns:
            DataProto: For all indexing types except single integers
            DataProtoItem: Only for single integer indices
        """
        # Case 1: Slice object - use the slice method
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)

        # Case 2: List, numpy array, or torch tensor - use sel_idxs
        elif isinstance(item, list | np.ndarray | torch.Tensor):
            return self.select_idxs(item)

        # Case 3: Single integer - return DataProtoItem for backward compatibility
        elif isinstance(item, int | np.integer):
            tensor_data = self.batch[item] if self.batch is not None else None
            non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()}
            return DataProtoItem(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

        # # Case 4: Unsupported type
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported")

    def __getstate__(self):
        if version.parse(tensordict.__version__) >= version.parse("0.5.0") and self.batch is not None:
            # Check if batch is empty to avoid torch.cat error in consolidate
            if len(self.batch.keys()) > 0:
                batch = self.batch.contiguous().consolidate()
            else:
                batch = self.batch
        else:
            batch = self.batch

        if os.getenv("VERL_DATAPROTO_SERIALIZATION_METHOD") == "numpy":
            if batch is not None:
                batch = serialize_tensordict(self.batch)

            return (
                batch,
                self.non_tensor_batch,
                self.meta_info,
            )
        else:
            import io

            buffer = io.BytesIO()
            torch.save(batch, buffer)
            buffer_bytes = buffer.getvalue()
            return buffer_bytes, self.non_tensor_batch, self.meta_info

    def __setstate__(self, data):
        batch_deserialized_bytes, non_tensor_batch, meta_info = data

        if os.getenv("VERL_DATAPROTO_SERIALIZATION_METHOD") == "numpy":
            if batch_deserialized_bytes is not None:
                self.batch = deserialize_tensordict(batch_deserialized_bytes)
            else:
                self.batch = None
        else:
            import io

            batch_deserialized = io.BytesIO(initial_bytes=batch_deserialized_bytes)
            batch = torch.load(
                batch_deserialized,
                weights_only=False,
                map_location="cpu" if not get_torch_device().is_available() else None,
            )
            self.batch = batch

        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    def save_to_disk(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filepath) -> "DataProto":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            return data

    def print_size(self, prefix=""):
        size_of_tensordict = 0
        if self.batch is not None:
            for _, tensor in self.batch.items():
                size_of_tensordict += tensor.element_size() * tensor.numel()
        size_of_numpy_array = 0
        for _, numpy_array in self.non_tensor_batch.items():
            size_of_numpy_array += numpy_array.nbytes

        size_of_numpy_array /= 1024**3
        size_of_tensordict /= 1024**3

        message = f"Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB"

        if prefix:
            message = f"{prefix}, " + message
        print(message)

    def check_consistency(self):
        """Check the consistency of the DataProto. Mainly for batch and non_tensor_batch
        We expose this function as a public one so that user can call themselves directly
        """
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"

        if self.non_tensor_batch is not None:
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray)

        if self.batch is not None and self.non_tensor_batch is not None and len(self.non_tensor_batch) != 0:
            # TODO: we can actually lift this restriction if needed
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1 when non_tensor_batch is not empty."

            batch_size = self.batch.batch_size[0]
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray), (
                    f"data in the non_tensor_batch must be a numpy.array with dtype=object, but for "
                    f"{key=}, got {type(val)=}"
                )
                assert val.shape[0] == batch_size, (
                    f"key {key} length {len(val)} is not equal to batch size {batch_size}"
                )

    @classmethod
    def from_single_dict(cls, data: dict[str, torch.Tensor | np.ndarray], meta_info=None, auto_padding=False):
        """Create a DataProto from a dict of tensors and non_tensors"""
        tensors = {}
        non_tensors = {}

        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f"Unsupported type in data {type(val)}")

        return cls.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info, auto_padding=auto_padding)

    @classmethod
    def from_dict(
        cls,
        tensors: Optional[dict[str, torch.Tensor]] = None,
        non_tensors=None,
        meta_info=None,
        num_batch_dims=1,
        auto_padding=False,
    ):
        """Create a DataProto from a dict of tensors. This assumes that
        1. All the tensor in tensors have the same dim0
        2. Only dim0 is the batch dim
        """

        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if non_tensors is not None:
            assert num_batch_dims == 1, "only support num_batch_dims=1 when non_tensors is not None."

        if tensors is None:
            tensors = {}
        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}

        assert isinstance(non_tensors, dict)

        # get and check batch size
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert batch_size == current_batch, (
                    f"Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. "
                    f"Got {pivot_key} has {batch_size}, {key} has {current_batch}"
                )

        for key, val in non_tensors.items():
            if not isinstance(val, np.ndarray):
                non_tensors[key] = np.array(val, dtype=object)

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size) if tensors else None
        if auto_padding:
            meta_info[DataProtoConfig.auto_padding_key] = True
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    @classmethod
    def from_tensordict(
        cls,
        tensor_dict: TensorDict = None,
        meta_info=None,
        num_batch_dims=1,
    ):
        """Create a DataProto from a TensorDict. This assumes that
        1. All the tensor in tensor_dict have the same dim0
        2. Only dim0 is the batch dim
        """
        assert version.parse(tensordict.__version__) >= version.parse("0.10.0"), (
            "Build DataProto from TensorDict at least requires tensordict version 0.10.0"
        )
        from tensordict import NonTensorData, NonTensorStack

        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if not all(isinstance(val, torch.Tensor) for val in tensor_dict.values()):
            assert num_batch_dims == 1, "only support num_batch_dims=1 when tensor_dict contains non tensor data."

        if meta_info is None:
            meta_info = {}
        batch = {}
        non_tensor_batch = {}
        batch_size = None
        for key, val in tensor_dict.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val
                if batch_size is None:
                    batch_size = val.shape[:num_batch_dims]
            elif isinstance(val, NonTensorStack):
                non_tensor_batch[key] = np.array([elem.data for elem in val], dtype=object)
            elif isinstance(val, NonTensorData):
                meta_info[key] = val.data

        return cls(
            batch=TensorDict(batch, batch_size=batch_size),
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )

    def to(self, device) -> "DataProto":
        """move the batch to device

        Args:
            device (torch.device, str): torch device

        Returns:
            DataProto: the current DataProto

        """
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self

    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> "DataProto":
        """Select a subset of the DataProto via batch_keys and meta_info_keys

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to select
            meta_info_keys (list, optional): a list of keys indicating the meta info to select

        Returns:
            DataProto: the DataProto with the selected batch_keys and meta_info_keys
        """
        # TODO (zhangchi.usc1992) whether to copy
        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self.batch.select(*batch_keys)
        else:
            sub_batch = self.batch

        if non_tensor_batch_keys is not None:
            non_tensor_batch = {key: val for key, val in self.non_tensor_batch.items() if key in non_tensor_batch_keys}
        else:
            non_tensor_batch = self.non_tensor_batch

        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        if meta_info_keys is not None:
            sub_meta_info = {key: val for key, val in self.meta_info.items() if key in meta_info_keys}
        else:
            sub_meta_info = self.meta_info

        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)

        return type(self)(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    def select_idxs(self, idxs):
        """
        Select specific indices from the DataProto.

        Args:
            idxs (torch.Tensor or numpy.ndarray or list): Indices to select

        Returns:
            DataProto: A new DataProto containing only the selected indices
        """
        if isinstance(idxs, list):
            idxs = torch.tensor(idxs)
            if idxs.dtype != torch.bool:
                idxs = idxs.type(torch.int32)

        if isinstance(idxs, np.ndarray):
            idxs_np = idxs
            idxs_torch = torch.from_numpy(idxs)
        else:  # torch.Tensor
            idxs_torch = idxs
            idxs_np = idxs.detach().cpu().numpy()

        batch_size = int(idxs_np.sum()) if idxs_np.dtype == bool else idxs_np.shape[0]

        if self.batch is not None:
            # Use TensorDict's built-in indexing capabilities
            selected_batch = TensorDict(
                source={key: tensor[idxs_torch] for key, tensor in self.batch.items()},
                batch_size=(batch_size,),
                device=self.batch.device,
            )
        else:
            selected_batch = None

        selected_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            selected_non_tensor[key] = val[idxs_np]

        return type(self)(batch=selected_batch, non_tensor_batch=selected_non_tensor, meta_info=self.meta_info)

    def slice(self, start=None, end=None, step=None):
        """
        Slice the DataProto and return a new DataProto object.
        This is an improved version of direct slicing which returns a DataProtoItem.

        Args:
            start (int, optional): Start index. Defaults to None (start from beginning).
            end (int, optional): End index (exclusive). Defaults to None (go to end).
            step (int, optional): Step size. Defaults to None (step=1).

        Returns:
            DataProto: A new DataProto containing the sliced data

        Examples:
            # Using the slice method directly
            sliced_data = data_proto.slice(10, 20)

            # Using enhanced indexing (returns DataProto)
            sliced_data = data_proto[10:20]
            sliced_data = data_proto[::2]  # Every other element

            # Using list indexing (returns DataProto)
            indices = [1, 5, 10]
            selected_data = data_proto[indices]

            # Single index still returns DataProtoItem
            single_item = data_proto[5]
        """
        # Create a slice object
        slice_obj = slice(start, end, step)

        # Handle the batch data
        if self.batch is not None:
            # Use TensorDict's built-in slicing capabilities
            sliced_batch = self.batch[slice_obj]
        else:
            sliced_batch = None

        # Handle the non-tensor batch data
        sliced_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            sliced_non_tensor[key] = val[slice_obj]

        # Return a new DataProto object
        return type(self)(batch=sliced_batch, non_tensor_batch=sliced_non_tensor, meta_info=self.meta_info)

    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> "DataProto":
        """Pop a subset of the DataProto via `batch_keys` and `meta_info_keys`

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to pop
            meta_info_keys (list, optional): a list of keys indicating the meta info to pop

        Returns:
            DataProto: the DataProto with the poped batch_keys and meta_info_keys
        """
        if batch_keys is None:
            batch_keys = []
        if meta_info_keys is None:
            meta_info_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []

        tensors = {}
        # tensor batch
        for key in batch_keys:
            assert key in self.batch.keys()
            tensors[key] = self.batch.pop(key)
        non_tensors = {}
        # non tensor batch
        for key in non_tensor_batch_keys:
            assert key in self.non_tensor_batch.keys()
            non_tensors[key] = self.non_tensor_batch.pop(key)
        meta_info = {}
        for key in meta_info_keys:
            assert key in self.meta_info.keys()
            meta_info[key] = self.meta_info.pop(key)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def rename(self, old_keys=None, new_keys=None) -> "DataProto":
        """
        Note that this function only rename the key in the batch
        """

        def validate_input(keys):
            if keys is not None:
                if isinstance(keys, str):
                    keys = [keys]
                elif isinstance(keys, list):
                    pass
                else:
                    raise TypeError(f"keys must be a list or a string, but got {type(keys)}")
            return keys

        old_keys = validate_input(old_keys)
        new_keys = validate_input(new_keys)

        if len(new_keys) != len(old_keys):
            raise ValueError(
                f"new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}"
            )

        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        return self

    def union(self, other: "DataProto") -> "DataProto":
        """Union with another DataProto. Union batch and meta_info separately.
        Throw an error if

        - there are conflict keys in batch and they are not equal
        - the batch size of two data batch is not the same
        - there are conflict keys in meta_info and they are not the same.

        Args:
            other (DataProto): another DataProto to union

        Returns:
            DataProto: the DataProto after union
        """
        self.batch = union_tensor_dict(self.batch, other.batch)
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        r"""Make an iterator from the DataProto. This is built upon that TensorDict can be used as a normal Pytorch
        dataset. See https://pytorch.org/tensordict/stable/tutorials/data_fashion for more details.


        Args:
            mini_batch_size (int): mini-batch size when iterating the dataset. We require that
                ``batch.batch_size[0] % mini_batch_size == 0``.
            epochs (int): number of epochs when iterating the dataset.
            dataloader_kwargs (Any): internally, it returns a DataLoader over the batch. The
                dataloader_kwargs is the kwargs passed to the DataLoader.

        Returns:
            Iterator: an iterator that yields a mini-batch data at a time. The total number of iteration
                steps is ``self.batch.batch_size * epochs // mini_batch_size``
        """
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        # we can directly create a dataloader from TensorDict
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        assert isinstance(dataloader_kwargs, dict)
        train_dataloader = DataLoader(
            dataset=self, batch_size=mini_batch_size, collate_fn=collate_fn, generator=generator, **dataloader_kwargs
        )

        def get_data():
            for _ in range(epochs):
                for d in train_dataloader:
                    d.meta_info = self.meta_info
                    yield d

        return iter(get_data())

    def is_padding_enabled(self):
        """
        Check if padding is enabled for the DataProto.
        Returns:
            bool: True if padding is enabled, False otherwise.
        """
        dataproto_specific_padding = self.meta_info.get(DataProtoConfig.auto_padding_key, False)
        return dataproto_specific_padding or DataProtoConfig.auto_padding

    def padding(self, padding_size, padding_candidate=""):
        """Pad the DataProto by concating with padding_candidate.repeat(padding_size)

        Args:
            padding_size (int): the number of repeated padding_candidate
            padding_candidate: the item to be repeated and appended to the DataProto, only supporting ["first", "last"]
        """
        if padding_size == 0:
            return
        padding_candidate = self.select_idxs([0 if padding_candidate == "first" else len(self) - 1])
        padding_part = padding_candidate.repeat(padding_size)
        padded_dp = DataProto.concat([self, padding_part])
        self.batch = padded_dp.batch
        self.non_tensor_batch = padded_dp.non_tensor_batch

    def chunk(self, chunks: int) -> list["DataProto"]:
        """Split the batch among dim=0 into chunks. The meta_info is passed to each DataProto after split.

        Args:
            chunks (int): the number of chunks to split on dim=0

        Returns:
            List[DataProto]: a list of DataProto after splitting
        """
        if not self.is_padding_enabled():
            assert len(self) % chunks == 0, (
                f"only support equal chunk. Got size of DataProto {len(self)} and chunk {chunks}."
            )

        bsz_in_batch = None
        if self.batch is not None:
            batch_lst = self.batch.chunk(chunks=chunks, dim=0)
            bsz_in_batch = np.array([batch.batch_size[0] for batch in batch_lst])
            chunk_indices = np.cumsum(bsz_in_batch)[:-1]
        else:
            batch_lst = [None for _ in range(chunks)]

        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            if bsz_in_batch is not None:
                non_tensor_lst = np.array_split(val, chunk_indices.tolist())
            else:
                non_tensor_lst = np.array_split(val, chunks)
            assert len(non_tensor_lst) == chunks
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        output = []
        for i in range(chunks):
            output.append(
                type(self)(batch=batch_lst[i], non_tensor_batch=non_tensor_batch_lst[i], meta_info=self.meta_info)
            )

        return output

    def split(self, split_size: int) -> list["DataProto"]:
        """Split the batch among dim=0 into chunks. The meta_info is passed to each DataProto after split.

        Args:
            split_size (int): the size of each split

        Returns:
            List[DataProto]: a list of DataProto after splitting
        """
        return [self[i : i + split_size] for i in range(0, len(self), split_size)]

    @staticmethod
    def concat(data: list["DataProto"]) -> "DataProto":
        """Concat a list of DataProto. The batch is concatenated among dim=0.
        The meta_info is merged, with special handling for metrics from different workers.

        Args:
            data (List[DataProto]): list of DataProto

        Returns:
            DataProto: concatenated DataProto
        """
        batch_lst = []
        for batch in data:
            batch_lst.append(batch.batch)
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst[0] is not None else None

        non_tensor_batch = list_of_dict_to_dict_of_list(list_of_dict=[d.non_tensor_batch for d in data])
        for key, val in non_tensor_batch.items():
            non_tensor_batch[key] = np.concatenate(val, axis=0)

        # Merge meta_info with special handling for metrics
        merged_meta_info = {}
        if data:
            # Merge non-metric meta_info and aggregate metrics from all workers.
            all_metrics = []
            for d in data:
                for k, v in d.meta_info.items():
                    if k == "metrics":
                        if v is not None:
                            if isinstance(v, list):
                                all_metrics.extend(v)
                            else:
                                all_metrics.append(v)
                    else:
                        if k in merged_meta_info:
                            # Ensure consistency for overlapping non-metric keys
                            assert merged_meta_info[k] == v, f"Conflicting values for meta_info key '{k}'"
                        else:
                            merged_meta_info[k] = v

            # Flatten list of dicts to dict of lists for consistent metrics structure
            if all_metrics:
                merged_meta_info["metrics"] = list_of_dict_to_dict_of_list(all_metrics)

        cls = type(data[0]) if len(data) > 0 else DataProto
        return cls(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=merged_meta_info)

    def reorder(self, indices):
        """
        Note that this operation is in-place
        """
        indices_np = indices.detach().numpy()
        self.batch = self.batch[indices]
        self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}

    def repeat(self, repeat_times=2, interleave=True):
        """
        Repeat the batch data a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        if self.batch is not None:
            if interleave:
                # Interleave the data
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
                }
            else:
                # Stack the data
                repeated_tensors = {
                    key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:])
                    for key, tensor in self.batch.items()
                }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def unfold_column_chunks(self, n_split: int, split_keys: Optional[list[str]] = None):
        """Split along the second dim into `n_split`, unfold it to the first dim (batch dim)
        Useful in passing grouped tensors that doesn't want to be shuffled in dataset.
        keys not in split_keys are repeated to match the shape
        Note that if the `split_keys` is not provided, it will repeat all the keys in the second dim.
        """
        if self.batch is not None:
            unfolded_batch = {}
            for key in self.batch.keys():
                if key in split_keys if split_keys is not None else False:
                    shape = list(self.batch[key].shape)
                    shape[0] = self.batch[key].shape[0] * n_split
                    shape[1] = self.batch[key].shape[1] // n_split
                    unfolded_batch[key] = self.batch[key].reshape(*shape)
                else:
                    unfolded_batch[key] = torch.repeat_interleave(self.batch[key], n_split, dim=0)
            # locate the `unfolded_batch` as a TensorDict on the same device as the original batch
            unfolded_batch = TensorDict(
                source=unfolded_batch, batch_size=(self.batch.batch_size[0] * n_split,), device=self.batch.device
            )
        else:
            unfolded_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if key in split_keys:
                shape = list(val.shape)
                shape[0] = val.shape[0] * n_split
                shape[1] = val.shape[1] // n_split
                repeated_non_tensor_batch[key] = val.reshape(*shape)
            else:
                repeated_non_tensor_batch[key] = np.repeat(val, n_split, axis=0)

        return type(self)(
            batch=unfolded_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def sample_level_repeat(self, repeat_times):
        """
        Repeat each row of the batch data a specified number of times.

        Args:
            repeat_times (torch.tensor, list, tuple, ndarray):  Number of times to repeat the data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        if isinstance(repeat_times, tuple):
            repeat_times = list(repeat_times)
        elif isinstance(repeat_times, torch.Tensor):
            assert len(repeat_times.shape) == 1
            repeat_times = repeat_times.tolist()
        elif isinstance(repeat_times, np.ndarray):
            assert len(repeat_times.shape) == 1
            repeat_times = repeat_times.tolist()
        else:
            assert isinstance(repeat_times, list), (
                f"repeat_times type must be in [list, torch.Tensor, np.ndarray, tuple], got {type(repeat_times)}"
            )
        repeat_times = torch.tensor(repeat_times)

        if self.batch is not None:
            # Interleave the data
            repeated_tensors = {
                key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
            }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(repeat_times.sum().item(),),
                device=self.batch.device,
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)

        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def to_tensordict(self) -> TensorDict:
        """Convert this DataProto to TensorDict. Note that this requires tensordict version at least 0.10

        Returns:

        """
        assert parse_version(tensordict.__version__) >= parse_version("0.10"), (
            "Convert DataProto to TensorDict at least requires tensordict version 0.10"
        )
        tensor_batch = self.batch.to_dict()
        non_tensor_batch = self.non_tensor_batch

        from tensordict.tensorclass import NonTensorData, NonTensorStack

        from verl.utils import tensordict_utils as tu

        common_keys = set(tensor_batch.keys()) & set(non_tensor_batch.keys())
        assert len(common_keys) == 0, f"tensor_batch and non_tensor_batch have common keys {common_keys}"

        for key, val in non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            # Convert to NonTensorStack instead of plain list to handle nested structures
            tensor_batch[key] = NonTensorStack.from_list([NonTensorData(item) for item in val])
        output = tu.get_tensordict(tensor_dict=tensor_batch, non_tensor_dict=self.meta_info)
        return output

    def get_data_info(self) -> str:
        """Return formatted information about stored data with nested type details.

        Returns:
            str: Formatted string showing tensor details and recursive metadata types
        """
        info = ["batch"]

        for key, tensor in self.batch.items():
            if hasattr(tensor, "shape") and hasattr(tensor, "dtype") and hasattr(tensor, "device"):
                info.append(f"  {key}: {tuple(tensor.shape)} ({tensor.dtype}) {tensor.device}")
            elif hasattr(tensor, "shape") and hasattr(tensor, "dtype"):
                info.append(f"  {key}: {tuple(tensor.shape)} ({tensor.dtype})")
            else:
                info.append(f"  {key}: {type(tensor).__name__}")

        info.append("non_tensor_batch")
        for key, array in self.non_tensor_batch.items():
            info.append(f"  {key}: ndarray{array.shape} ({array.dtype})")

        info.append("meta_info")
        for k, v in self.meta_info.items():
            type_info = self._get_type_info(v)
            info.append(f"  {k}: {type_info}")

        return "\n".join(info)

    def _get_type_info(self, value):
        """Recursively get type information for nested structures"""
        if isinstance(value, list):
            elem_types = {self._get_type_info(v) for v in value[:3]}
            return f"list[{'|'.join(elem_types) if elem_types else '...'}]"
        if isinstance(value, tuple):
            elem_types = [self._get_type_info(v) for v in value]
            return f"tuple({', '.join(elem_types)})"
        if isinstance(value, dict):
            if not value:
                return "dict"
            k, v = next(iter(value.items()))
            return f"dict[{self._get_type_info(k)}: {self._get_type_info(v)}]"
        if isinstance(value, np.ndarray):
            return f"ndarray{value.shape} ({value.dtype})"
        return type(value).__name__


@dataclass
class DataProtoFuture:
    """
    DataProtoFuture aims to eliminate actual data fetching on driver. By doing so, the driver doesn't have to wait
    for data so that asynchronous execution becomes possible.
    DataProtoFuture contains a list of futures from another WorkerGroup of size world_size.
    - collect_fn is a Callable that reduces the list of futures to a DataProto
    - dispatch_fn is a Callable that partitions the DataProto into a list of DataProto of size world_size
        and then select

    Potential issue: we can optimize dispatch_fn(collect_fn) such that only needed data is fetched on destination
    - DataProtoFuture only supports directly passing from the output of a method to another input. You can't perform any
    operation on the DataProtoFuture in driver.
    """

    collect_fn: Callable
    futures: list[ray.ObjectRef]
    dispatch_fn: Callable = None

    @staticmethod
    def concat(data: list[ray.ObjectRef]) -> "DataProtoFuture":
        output = DataProtoFuture(collect_fn=DataProto.concat, futures=data)
        return output

    def chunk(self, chunks: int) -> list["DataProtoFuture"]:
        from functools import partial

        arg_future_lst = []
        for i in range(chunks):
            # note that we can't directly pass i and chunks
            def dispatch_fn(x, i, chunks):
                return x.chunk(chunks=chunks)[i]

            arg_future = DataProtoFuture(
                collect_fn=self.collect_fn, dispatch_fn=partial(dispatch_fn, i=i, chunks=chunks), futures=self.futures
            )
            arg_future_lst.append(arg_future)
        return arg_future_lst

    def get(self):
        output = ray.get(self.futures)  # dp_size.
        for o in output:
            assert isinstance(o, DataProto | TensorDict)

        if isinstance(output[0], DataProto):
            output = DataProto.concat(output)  # select dp, concat
        elif isinstance(output[0], TensorDict):
            from verl.utils.tensordict_utils import concat_tensordict

            output = concat_tensordict(output)
        else:
            raise TypeError(f"Unknown type {type(o[0])} in DataProtoFuture")

        if self.dispatch_fn is not None:
            output = self.dispatch_fn(output)  # split in batch dim, select using dp
        return output


def all_gather_data_proto(data: DataProto, process_group):
    # Note that this is an inplace operator just like torch.distributed.all_gather
    group_size = torch.distributed.get_world_size(group=process_group)
    assert isinstance(data, DataProto)
    prev_device = data.batch.device
    data = data.to(get_device_id())
    data.batch = allgather_dict_tensors(data.batch.contiguous(), size=group_size, group=process_group, dim=0)
    data = data.to(prev_device)
    # all gather non_tensor_batch
    all_non_tensor_batch = [None for _ in range(group_size)]
    torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=process_group)
    data.non_tensor_batch = {k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch}
