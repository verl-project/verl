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

import logging
from typing import Any, Iterable

import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack


def assign_non_tensor_data(tensor_dict: TensorDict, key, val):
    """Assign a single non-tensor value to a TensorDict.

    Wraps the value in NonTensorData so it can be stored alongside tensors
    in the TensorDict. Use this for scalar metadata or simple non-tensor values.

    Args:
        tensor_dict: The TensorDict to assign to.
        key: The key under which to store the value.
        val: Any non-tensor value to store (e.g., string, int, dict).

    Raises:
        AssertionError: If tensor_dict is not a TensorDict.

    Example:
        >>> td = TensorDict({"obs": torch.randn(3, 4)}, batch_size=[3])
        >>> assign_non_tensor_data(td, "experiment_name", "run_001")
    """
    assert isinstance(tensor_dict, TensorDict), "input dict must be a TensorDict"
    tensor_dict[key] = NonTensorData(val)


def assign_non_tensor_stack(tensor_dict: TensorDict, key, val: list):
    """Assign a list with potentially nested structures (lists, dicts, etc.) to TensorDict.

    This function handles complex nested data structures like:
    - Lists of lists: [[], [0.5, 0.8], [0.9]]
    - Lists of dicts: [{"acc": 1.0}, {"acc": 0.0}]
    - Lists of lists of dicts: [[{"content": "...", "role": "user"}]]

    These structures are wrapped in NonTensorStack so TensorDict can handle them correctly.

    Args:
        tensor_dict: The TensorDict to assign to
        key: The key to assign the value under
        val: A list containing potentially nested structures

    Example:
        >>> td = TensorDict({}, batch_size=[])
        >>> turn_scores = [[], [0.5, 0.8], [0.9]]
        >>> assign_non_tensor_stack(td, "turn_scores", turn_scores)
        >>> # Now td["turn_scores"] contains the nested data
    """
    # Convert list to NonTensorStack to handle nested structures
    # This wraps each item in NonTensorData to preserve complex objects
    # TODO(petersh6): can convert back to val directly if we are not accessing .data from the NonTensorStack
    assert isinstance(tensor_dict, TensorDict), "input dict must be a TensorDict"
    tensor_dict[key] = NonTensorStack.from_list([NonTensorData(item) for item in val])


def assign_non_tensor(tensor_dict: TensorDict, **kwargs):
    """Assign non-tensor data to a TensorDict.

    Automatically detects if the value is a list with nested structures and uses
    the appropriate assignment method (NonTensorData for simple values,
    NonTensorStack for lists with nested structures).

    Args:
        tensor_dict: The TensorDict to assign to
        **kwargs: Key-value pairs where values can be:
            - Simple values (stored as NonTensorData)
            - Lists with nested structures (stored as NonTensorStack)

    Example:
        >>> td = TensorDict({"obs": torch.randn(3, 4)}, batch_size=[3])
        >>> assign_non_tensor(
        ...     tensor_dict=td,
        ...     metadata="experiment_1",  # Simple value
        ...     turn_scores=[[], [0.5, 0.8], [0.9]]  # Nested list
        ... )
    """
    assert isinstance(tensor_dict, TensorDict), "input dict must be a TensorDict"
    for key, val in kwargs.items():
        if isinstance(val, (NonTensorData | NonTensorStack)):
            tensor_dict[key] = val
        elif isinstance(val, list):
            # For lists, use NonTensorStack
            assign_non_tensor_stack(tensor_dict=tensor_dict, key=key, val=val)
        else:
            # For non-list values, use NonTensorData
            assign_non_tensor_data(tensor_dict=tensor_dict, key=key, val=val)
    return tensor_dict


def unwrap_non_tensor_data(data):
    """Unwrap a NonTensorData object to get the underlying value.

    If the input is a NonTensorData wrapper, extracts and returns the
    underlying data. Otherwise, returns the input unchanged.

    Args:
        data: Either a NonTensorData object or any other value.

    Returns:
        The unwrapped data if input was NonTensorData, otherwise the
        original input unchanged.

    Example:
        >>> wrapped = NonTensorData("hello")
        >>> unwrap_non_tensor_data(wrapped)
        'hello'
        >>> unwrap_non_tensor_data(42)  # Non-wrapped value
        42
    """
    if isinstance(data, NonTensorData):
        return data.data
    return data


def get_non_tensor_data(data: TensorDict, key: str, default):
    """Retrieve and unwrap non-tensor data from a TensorDict.

    Fetches the value for the given key from the TensorDict and automatically
    unwraps it if it's stored as NonTensorData.

    Args:
        data: The TensorDict to retrieve from.
        key: The key to look up.
        default: Value to return if the key is not found.

    Returns:
        The unwrapped value if the key exists and was wrapped in NonTensorData,
        the raw value if it wasn't wrapped, or the default if key not found.

    Example:
        >>> td = TensorDict({}, batch_size=[])
        >>> assign_non_tensor_data(td, "config", {"lr": 0.01})
        >>> get_non_tensor_data(td, "config", None)
        {'lr': 0.01}
        >>> get_non_tensor_data(td, "missing", "default_value")
        'default_value'
    """
    output = data.get(key, default)
    return unwrap_non_tensor_data(output)


def concat_nested_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate multiple nested tensors along the batch dimension.

    Takes a list of nested tensors with jagged layout and concatenates them
    into a single nested tensor. Each input tensor must have 2 or more dimensions and be contiguous.

    Args:
        tensors: List of nested tensors to concatenate. All tensors must
            be nested, contiguous, and have 2 or more dimensions.

    Returns:
        A new nested tensor with jagged layout containing all rows from
        the input tensors concatenated along dimension 0.

    Raises:
        AssertionError: If any tensor is not nested, not contiguous, or
            doesn't have 2 or more dimensions.

    Example:
        >>> t1 = torch.nested.as_nested_tensor([torch.randn(3), torch.randn(5)], layout=torch.jagged)
        >>> t2 = torch.nested.as_nested_tensor([torch.randn(2), torch.randn(4)], layout=torch.jagged)
        >>> result = concat_nested_tensors([t1, t2])
        >>> # result contains 4 rows: lengths [3, 5, 2, 4]
    """
    for tensor in tensors:
        assert tensor.is_nested and tensor.is_contiguous()
    unbind_tensors = []
    for tensor in tensors:
        assert len(tensor.shape) >= 2, f"nested tensor must have 2 or more dimensions. Got {tensor.shape}"
        unbind_tensor = tensor.unbind(0)
        unbind_tensors.extend(list(unbind_tensor))

    tensor = torch.nested.as_nested_tensor(unbind_tensors, layout=torch.jagged)
    return tensor


def concat_tensordict_with_none_bsz(data: list[TensorDict]):
    """Handle concatenation of TensorDicts with empty batch size.

    For TensorDicts that contain only metadata (NonTensorData) with no batch
    dimension, returns the first TensorDict as the concatenation result.

    Args:
        data: List of TensorDicts, each with empty batch_size (batch_size=[]).

    Returns:
        The first TensorDict from the list, as metadata concatenation
        simply preserves the first instance.

    Raises:
        AssertionError: If any TensorDict has a non-empty batch_size.

    Note:
        This is used internally by concat_tensordict when handling
        TensorDicts that contain only non-tensor metadata.
    """
    for d in data:
        assert len(d.batch_size) == 0
    # directly return the first meta info
    return data[0]


def concat_tensordict(data: list[TensorDict]) -> TensorDict:
    """Concatenate multiple TensorDicts along dimension zero.

    Combines a list of TensorDicts into a single TensorDict by concatenating
    all tensors along the batch dimension (dim=0). Handles nested tensors
    specially by unbinding and rebinding them.

    Args:
        data: List of TensorDicts to concatenate. All TensorDicts must have
            the same keys and the same set of nested tensor keys.

    Returns:
        A new TensorDict containing concatenated tensors from all inputs.

    Raises:
        AssertionError: If data is empty or if TensorDicts have inconsistent
            nested tensor keys.

    Note:
        - For TensorDicts with empty batch_size, returns the first one
        - Nested tensors are handled specially via concat_nested_tensors
        - Regular tensors use TensorDict.cat for efficient concatenation
    """
    assert len(data) > 0, "Must have at least one tensordict"

    # Find nested tensor keys from the first tensordict
    nested_tensor_keys = {key for key, value in data[0].items() if isinstance(value, torch.Tensor) and value.is_nested}

    if not nested_tensor_keys:
        if len(data[0].batch_size) == 0:
            return concat_tensordict_with_none_bsz(data)
        # if batch size is None (only contain NonTensorData)
        return TensorDict.cat(data, dim=0)

    # Create a list of tensordicts containing only non-nested tensors for concatenation
    regular_tds = []
    for td in data:
        current_nested_keys = {k for k, v in td.items() if isinstance(v, torch.Tensor) and v.is_nested}
        assert current_nested_keys == nested_tensor_keys, "All tensordicts must have the same set of nested tensors."

        # Create a new TensorDict with non-nested items without modifying the original
        regular_items = {k: v for k, v in td.items() if k not in nested_tensor_keys}
        regular_tds.append(TensorDict(regular_items, batch_size=td.batch_size, device=td.device))

    # Concatenate the regular tensordicts
    output = TensorDict.cat(regular_tds, dim=0)

    # Concatenate and add nested tensors to the output
    for key in nested_tensor_keys:
        nested_tensors_to_concat = [td[key] for td in data]
        output[key] = concat_nested_tensors(nested_tensors_to_concat)

    return output


def chunk_tensordict(td: TensorDict, chunks: int) -> list[TensorDict]:
    """Split a TensorDict into equal-sized chunks with special nested tensor handling.

    Divides a TensorDict into the specified number of chunks along the batch
    dimension. Handles 3D+ nested tensors specially since torch.chunk() doesn't
    support jagged tensors with 3 or more dimensions.

    Args:
        td: The TensorDict to split.
        chunks: Number of chunks to create. Must evenly divide len(td).

    Returns:
        List of TensorDicts, each containing a portion of the original data.

    Raises:
        AssertionError: If td is not a TensorDict or if its length is not
            evenly divisible by chunks.

    Note:
        This is a workaround for PyTorch issue #153238 where torch.chunk()
        doesn't support 3D jagged tensors (e.g., MRoPE position_ids).
        See: https://github.com/pytorch/pytorch/issues/153238
    """
    assert isinstance(td, TensorDict) and len(td) % chunks == 0, (
        f"expecting td with length divisible by chunks, but got {len(td)} and {chunks}"
    )
    chunk_size = len(td) // chunks
    keys = {key for key, val in td.items() if isinstance(val, torch.Tensor) and val.is_nested and val.dim() >= 3}
    new_td = TensorDict({k: v for k, v in td.items() if k not in keys}, batch_size=td.batch_size, device=td.device)

    tds = new_td.chunk(chunks=chunks)
    for key in keys:
        tensors = td[key].unbind(dim=0)
        for i, chunk_td in enumerate(tds):
            chunk_td[key] = torch.nested.as_nested_tensor(
                tensors[i * chunk_size : (i + 1) * chunk_size], layout=torch.jagged
            )

    return tds


def get_tensordict(tensor_dict: dict[str, torch.Tensor | list], non_tensor_dict: dict = None) -> TensorDict:
    """Create a TensorDict from tensors and non-tensor data.

    Automatically handles nested structures in lists by converting them to NonTensorStack.
    This enables support for:
    - Lists of lists: [[], [0.5, 0.8], [0.9]]
    - Lists of dicts: [{"acc": 1.0}, {"acc": 0.0}]
    - Lists of lists of dicts: [[{"content": "...", "role": "user"}]]

    Args:
        tensor_dict: Dictionary of tensors and lists to include in the TensorDict
        non_tensor_dict: Dictionary of metadata to store as NonTensorData

    Returns:
        TensorDict with proper handling of nested structures

    Example:
        >>> td = get_tensordict(
        ...     tensor_dict={
        ...         "obs": torch.randn(3, 4),
        ...         "turn_scores": [[], [0.5, 0.8], [0.9]]  # Nested list
        ...     },
        ...     non_tensor_dict={"experiment": "test"}
        ... )
    """
    tensor_dict = tensor_dict.copy()
    if non_tensor_dict is None:
        non_tensor_dict = {}

    batch_size = None

    for key, val in tensor_dict.items():
        if isinstance(val, torch.Tensor) and val.is_nested:
            assert val.is_contiguous(), "Nested tensors must be contiguous. Try setting layout=torch.jagged"
            assert val.layout == torch.jagged, "Nested tensors must be jagged."

        # Skip validation for NonTensorStack as it's already properly formatted
        if isinstance(val, NonTensorStack):
            if batch_size is None:
                batch_size = len(val)
            else:
                assert len(val) == batch_size, (
                    f"Batch size of NonTensorStack {key} is not consistent with other tensors. "
                    f"Expected {batch_size}, got {len(val)}"
                )
            continue

        if isinstance(val, list):
            for v in val:
                assert not isinstance(v, torch.Tensor), (
                    "Passing a list makes the data NonTensorStack, "
                    "which doesn't support torch.Tensor. Please convert to numpy first"
                )
            # Convert to NonTensorStack to handle nested structures
            tensor_dict[key] = NonTensorStack.from_list([NonTensorData(item) for item in val])

        assert isinstance(val, torch.Tensor | list)

        if batch_size is None:
            batch_size = val.size(0) if isinstance(val, torch.Tensor) else len(val)
        else:
            val_batch_size = val.size(0) if isinstance(val, torch.Tensor) else len(val)
            assert val_batch_size == batch_size, (
                f"Batch size of tensor {key} is not consistent with other tensors. "
                f"Expected {batch_size}, got {val_batch_size}"
            )

    if batch_size is None:
        batch_size = []
    else:
        batch_size = [batch_size]

    for key, val in non_tensor_dict.items():
        assert key not in tensor_dict
        tensor_dict[key] = NonTensorData(val)

    return TensorDict(source=tensor_dict, batch_size=batch_size)


def index_select_tensor_dict(batch: TensorDict, indices: torch.Tensor | list[int]) -> TensorDict:
    """Select rows from a TensorDict using indices.

    Creates a new TensorDict containing only the rows specified by indices.
    Handles regular tensors, nested tensors, NonTensorStack, and NonTensorData
    appropriately.

    Args:
        batch: The TensorDict to index into. Can be None.
        indices: 1D tensor or list of integers specifying which rows to select.

    Returns:
        A new TensorDict containing only the selected rows, or None if
        batch was None.

    Raises:
        AssertionError: If indices is not 1-dimensional.

    Note:
        - Regular tensors are indexed directly
        - Nested tensors are unbound, indexed, and rebound
        - NonTensorStack is indexed by batch dimension
        - NonTensorData (scalar metadata) is preserved unchanged
    """
    if isinstance(indices, list):
        indices = torch.tensor(indices)

    assert indices.dim() == 1, "indices must be a 1D tensor"

    data_dict = {}
    batch_size = indices.shape[0]

    if batch is not None:
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and not tensor.is_nested:
                data_dict[key] = tensor[indices]
            elif isinstance(tensor, torch.Tensor) and tensor.is_nested:
                tensor_lst = tensor.unbind()  # for performance
                data_dict[key] = torch.nested.as_nested_tensor(
                    [tensor_lst[idx] for idx in indices], layout=torch.jagged
                )
            else:
                # This handles NonTensorStack (indexable by batch dim) and NonTensorData (scalar metadata).
                if tensor.shape:
                    data_dict[key] = tensor[indices]
                else:
                    data_dict[key] = tensor
        selected_batch = TensorDict(source=data_dict, batch_size=batch_size)
    else:
        selected_batch = None

    return selected_batch


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """Merge two TensorDicts, adding keys from the second to the first.

    Performs an in-place union of two TensorDicts. Keys from tensor_dict2
    that don't exist in tensor_dict1 are added. Keys that exist in both
    must have identical values.

    Args:
        tensor_dict1: The base TensorDict to merge into (modified in-place).
        tensor_dict2: The TensorDict whose keys will be added to tensor_dict1.

    Returns:
        The modified tensor_dict1 containing the union of both TensorDicts.

    Raises:
        AssertionError: If batch sizes don't match, or if a key exists in
            both TensorDicts with different values.

    Example:
        >>> td1 = TensorDict({"a": torch.tensor([1, 2])}, batch_size=[2])
        >>> td2 = TensorDict({"b": torch.tensor([3, 4])}, batch_size=[2])
        >>> result = union_tensor_dict(td1, td2)
        >>> list(result.keys())
        ['a', 'b']
    """
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, (
        f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
    )
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            # Note that there is a difference between tensor_dict2[key] and tensor_dict2.get(key)
            tensor_dict1[key] = tensor_dict2.get(key)
        else:
            if isinstance(tensor_dict2[key], torch.Tensor):
                assert tensor_dict1[key].equal(tensor_dict2[key]), (
                    f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
                )
            else:
                # non-tensor
                assert tensor_dict1[key] == tensor_dict2[key], (
                    f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
                )

    return tensor_dict1


def make_iterator(tensordict: TensorDict, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
    """Create an iterator that yields mini-batches from a TensorDict.

    Wraps a TensorDict in a DataLoader-style iterator that yields mini-batches
    for the specified number of epochs. Useful for training loops.

    Args:
        tensordict: The TensorDict to iterate over.
        mini_batch_size: Size of each mini-batch. Must evenly divide the
            TensorDict's batch size.
        epochs: Number of times to iterate through the entire dataset.
        seed: Optional random seed for reproducible shuffling.
        dataloader_kwargs: Optional dict of additional kwargs to pass to
            the underlying DataLoader (e.g., shuffle=True, num_workers=4).

    Returns:
        An iterator that yields TensorDict mini-batches.

    Raises:
        AssertionError: If batch size is not divisible by mini_batch_size.

    Example:
        >>> td = TensorDict({"obs": torch.randn(100, 4)}, batch_size=[100])
        >>> for batch in make_iterator(td, mini_batch_size=10, epochs=2):
        ...     # batch is a TensorDict with batch_size=[10]
        ...     pass
    """
    from torch.utils.data import DataLoader

    assert tensordict.batch_size[0] % mini_batch_size == 0, f"{tensordict.batch_size[0]} % {mini_batch_size} != 0"
    # we can directly create a dataloader from TensorDict
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    else:
        generator = None

    assert isinstance(dataloader_kwargs, dict)

    idx_lst = torch.arange(tensordict.shape[0])

    train_dataloader = DataLoader(
        dataset=idx_lst, batch_size=mini_batch_size, collate_fn=lambda x: x, generator=generator, **dataloader_kwargs
    )

    def get_data():
        for _ in range(epochs):
            for idx in train_dataloader:
                yield index_select_tensor_dict(tensordict, idx)

    return iter(get_data())


def assert_tensordict_eq(tensordict1: TensorDict, tensordict2: TensorDict):
    """Assert that two TensorDicts are equal.

    Performs a deep equality check between two TensorDicts, verifying that
    they have the same keys with identical values. Handles nested tensors
    by comparing their unbound components.

    Args:
        tensordict1: First TensorDict to compare.
        tensordict2: Second TensorDict to compare.

    Raises:
        AssertionError: If the TensorDicts differ in keys, value types, or
            value contents. The error message indicates what differs.

    Note:
        - Regular tensors are compared element-wise
        - Nested tensors are unbound and compared component by component
        - Non-tensor values are compared with standard equality
    """
    tensordict1_key_set = set(tensordict1.keys())
    tensordict2_key_set = set(tensordict2.keys())
    assert tensordict1_key_set == tensordict2_key_set, (
        f"key set diffs. Got {tensordict2_key_set=} vs {tensordict1_key_set=}"
    )

    for key in tensordict1.keys():
        val = tensordict1[key]
        val2 = tensordict2[key]

        assert type(val) is type(val2), f"The type of {key} must be the same. Got {type(val)} vs {type(val2)}"

        if isinstance(val, torch.Tensor):
            if val.is_nested:
                assert val.is_nested and val2.is_nested, (
                    f"Both tensors must be nested tensors. {val.is_nested=}, {val2.is_nested=}"
                )
                t1, t2 = val.unbind(), val2.unbind()
                assert len(t1) == len(t2), f"Nested tensor should have the same lengths. {len(t1)=} vs {len(t2)=}"
                for c1, c2 in zip(t1, t2, strict=True):
                    assert torch.equal(c1, c2), f"Nested tensor components have different values. {c1=} vs {c2=}"
            else:
                assert torch.all(torch.eq(val, val2)).item()
        else:
            assert val == val2


def get(tensordict: TensorDict, key: str, default=None) -> Any:
    """Get a value from a TensorDict with automatic unwrapping.

    Retrieves a value from the TensorDict and automatically converts it
    to a Python-native format:
    - Tensors are returned as-is
    - NonTensorStack is converted to a Python list
    - NonTensorData is unwrapped to its underlying value

    Args:
        tensordict: The TensorDict to retrieve from.
        key: The key to look up.
        default: Value to return if the key doesn't exist. Defaults to None.

    Returns:
        The value for the key in its native format, or default if not found.

    Example:
        >>> td = get_tensordict({"obs": torch.randn(3, 4), "labels": ["a", "b", "c"]})
        >>> get(td, "obs")  # Returns torch.Tensor
        >>> get(td, "labels")  # Returns ["a", "b", "c"] as a list
        >>> get(td, "missing", "default")  # Returns "default"
    """
    if key not in tensordict:
        return default

    output = tensordict.get(key)
    if isinstance(output, torch.Tensor):
        return output
    elif isinstance(output, NonTensorStack):
        return output.tolist()
    else:
        assert isinstance(output, NonTensorData)
        return output.data


def get_keys(tensordict: TensorDict, keys: Iterable[str]) -> TensorDict:
    """Extract a subset of keys from a TensorDict into a new TensorDict.

    Creates a new TensorDict containing only the specified keys. Values
    are properly categorized as tensor or non-tensor data.

    Args:
        tensordict: The source TensorDict.
        keys: Iterable of key names to extract.

    Returns:
        A new TensorDict containing only the specified keys with their values.

    Raises:
        KeyError: If any key in keys doesn't exist in the tensordict.

    Example:
        >>> td = get_tensordict({"a": torch.randn(3), "b": torch.randn(3), "c": torch.randn(3)})
        >>> subset = get_keys(td, ["a", "c"])
        >>> list(subset.keys())
        ['a', 'c']
    """
    tensor_output = {}
    non_tensor_output = {}
    for key in keys:
        if key not in tensordict.keys():
            raise KeyError(f"key {key} not in tensordict")
        output = tensordict.get(key)
        if isinstance(output, torch.Tensor):
            tensor_output[key] = output
        elif isinstance(output, NonTensorStack):
            tensor_output[key] = output.tolist()
        else:
            assert isinstance(output, NonTensorData)
            non_tensor_output[key] = output.data

    return get_tensordict(tensor_output, non_tensor_output)


def pop(tensordict: TensorDict, key: str, default=None) -> Any:
    """Remove and return a value from a TensorDict with automatic unwrapping.

    Removes the specified key from the TensorDict and returns its value,
    automatically converting to Python-native format (same as get()).

    Args:
        tensordict: The TensorDict to pop from.
        key: The key to remove and return.
        default: Value to return if the key doesn't exist. Defaults to None.

    Returns:
        The value for the key in its native format, or default if not found.
        The key is removed from the TensorDict.

    Example:
        >>> td = get_tensordict({"obs": torch.randn(3, 4), "labels": ["a", "b", "c"]})
        >>> labels = pop(td, "labels")  # Returns ["a", "b", "c"], removes from td
        >>> "labels" in td.keys()
        False
    """
    _sentinel = object()
    output = tensordict.pop(key, _sentinel)
    if output is _sentinel:
        return default

    if isinstance(output, torch.Tensor):
        return output
    elif isinstance(output, NonTensorStack):
        return output.tolist()
    else:
        assert isinstance(output, NonTensorData)
        return output.data


def pop_keys(tensordict: TensorDict, keys: Iterable[str]) -> TensorDict:
    """Remove multiple keys from a TensorDict and return them as a new TensorDict.

    Removes the specified keys from the source TensorDict and creates a new
    TensorDict containing those keys and their values.

    Args:
        tensordict: The source TensorDict to pop from (modified in-place).
        keys: Iterable of key names to remove and return.

    Returns:
        A new TensorDict containing the popped keys and their values.

    Raises:
        KeyError: If any key in keys doesn't exist in the tensordict.

    Example:
        >>> td = get_tensordict({"a": torch.randn(3), "b": torch.randn(3), "c": torch.randn(3)})
        >>> popped = pop_keys(td, ["a", "c"])
        >>> list(td.keys())  # Only 'b' remains
        ['b']
        >>> list(popped.keys())
        ['a', 'c']
    """
    tensor_output = {}
    non_tensor_output = {}
    for key in keys:
        if key not in tensordict.keys():
            raise KeyError(f"key {key} not in tensordict")
        output = tensordict.get(key)
        if isinstance(output, torch.Tensor):
            tensor_output[key] = tensordict.pop(key)
        elif isinstance(output, NonTensorStack):
            tensor_output[key] = tensordict.pop(key).tolist()
        else:
            assert isinstance(output, NonTensorData)
            non_tensor_output[key] = tensordict.pop(key)

    return get_tensordict(tensor_output, non_tensor_output)


def pad_to_divisor(data: TensorDict, size_divisor: int):
    """Pad a TensorDict's batch dimension to be divisible by a given divisor.

    If the TensorDict's length is not evenly divisible by size_divisor,
    pads the batch dimension by repeating elements from the beginning.
    Useful for ensuring even distribution across workers in distributed training.

    Args:
        data: The TensorDict to pad.
        size_divisor: The divisor that the padded length must be divisible by.

    Returns:
        tuple: A tuple containing:
            - data (TensorDict): The padded TensorDict (or original if no padding needed)
            - pad_size (int): Number of elements added as padding (0 if none)

    Raises:
        AssertionError: If data is not a TensorDict.

    Example:
        >>> td = TensorDict({"obs": torch.randn(10, 4)}, batch_size=[10])
        >>> padded, pad_size = pad_to_divisor(td, 4)
        >>> len(padded)  # 12 (next multiple of 4 after 10)
        12
        >>> pad_size
        2
    """
    assert isinstance(data, TensorDict), "data must be a TensorDict"
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        data_padded = torch.cat([data] + padding_protos)
    else:
        if len(data) == 0:
            logging.warning("padding a DataProto with no item, no changed made")
        pad_size = 0
        data_padded = data
    return data_padded, pad_size


def unpad(data: TensorDict, pad_size):
    """Remove padding from a TensorDict.

    Reverses the effect of pad_to_divisor by removing the specified number
    of elements from the end of the TensorDict.

    Args:
        data: The padded TensorDict.
        pad_size: Number of padding elements to remove. If 0, returns
            data unchanged.

    Returns:
        The TensorDict with padding removed, equivalent to data[:-pad_size].

    Example:
        >>> td = TensorDict({"obs": torch.randn(12, 4)}, batch_size=[12])
        >>> unpadded = unpad(td, pad_size=2)
        >>> len(unpadded)
        10
    """
    if pad_size != 0:
        data = data[:-pad_size]
    return data


def contiguous(data: TensorDict) -> TensorDict:
    """Call contiguous on a tensor dict. The contiguous function of tensordict lib will make NonTensorStack.
    This function will always return a new tensordict

    Args:
        data: The input tensordict

    Returns:
        a tensordict that is contiguous

    """
    tensor_dict = {}
    non_tensor_dict = {}

    for key in data.keys():
        val = data.get(key)
        if isinstance(val, NonTensorData):
            non_tensor_dict[key] = val
        elif isinstance(val, NonTensorStack):
            tensor_dict[key] = val
        else:
            assert isinstance(val, torch.Tensor), f"Expect val to be a torch.Tensor. Got {type(val)}"
            tensor_dict[key] = val.contiguous()

    return get_tensordict(tensor_dict=tensor_dict, non_tensor_dict=non_tensor_dict)


def maybe_fix_3d_position_ids(data: TensorDict):
    # note for tensordict with pickle/unpickle. nested tensor in tensordict after consolidate and pickle/unpickle
    # will incur indexing error for ragged tensor. This only happens when using 3D position ids in VLMs.
    # This is likely a bug in tensordict. As a workaround, we manually set _ragged_index.
    if "position_ids" in data.keys() and data["position_ids"].dim() == 3 and data["position_ids"].is_nested:
        data["position_ids"]._ragged_idx = 2
