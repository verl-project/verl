Data interface
=========================

Last updated: 05/19/2025 (API docstrings are auto-generated).

DataProto is the interface for data exchange.

The :class:`verl.DataProto` class contains two key members:

- batch: a :class:`tensordict.TensorDict` object for the actual data
- meta_info: a :class:`Dict` with additional meta information

TensorDict
~~~~~~~~~~~~

:attr:`DataProto.batch` is built on top of :class:`tensordict`, a project in the PyTorch ecosystem.
A TensorDict is a dict-like container for tensors. To instantiate a TensorDict, you must specify key-value pairs as well as the batch size.

.. code-block:: python

    >>> import torch
    >>> from tensordict import TensorDict
    >>> tensordict = TensorDict({"zeros": torch.zeros(2, 3, 4), "ones": torch.ones(2, 3, 5)}, batch_size=[2,])
    >>> tensordict["twos"] = 2 * torch.ones(2, 5, 6)
    >>> zeros = tensordict["zeros"]
    >>> tensordict
    TensorDict(
    fields={
        ones: Tensor(shape=torch.Size([2, 3, 5]), device=cpu, dtype=torch.float32, is_shared=False),
        twos: Tensor(shape=torch.Size([2, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
        zeros: Tensor(shape=torch.Size([2, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([2]),
    device=None,
    is_shared=False)

One can also index a tensordict along its batch_size. The contents of the TensorDict can be manipulated collectively as well.

.. code-block:: python

    >>> tensordict[..., :1]
    TensorDict(
    fields={
        ones: Tensor(shape=torch.Size([1, 3, 5]), device=cpu, dtype=torch.float32, is_shared=False),
        twos: Tensor(shape=torch.Size([1, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
        zeros: Tensor(shape=torch.Size([1, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([1]),
    device=None,
    is_shared=False)
    >>> tensordict = tensordict.to("cuda:0")
    >>> tensordict = tensordict.reshape(6)

For more about :class:`tensordict.TensorDict` usage, see the official tensordict_ documentation.

.. _tensordict: https://pytorch.org/tensordict/stable/overview.html


Core APIs
~~~~~~~~~~~~~~~~~

.. autoclass::  verl.DataProto
   :members: to, select, union, make_iterator, concat
