# TransferQueue Data System

Last updated: 04/07/2026.

This doc introduce [TransferQueue](https://github.com/Ascend/TransferQueue), an asynchronous streaming data management system for efficient post-training.

🔥 **Now TransferQueue is open-sourced at both [Github](https://github.com/Ascend/TransferQueue) and [GitCode](https://gitcode.com/Ascend/TransferQueue). <span style="color: #FF0000;">You are welcome to submit contributions or propose new ideas on either platform!**</span>


> At the mean time, the early development history remains accessible at: https://github.com/TransferQueue/TransferQueue.

<h2 id="overview"> Overview</h2>

TransferQueue is a high-performance data storage and transfer module with panoramic data visibility and streaming scheduling capabilities, optimized for efficient dataflow in post-training workflows.

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/tq_arch.png?raw=true" width="70%">
</p>

TransferQueue offers **fine-grained, sub-sample-level** data management and **load-balancing** capabilities. It serves as a data gateway that decouples explicit data dependencies across computational tasks, enabling a divide-and-conquer approach that significantly simplifies algorithm controller design.

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/main_func.png?raw=true" width="100%">
</p>

<h2 id="updates"> Updates</h2>

 - **Feb 8, 2026**: 🔥 Initialization and usage are greatly simplified by high-level APIs [PR#26](https://github.com/Ascend/TransferQueue/pull/26), [PR#28](https://github.com/Ascend/TransferQueue/pull/28). You can now use a Redis-style API to take advantage of most of the advanced features provided by TransferQueue!
 - **Jan 28, 2026**: We experimentally introduce the `StreamingDataLoader` interface for a fully-streamed production-consumption pipeline. Refer to our [tutorials/06_streaming_dataloader.py](https://github.com/Ascend/TransferQueue/blob/main/tutorial/06_streaming_dataloader.py) for details.
 - **Dec 30, 2025**: **TransferQueue x verl** integration has been tested with the DAPO algorithm at scale **(64 nodes, 1024 cards)**. It significantly optimizes host memory utilization and accelerates data transfers. Stay tuned for more details!
 - **Dec 20, 2025**: 🔥 The official [tutorial](https://github.com/Ascend/TransferQueue/tree/main/tutorial) is released! Feel free to check it out.
 - **Nov 10, 2025**: We disentangled the data retrieval logic from TransferQueueController [PR#101](https://github.com/TransferQueue/TransferQueue/pull/101). Now you can implement your own `Sampler` to customize data consumption.
 - **Nov 5, 2025**: We provide a `KVStorageManager` that simplifies the integration with KV-based storage backends [PR#96](https://github.com/TransferQueue/TransferQueue/pull/96). The first available KV-based backend is [Yuanrong](https://gitcode.com/openeuler/yuanrong-datasystem).
 - **Nov 4, 2025**: Data partitioning capability is available in [PR#98](https://github.com/TransferQueue/TransferQueue/pull/98). Now you can define logical data partitions to manage your train/val/test datasets.
 - **Oct 25, 2025**: Storage backends are now pluggable in [PR#66](https://github.com/TransferQueue/TransferQueue/pull/66). You can try to integrate your own storage backend with TransferQueue now!
 - **Oct 21, 2025**: Official integration into verl is ready [verl/pulls/3649](https://github.com/volcengine/verl/pull/3649). Following PRs will optimize the single controller architecture by fully decoupling data & control flows.
 - **July 22, 2025**: We published a series of Chinese blog posts on <a href="https://zhuanlan.zhihu.com/p/1930244241625449814">Zhihu 1</a>, <a href="https://zhuanlan.zhihu.com/p/1933259599953232589">2</a>.
 - **July 21, 2025**: We initiated an RFC in the verl community [verl/RFC#2662](https://github.com/volcengine/verl/discussions/2662).
 - **July 2, 2025**: We published the paper [AsyncFlow](https://arxiv.org/abs/2507.01663).

<h2 id="components"> Components</h2>

### Control Plane: Panoramic Data Management

In the control plane, `TransferQueueController` tracks the **production status** and **consumption status** of each training sample as metadata. Once all required data fields are ready (i.e., written to the `TransferQueueStorageManager`), the data sample can be consumed by downstream tasks.

We also track the consumption history for each computational task (e.g., `generate_sequences`, `compute_log_prob`, etc.). Therefore, even when different computational tasks require the same data field, they can consume the data independently without interfering with each other.

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/control_plane.png?raw=true" width="70%">
</p>

To make the data retrieval process more customizable, we provide a `Sampler` class that allows users to define their own data retrieval and consumption logic. Refer to the [Customize](#customize) section for details.

> **load-balancing** capabilities are experimentally supported in the control plane. This design enables us to offload some data management capabilities from single controller. Refer to [#PR70](https://github.com/Ascend/TransferQueue/pull/70) for details.

### Data Plane: Distributed Data Storage

In the data plane, we utilize a pluggable design, enabling TransferQueue to integrate with different storage backends based on user requirements.

Specifically, we provide a `TransferQueueStorageManager` abstraction class that defines the core APIs as follows:

- `async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None`
- `async def get_data(self, metadata: BatchMeta) -> TensorDict`
- `async def clear_data(self, metadata: BatchMeta) -> None`

This class encapsulates the core interaction logic within the TransferQueue system. You only need to write a simple subclass to integrate your custom storage backend. Refer to the [Customize](#customize) section for details.

Currently, we support the following storage backends:

- SimpleStorage: A basic CPU memory storage with minimal data format constraints and ease of use.
- [Yuanrong](https://gitee.com/openeuler/yuanrong-datasystem) (beta, [#PR107](https://github.com/TransferQueue/TransferQueue/pull/107), [#PR96](https://github.com/TransferQueue/TransferQueue/pull/96)): An Ascend native data system that provides hierarchical storage interfaces including HBM/DRAM/SSD.
- [MooncakeStore](https://github.com/kvcache-ai/Mooncake) (beta, [#PR162](https://github.com/TransferQueue/TransferQueue/pull/162)): A high-performance, KV-based hierarchical storage that supports RDMA transport between GPU and DRAM.
- [RayRDT](https://docs.ray.io/en/master/ray-core/direct-transport.html) (alpha, [#PR167](https://github.com/TransferQueue/TransferQueue/pull/167)): Ray's new feature that allows Ray to store and pass objects directly between Ray actors.

Among them, `SimpleStorageUnit` serves as our default storage backend, coordinated by the `AsyncSimpleStorageManager` class. Each storage unit can be deployed on a separate node, allowing for distributed data management.

`SimpleStorageUnit` employs a 2D data structure as follows:

- Each row corresponds to a training sample, assigned a unique index within the corresponding global batch.
- Each column represents the input/output data fields for computational tasks.

This data structure design is motivated by the computational characteristics of the post-training process, where each training sample is generated in a relayed manner across task pipelines. It provides precise addressing capabilities, enabling fine-grained, concurrent data read/write operations in a streaming manner.

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/data_plane.png?raw=true" width="70%">
</p>

### User Interface: High-Level & Low-Level APIs

| Level | Tier | Style | Fine-Grained Access | Streaming | Sampler | Multiple-Backends | 
|---|---|---|---|------------------|---|---|
| High | **KV Interface** ([PR#28](https://github.com/Ascend/TransferQueue/pull/28))| Put/Get/List/Clear | ✓ | ○                | ✗ | ✓ | 
| High |  **StreamingDataLoader** ([PR#23](https://github.com/Ascend/TransferQueue/pull/23)) | PyTorch DataLoader | ✓ | ✓                | ✓ | ✓ | 
| Low |  **TransferQueueClient** | Metadata-based | ✓ | ✓                | ✓ | ✓ | 


#### Key-Value based API

To simplify the usage of TransferQueue, we provide a Redis-style high-level API that exposes most of its advanced features ([PR#28](https://github.com/Ascend/TransferQueue/pull/28)).

**Methods**

- **(async_)kv_put**: Insert/Update a multi-column sample by key, with an optional metadata tag.
- **(async_)kv_batch_put**: Put multiple key-value pairs efficiently in batches.
- **(async_)kv_batch_get**: Retrieve samples (by keys), supporting column selection (by fields).
- **(async_)kv_list**: List keys and tags (metadata) in a partition.
- **(async_)kv_clear**: Remove key-value pairs from storage.

**Key Features**

- **Redis-style Semantics**: Familiar KV interface (Put/Get/List) for a zero learning curve.
- **Fine-grained Access**: Update or retrieve specific fields (columns) within a key (row) without requiring a full-row operation.
- **Partition Isolation**: Logical separation of storage namespaces.
- **Metadata Tags**: Lightweight metadata for status tracking.
- **Pluggable Backends**: Supports multiple backends.

Refer to [tutorials/basic.ipynb](https://github.com/Ascend/TransferQueue/blob/main/tutorial/basic.ipynb) and [tutorials/02_kv_interface.py](https://github.com/Ascend/TransferQueue/blob/main/tutorial/02_kv_interface.py) for detailed usage examples.

#### StreamingDataLoader API

Designed as a drop-in replacement for the standard PyTorch `DataLoader`, this API allows each rank to automatically consume data without single-controller intervention.

In this scenario, `TransferQueueController` serves as a side-controller for data dispatching, with a user-defined `Sampler` class to organize the dataflow.
It encapsulates the complex scheduling and data transfer logic required for various parallelism strategies, seamlessly integrating TransferQueue into existing training workflows and simplifying the development of disaggregated frameworks.

See the [Roadmap](https://github.com/Ascend/TransferQueue/issues/1) and [tutorials/06_streaming_dataloader.py](https://github.com/Ascend/TransferQueue/blob/main/tutorial/06_streaming_dataloader.py) for more details.

#### Low-Level Native API

The native interfaces of TransferQueue are implemented in `TransferQueueClient`. It offers maximum flexibility through native, atomic operations.

Developers can leverage `TransferQueueClient` directly to implement advanced features that require fine-grained control and fully streamed data scheduling, as illustrated in the following tutorials:
- [tutorial/03_metadata_concepts.py](https://github.com/Ascend/TransferQueue/blob/main/tutorial/03_metadata_concepts.py)
- [tutorial/04_understanding_controller.py](https://github.com/Ascend/TransferQueue/blob/main/tutorial/04_understanding_controller.py)
- [tutorial/05_custom_sampler.py](https://github.com/Ascend/TransferQueue/blob/main/tutorial/05_custom_sampler.py)


<h2 id="show-cases">🔥 Showcases</h2>

### Collocated Example

#### verl
The primary motivation for integrating TransferQueue into verl is to **alleviate the data transfer bottleneck of the single controller `RayPPOTrainer`**. Currently, all `DataProto` objects must be routed through `RayPPOTrainer`, resulting in a single-point bottleneck for the entire post-training system.

<p align="center">
  <img src="https://raw.githubusercontent.com/wuxibin89/verl/refs/heads/wuxibin/doc_images/docs/_static/transfer_queue.png" width="100%">
</p>

Official integration with verl is available at [verl/pulls/5401](https://github.com/verl-project/verl/pull/5401), with the design doc at [[RFC] PPOTrainer with TransferQueue Integration](https://github.com/verl-project/verl/issues/5400). You may also refer to our [recipe](https://github.com/Ascend/TransferQueue/blob/main/recipe/simple_use_case/single_controller_demo.py), where we mimic verl usage in a high-level manner. 


### Disaggregated Example

We have experimentally implemented a **standardized, fully-streamed distributed** workflow via TransferQueue. 

By leveraging the `RankAwareSampler` and `StreamingDataLoader` interfaces, we achieve a **streamlined micro-batch-level producer-consumer pipeline**. This design eliminates the need to manually determine data dispatching logic across varying parallelism strategies—a typical complexity in the single-controller paradigm—thereby greatly simplifying framework design. 

Please refer to our [Roadmap](https://github.com/Ascend/TransferQueue/issues/1) and [tutorials/05_streaming_dataloader.py](https://github.com/Ascend/TransferQueue/blob/main/tutorial/05_streaming_dataloader.py) for more details.

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/tq_streaming_dataloader.png?raw=true" width="70%">
</p>

<h2 id="quick-start">🚀 Quick Start</h2>

### Use Python package
```bash
pip install TransferQueue
```

### Install from source code

1. Clone the source code from the GitHub repository
   ```bash
   git clone https://github.com/Ascend/TransferQueue/
   cd TransferQueue
   ```

2. Install from source code
   ```bash
   pip install .
   ```
   
### Build wheel package from source code

1. Clone the source code from the GitHub repository
   ```bash
   git clone https://github.com/Ascend/TransferQueue/
   cd TransferQueue
   ```
   
2. Install dependencies
   ```bash
   pip install build
   ```

3. Build and install
   ```bash
   python -m build --wheel
   pip install dist/*.whl
   ```

<h2 id="performance">📊 Performance</h2>

### Simple Case: Regular Tensor Only
<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/performance_simple_0.1.6.png?raw=true" width="100%">
</p>

### Complex Case: Regular Tensor + NestedTensor + NonTensor
<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/performance_complex_0.1.6.png?raw=true" width="100%">
</p>

> Note: Optimization for MooncakeStore and other backends are still in process. Warmly welcome contributions from the community!

For detailed performance benchmarks, please refer to [this blog](https://www.yuque.com/haomingzi-lfse7/lhp4el/tml8ke0zkgn6roey?singleDoc#).

We also provide a [stress test report](https://www.yuque.com/haomingzi-lfse7/lhp4el/mt0vedqy7c337pgg?singleDoc#) that demonstrates more than **8192 concurrent clients writing 2 TB of data** into TransferQueue across 4 nodes. The system remains stable without any crashes or data loss.

<h2 id="customize"> 🛠️ Customize TransferQueue</h2>

### Define your own data retrieval logic
We provide a `BaseSampler` abstraction class, which defines the following interface:

```python3
@abstractmethod
def sample(
    self,
    ready_indexes: list[int],
    batch_size: int,
    *args: Any,
    **kwargs: Any,
) -> tuple[list[int], list[int]]:
    """Sample a batch of indices from the ready indices.

    Args:
        ready_indexes: List of global indices for which all required fields of the
        corresponding samples have been produced, and the samples are not labeled as
        consumed in the corresponding task.
        batch_size: Number of samples to select
        *args: Additional positional arguments for specific sampler implementations
        **kwargs: Additional keyword arguments for specific sampler implementations

    Returns:
        List of sampled global indices of length batch_size
        List of global indices of length batch_size that should be labeled as consumed
        (will never be retrieved in the future)

    Raises:
        ValueError: If batch_size is invalid or ready_indexes is insufficient
    """
    raise NotImplementedError("Subclasses must implement sample")
```

In this design, we separate data retrieval and data consumption through the two return values, which enables us to easily control sample replacement. We have implemented two reference designs: `SequentialSampler` and `GRPOGroupNSampler`.

The `Sampler` class or instance should be passed to the `TransferQueueController` during initialization. During each `get_meta` call, you can provide dynamic sampling parameters to the `Sampler`.

```python3
from transfer_queue import TransferQueueController, TransferQueueClient, GRPOGroupNSampler, process_zmq_server_info

# Option 1: Pass the sampler class to the TransferQueueController
controller = TransferQueueController.remote(GRPOGroupNSampler)

# Option 2: Pass the sampler instance to the TransferQueueController (if you need custom configuration)
your_own_sampler = YourOwnSampler(config)
controller = TransferQueueController.remote(your_own_sampler)

# Use the sampler
batch_meta = client.get_meta(
    data_fields=["input_ids", "attention_mask"],
    batch_size=8,
    partition_id="train_0",
    task_name="generate_sequences",
    sampling_config={"n_samples_per_prompt": 4}  # Put the required sampling parameters here
)
```

<span style="color: #FF0000;">**Refer to [tutorial/05_custom_sampler.py](https://github.com/Ascend/TransferQueue/blob/main/tutorial/05_custom_sampler.py) for more details.**</span>


### How to integrate a new storage backend

The data plane is organized as follows:
```text
  transfer_queue/
  ├── storage/
  │   ├── __init__.py
  │   │── simple_backend.py             # Default distributed storage backend (SimpleStorageUnit) by TQ 
  │   ├── managers/                     # Managers are upper level interfaces that encapsulate the interaction logic with TQ system.
  │   │   ├── __init__.py
  │   │   ├──base.py                    # TransferQueueStorageManager, KVStorageManager
  │   │   ├──simple_backend_manager.py  # AsyncSimpleStorageManager
  │   │   ├──yuanrong_manager.py        # YuanrongStorageManager
  │   │   ├──mooncake_manager.py        # MooncakeStorageManager
  │   │   └──factory.py                 # TransferQueueStorageManagerFactory
  │   └── clients/                      # Clients are lower level interfaces that directly manipulate the target storage backend.
  │   │   ├── __init__.py
  │   │   ├── base.py                   # TransferQueueStorageKVClient
  │   │   ├── yuanrong_client.py        # YuanrongStorageClient
  │   │   ├── mooncake_client.py        # MooncakeStorageClient
  │   │   ├── ray_storage_client.py     # RayStorageClient
  │   │   └── factory.py                # TransferQueueStorageClientFactory
```

To integrate TransferQueue with a custom storage backend, start by implementing a subclass that inherits from `TransferQueueStorageManager`. This subclass acts as an adapter between the TransferQueue system and the target storage backend. For KV-based storage backends, you can simply inherit from `KVStorageManager`, which can serve as the general manager for all KV-based backends.

Distributed storage backends often come with their own native clients serving as the interface of the storage system. In such cases, a low-level adapter for this client can be written, following the examples provided in the `storage/clients` directory.

Factory classes are provided for both `StorageManager` and `StorageClient` to facilitate easy integration. Adding necessary descriptions of required parameters in the factory class helps enhance the overall user experience.

<h2 id="contribution"> ✏️ Contribution Guide</h2>

<span style="color: #FF0000;">**Contributions are warmly welcome!**</span>

New ideas, feature suggestions, and user experience feedback are all encouraged—feel free to submit issues or PRs. We will respond as soon as possible.

We recommend using pre-commit for better code format.

```bash
# install pre-commit
pip install pre-commit

# run the following command in your repo folder, then fix the check before committing your code
pre-commit install && pre-commit run --all-files --show-diff-on-failure --color=always
```


<h2 id="citation"> Citation</h2>
Please kindly cite our paper if you find this repo is useful:

```bibtex
@article{han2025asyncflow,
  title={AsyncFlow: An Asynchronous Streaming RL Framework for Efficient LLM Post-Training},
  author={Han, Zhenyu and You, Ansheng and Wang, Haibo and Luo, Kui and Yang, Guang and Shi, Wenqi and Chen, Menglong and Zhang, Sicheng and Lan, Zeshun and Deng, Chunshi and others},
  journal={arXiv preprint arXiv:2507.01663},
  year={2025}
}
```