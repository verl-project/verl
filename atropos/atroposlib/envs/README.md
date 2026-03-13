# Base Environment (`BaseEnv`)

The `BaseEnv` class (located in `atroposlib/envs/base.py`) provides a foundation for creating custom reinforcement learning environments that interact with Atropos. When creating your own environment, you will typically subclass `BaseEnv` and implement several key methods.

## Design philosophy

Every environment in Atropos is a microservice that generates rollout data async from whatever trainer you attach to it. Environments (possibly many at once) send data to the [Atropos API server](https://github.com/NousResearch/atropos/tree/main/atroposlib/api) which sequesters rollout data. Your trainer of choice grabs batches of data from the API and backpropagates.

![image](https://github.com/user-attachments/assets/1cc8634a-319d-4add-9db7-c8b3acc272ad)

Unlike other popular alternatives like Gymnasium which model environments as [MDPs](https://arxiv.org/abs/2412.05265), we think about environments as dataloaders and do not make any assumptions about how a trajectory is produced. For multi-agent for example, this means our design is agnostic to [AEC](https://pettingzoo.farama.org/api/aec/) vs. [POSG](https://pettingzoo.farama.org/api/parallel/) - both are supported out of the box.
To achieve this generality, our environment abstraction deviates from other open source alternatives in several key ways.

- **Inference-scoring fusion**: A popular design choice in open-source LLM RL trainers is to separate inference and scoring into independent abstractions. While this makes a lot of sense for single-turn environments like one-shot MCQA, we found that this led to awkwardness in multi-turn setups with process rewards. As such, we assume the existence of a single method `collect_trajectories` which is responsible for both inference and scoring. Users are still welcome to call separate inference and scoring methods from within `collect_trajectories`.

- **Groups as atomic units of data**: A natural choice for a data atom in RL is a single trajectory. However, many popular RL methods for fine-tuning LLMs such as DPO and GRPO involve packing contrastive data into the same batch. As such the most fundamental dataloading method in our abstraction is not `collect_trajectory` (singular) but `collect_trajectories` (plural). We do not enforce any definition of what a "group" is other than a set of rollouts. Although a "group" is most commonly constructed by generated multiple rollouts starting from the same initial state (as in DPO and GRPO), a user could just as easily pack `n` similar-sounding problems with very different solutions into a group. For cases like PPO where advantages don't depend on group statistics a user can simply use group size 1.

- **Environments return tokens (not messages!)**: One of the most peculiar design choices we made was that at least for text-only environments, environments are responsible for tokenization. This gives us the flexibility to assign token-level rewards and to mix completions-based (e.g. autocomplete suggestion accept/reject) and chat-based (e.g. instruct-model code generation) environments together in the same training run. For cases like multimodal where a OpenAI-formatted message list needs to be passed to a transformers `AutoProcessor`, we support a `list[dict]`-valued `messages` key within our group abstraction [ScoredDataGroup](https://github.com/NousResearch/atropos/blob/a282604baac8dbb3b201f992cfc889ee1e5a0f4a/atroposlib/envs/base.py#L55).

## Core Methods to Implement

These methods **must** be implemented in your subclass:

*   **`async def setup(self)`**: This method is called once at the beginning of the environment's lifecycle (`env_manager`). Use it for any initial setup required for your specific environment, such as loading datasets, initializing models, or connecting to external resources.

*   **`async def get_next_item(self) -> Item`**: This method is responsible for generating or retrieving the next piece of data (prompt, state, etc.) that will be used to start a new trajectory collection. If no more items are available or should be generated, it can return `None` to signal the worker to pause.

*   **`async def collect_trajectories(self, item: Item) -> Tuple[Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]], List[Any | None]], List[Item]]`**: The default implementation of this method runs `collect_trajectory` (see below) multiple times in parallel (controlled by `group_size`). You can override this if you have a more efficient way to generate the entire group of responses/trajectories at once based on the input `item` (e.g. the `n` parameter in the OpenAI chat completions API) or some desired coupling of rollouts (e.g. via MCTS). It should return the collected group data and a list of backlog items.

*   **`async def collect_trajectory(self, item: Item) -> Tuple[Any | ScoredDataItem | None, List[Item]]`**: If the rollouts for your environment can be sampled independently, the easiest way to implement GRPO-style grouping is to define the `collect_trajectory` method and use the default implementation of `collect_trajectories` which runs `group_size` instances of `collect_trajectory` in parallel. This method defines the logic for a *single* logical trajectory collection step based on the input `item`.
    *   **Return value**: It returns a tuple containing:\
        1.  The ScoredDataItem for this step (one trajectory). This data can be processed further in `postprocess_histories`, if you require additional filtering right before sending to the API.
        2.  A list of new `Item` objects to be added to the backlog for future processing (e.g., follow-up prompts).\
    * **Should I define `collect_trajectory` or override `collect_trajectories`?** If you've got some way to generate your group more efficiently than a bunch of separate but parallel calls to `collect_trajectory`, or if your rollouts aren't independent as in MCTS, you should override `collect_trajectories`. If simplicity and iteration speed is more valuable than efficiency (e.g. at the start of a development cycle) and your rollouts are independent then `collect_trajectory` is for you.

*   **`async def evaluate(self, *args, **kwargs)`**: This method is called periodically (controlled by `steps_per_eval` in the config) to perform evaluation runs. You define the evaluation logic here. The base class provides an example using `self.eval_workers` for parallel evaluation tasks, but you can implement any evaluation procedure suitable for your environment.

## Optional Methods to Override

These methods have default implementations or are optional based on your needs:

*   **`async def collect_trajectories(self, item: Item) -> Tuple[Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]], List[Any | None]], List[Item]]`**: The default implementation of this method runs `collect_trajectory` multiple times in parallel (controlled by `group_size`). You can override this *instead* of `collect_trajectory` if you have a more efficient way to generate the entire group of responses/trajectories at once based on the input `item`. It should return the collected group data and a list of backlog items.

*   **`async def postprocess_histories(self, trajectories: Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]`**: This method is called after `collect_trajectories` and before the data is sent to the training server. It receives the collected data from the parallel runs (or your custom `collect_trajectories` implementation). Use this to perform final processing, scoring, or formatting you may require before sending to the server. You usually won't need this.

*   **`async def wandb_log(self, wandb_metrics: Optional[Dict] = None)`**: Called periodically to log metrics to Weights & Biases. If you override this to add custom metrics, **ensure you call `super().wandb_log(wandb_metrics)`** at the end of your implementation. This ensures that the base class's performance metrics and rollout tables are also logged.
    ```python
    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}
        # Add your custom metrics
        wandb_metrics['my_custom_metric'] = calculate_my_metric()
        # ... add more metrics

        # Call the parent method to log base metrics
        await super().wandb_log(wandb_metrics)
    ```

*   **`save_checkpoint(self, step, data=None)`**: The base class calls this method automatically at checkpoint intervals determined by the server. It saves the provided `data` dictionary (which you might populate with environment-specific state) to a JSON file. You can override this to customize *what* data is saved or *how* it's saved (e.g., using a different format or location), but the triggering mechanism remains automatic.

*   **`@classmethod config_init(cls) -> Tuple[BaseEnvConfig, Union[ServerBaseline, List[APIServerConfig]]]`**: This class method is used by the default `get_cli_serve_config_cls` implementation to get the initial environment configuration (`BaseEnvConfig` subclass) and server configurations (`ServerBaseline` or `List[APIServerConfig]`) when setting up the `serve` command. The default implementation returns `cls.env_config_cls(), ServerBaseline()`. You might override this if your environment requires different default configurations or specific server setups (like multiple `APIServerConfig` instances) when run via the CLI `serve` command.

*   **`async def cleanup(self)`**: Called after each call to `handle_env`. You can implement this for any cleanup needed after processing a single item, though it's often not required.

## Overrideable Class Variables

These class-level variables in `BaseEnv` can be overridden in your subclass to customize its behavior:

*   **`name: Optional[str]`**:
    *   Default: `None`
    *   Purpose: You can set a string name for your environment. This name is used by default for `wandb_name` in the `BaseEnvConfig` if not otherwise specified, influencing how runs are grouped or named in Weights & Biases. It can also be useful for general identification or logging purposes.

*   **`env_config_cls: Type[BaseEnvConfig]`**:
    *   Default: `BaseEnvConfig`
    *   Purpose: This variable holds the Pydantic model class that will be used for your environment's configuration. If your environment requires custom configuration fields beyond what `BaseEnvConfig` offers, you should create a new class that inherits from `BaseEnvConfig` (or a subclass thereof) and assign it to `env_config_cls`. This allows the CLI and other parts of the system to correctly parse and manage your environment's specific settings.
    ```python
    from pydantic import Field
    from atroposlib.envs import BaseEnv, BaseEnvConfig

    class MyEnvConfig(BaseEnvConfig):
        my_custom_param: str = Field(default="default_value", description="A custom parameter for MyEnv")

    class MyEnv(BaseEnv):
        env_config_cls = MyEnvConfig
        name = "MyCustomEnvironment"
        # ... other implementations
    ```

*   **`server_cls: Type[APIServer]`**:
    *   Default: `APIServer`
    *   Purpose: Specifies the class to be used for managing interactions with API servers (e.g., inference endpoints). Should mostly be used for developing additional API interfaces, but if you need a nonstandard way of connecting with an existing API you can use this to easily slot in any modifications you need.

## Provided Functionality

`BaseEnv` provides several helpful features:

*   **Parallel Trajectory Collection (`collect_trajectories`)**: The base implementation runs your `collect_trajectory` method multiple times in parallel (based on `group_size`) and gathers the results. You can override `collect_trajectories` directly for custom group generation logic (see Optional Methods).
*   **Server Interaction**: Handles registration with the rollout server, fetching configuration (like `batch_size`), sending scored data (`handle_send_to_api` with retries), and status updates.
*   **WandB Integration**: Sets up WandB logging (if enabled) based on server information and provides the `wandb_log` hook for custom metrics (remember to call `super().wandb_log()`). It uses helper methods `add_rollouts_for_wandb` (to temporarily store rollout data) and `create_rollout_table` (to format the data into a `wandb.Table`). You can override either of these helpers for custom logging behavior (e.g., changing what data is stored or how the final table is structured).
*   **Checkpointing**:
    *   The environment automatically triggers checkpoint saves based on the `checkpoint_interval` received from the server, calling the `save_checkpoint` method (see Optional Methods).
    *   `load_checkpoint(self)`: Loads data from the checkpoint file corresponding to the environment's `curr_step`. It attempts to restore attributes of the environment object based on the keys in the loaded JSON data. This is called automatically if `curr_step > 0` during registration.
*   **Worker Management**: Manages asynchronous worker tasks for collecting trajectories (`add_train_workers`, `handle_env`).
*   **Performance Monitoring**: Tracks and logs various performance statistics (task durations, worker counts, etc.).
*   **CLI Integration**: Provides a `cli()` class method using `pydantic-cli` to easily create command-line interfaces for your environment (e.g., `python your_env_module.py serve --port 8001 ...`). See `get_cli_serve_config_cls` and `get_cli_process_config_cls`.

By implementing the required methods and optionally overriding others, you can create diverse environments that leverage the distributed training infrastructure provided by the `Atropos` framework.
