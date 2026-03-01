# AtroposLib Configuration

This document outlines the configuration options available for the `atroposlib` library, primarily defined using Pydantic models.
These configurations are often managed via a command-line interface built using `pydantic-cli`, especially when using the `serve` command provided by environment classes inheriting from `BaseEnv`.

## Base Environment Configuration (`atroposlib.envs.base.BaseEnvConfig`)

Basic environment configuration settings.

| Parameter                        | Type                     | Default                                         | Description                                                                                                |
| :------------------------------- | :----------------------- | :---------------------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| `group_size`                     | `int`                    | `4`                                             | How many responses are grouped together for scoring.                                                       |
| `max_num_workers`                | `int`                    | `-1`                                            | Maximum number of workers to use. `-1` calculates from `max_num_workers_per_node`.                       |
| `max_eval_workers`               | `int`                    | `16`                                            | Maximum number of workers to use for evaluation.                                                           |
| `max_num_workers_per_node`       | `int`                    | `8`                                             | Maximum number of workers to use per node.                                                                 |
| `steps_per_eval`                 | `int`                    | `100`                                           | Number of steps to take before evaluating.                                                                 |
| `max_token_length`               | `int`                    | `2048`                                          | Maximum token length used in generations.                                                                  |
| `eval_handling`                  | `EvalHandlingEnum`       | `EvalHandlingEnum.STOP_TRAIN`                   | How to handle evaluations (`STOP_TRAIN`, `LIMIT_TRAIN`, `NONE`).                                             |
| `eval_limit_ratio`               | `float`                  | `0.5`                                           | Ratio of training workers to limit during evals (used if `eval_handling` is `LIMIT_TRAIN`).                |
| `inference_weight`               | `float`                  | `1.0`                                           | Inference weight. Set to `-1` to ignore if doing something special.                                        |
| `batch_size`                     | `int`                    | `-1`                                            | Batch size for training. Usually set by the trainer via the API.                                           |
| `max_batches_offpolicy`          | `int`                    | `3`                                             | Maximum number of off-policy batches to have in the queue.                                                 |
| `tokenizer_name`                 | `str`                    | `"NousResearch/DeepHermes-3-Llama-3-3B-Preview"` | Hugging Face tokenizer to use.                                                                             |
| `use_wandb`                      | `bool`                   | `True`                                          | Whether to use Weights & Biases for logging.                                                               |
| `rollout_server_url`             | `str`                    | `"http://localhost:8000"`                       | URL of the rollout server (FastAPI interface).                                                             |
| `total_steps`                    | `int`                    | `1000`                                          | Total number of steps to run.                                                                              |
| `wandb_name`                     | `str | None`             | `None`                                          | Name to be grouped by in WandB.                                                                            |
| `num_rollouts_to_keep`           | `int`                    | `32`                                            | Number of rollouts to display on WandB.                                                                    |
| `num_rollouts_per_group_for_logging` | `int`                | `1`                                             | Number of rollouts per group to keep for logging. `-1` keeps all.                                          |
| `ensure_scores_are_not_same`     | `bool`                   | `True`                                          | Ensure that scores within a group are not identical (usually `True`).                                      |
| `data_path_to_save_groups`       | `str | None`             | `None`                                          | Path to save generated groups as a JSONL file. If set, groups will be written here.                         |
| `min_items_sent_before_logging`  | `int`                    | `2`                                             | Minimum number of items sent to the API before logging metrics. `0` or less logs every time.             |

## Server Manager Configuration (`atroposlib.envs.server_handling.server_manager.ServerManagerConfig`)

Settings for the `ServerManager`.

| Parameter | Type    | Default | Description                                       |
| :-------- | :------ | :------ | :------------------------------------------------ |
| `slurm`   | `bool`  | `True`  | Whether the environment is running on SLURM.      |
| `testing` | `bool`  | `False` | If `True`, uses mock OpenAI data for testing. |

## Server Baseline Configuration (`atroposlib.envs.server_handling.server_manager.ServerBaseline`)

Baseline configuration used by `ServerManager` if a list of `APIServerConfig` is not provided, particularly for setting up local or SLURM-based server discovery.

| Parameter                  | Type    | Default   | Description                                                                                             |
| :------------------------- | :------ | :-------- | :------------------------------------------------------------------------------------------------------ |
| `timeout`                  | `int`   | `1200`    | Timeout for the request in seconds.                                                                     |
| `num_max_requests_at_once` | `int`   | `512`     | Maximum number of concurrent requests (training). Divide this by the generation `n` parameter.          |
| `num_requests_for_eval`    | `int`   | `64`      | Maximum number of concurrent requests for evaluation.                                                   |
| `model_name`               | `str`   | `default` | Model name to use when calling inference servers.                                                     |
| `rolling_buffer_length`    | `int`   | `1000`    | Length of the rolling buffer to store server metrics (like request timings, attempts).                   |

## OpenAI Server Configuration (`atroposlib.envs.server_handling.openai_server.APIServerConfig`)

Configuration for individual OpenAI-compatible API servers (including local SGLang/vLLM instances).

| Parameter                  | Type         | Default   | Description                                                                                             |
| :------------------------- | :----------- | :-------- | :------------------------------------------------------------------------------------------------------ |
| `api_key`                  | `str \| None` | `None`    | API key for OpenAI API. Use `"x"` or any non-empty string for local servers that don't require auth.    |
| `base_url`                 | `str \| None` | `None`    | URL of the API endpoint. `None` for official OpenAI API, otherwise the local server URL (e.g., `http://localhost:9004/v1`). |
| `timeout`                  | `int`        | `1200`    | Timeout for the request in seconds.                                                                     |
| `num_max_requests_at_once` | `int`        | `512`     | Maximum number of concurrent requests (training). Divide this by the generation `n` parameter.          |
| `num_requests_for_eval`    | `int`        | `64`      | Maximum number of concurrent requests for evaluation.                                                   |
| `model_name`               | `str`        | `default` | The model name to use. Required for both OpenAI and local models (e.g., `"gpt-4"`, `"NousResearch/..."`). |
| `rolling_buffer_length`    | `int`        | `1000`    | Length of the rolling buffer to store server metrics (like request timings, attempts).                   |
| `n_kwarg_is_ignored`       | `bool`       | `False`   | If the n kwarg is ignored by the API you are using, set this to True.                                   |
