# Speculator Design Doc
背景参考：
[verifier_last_hidden_states](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/)

投机解码（Speculator）相关实现的设计与职责划分如下。

## 设计目标
训练流程与具体投机解码算法解耦

- 解耦：Trainer/Engine 不依赖具体 speculator 架构细节。
- 可扩展：新增 speculator 类型仅需新增 adapter + 模型实现。

```json
examples
└── sft
    ├── speculator
    │   └── README.md
    └── gsm8k
        ├── run_qwen_05_speculator.sh
        ├── run_qwen_05_speculator_engine.sh
        └── run_qwen_05_speculator_megatron.sh

verl
├── models(模型层)
│   └── speculator
│       ├── mlp_speculator.py
│       └── lstm_speculator.py
├── trainer
│   ├── fsdp_sft_trainer.py
│   └── speculators
│       ├── config.py
│       ├── interface.py（adapter 基类）
│       ├── lstm_adapter.py
│       └── mlp_adapter.py
├── workers
│   └── engine
│       ├── fsdp
│       │   └── transformer_impl.py(FSDPEngineWithLMHeadAndSpeculator)
│       └── megatron
│           └── transformer_impl.py(MegatronEngineWithLMHeadAndSpeculator)
└── utils
    ├── fsdp_utils.py
    ├── megatron_utils.py
    └── checkpoint
        ├── fsdp_checkpoint_manager.py（增加 speculator checkpoint 的 save/load）
        └── megatron_checkpoint_manager.py
```
## 模块与职责

### 1) 模型层（Speculator Models）
路径：`verl/models/speculator/`
职责：
- `speculator.py`: 定义模型基类与配置。
- `lstm_speculator.py`: `ArcticLSTMSpeculator` 与 `ArcticLSTMSpeculatorConfig`。
- `mlp_speculator.py`: `MLPSpeculator` 与 `MLPSpeculatorConfig`。
- 仅负责 `nn.Module` 的参数定义与前向逻辑（`reset_parameters()` / `forward()`）。

### 2) 适配层（Speculator Adapters）
路径：`verl/trainer/speculators/`
职责：
- `interface.py`: 定义抽象接口 + 通用实现（checkpoint、公共处理逻辑）。
- `lstm_adapter.py`: `LSTMSpeculatorAdapter`。
- `mlp_adapter.py`: `MLPSpeculatorAdapter`。

子类需实现的抽象接口：
- `build_speculator_module(self, model)`: 构建 speculator module。
- `get_optimizer_params(self)`: 返回需要优化的参数集合。
- `compute_speculator_loss()`: 计算 speculator loss。


Adapter 选择方式：

- `build_speculator_adapter` 读取 `model.speculator_adapter`。
- 支持字符串 FQN，或字典 `{fqn: ...}` / `{path: ..., name: ...}`。
   ```yaml
   verl/trainer/config/sft_trainer_engine.yaml
   speculator_adapter:
    fqn: verl.trainer.speculators.lstm_adapter.LSTMSpeculatorAdapter
    ....
   ```


### 3) Trainer 

路径：
- FSDP: `verl/trainer/fsdp_sft_trainer.py`

职责：
- 训练流程编排：在 `_build_model_optimizer`、`_compute_loss_and_backward`、`save_checkpoint`、`load_checkpoint` 等阶段接入 speculator 逻辑。


### 4) Engine 

路径：
- FSDP: `verl/workers/engine/fsdp/transformer_impl.py`
- Megatron: `verl/workers/engine/megatron/transformer_impl.py`

职责：
- 提供 `language_model_with_speculator` 引擎实现（`FSDPEngineWithLMHeadAndSpeculator` / `MegatronEngineWithLMHeadAndSpeculator`）。
- 介于 Trainer 与模型之间的“执行引擎”，利用 `speculator_adapter` 接口完成 speculator 的构建、前向与 loss 计算。
- checkpoint：保存/加载 speculator 权重与配置 `config.json`。



### 5) Checkpoint 处理

行为：
- speculator 权重**单独保存**到 `checkpoint/speculator/pytorch_model.bin`。
- speculator 配置保存到 `checkpoint/speculator/config.json`。
- 加载时从 `speculator/` 目录恢复 speculator 权重。

## 类图（核心接口与集成关系）

```text
                      +-----------------------------+
                      |      SpeculatorAdapter      |
                      | (trainer/speculators/...)   |
                      +-----------------------------+
                      | interface:                  |
                      | - build_speculator_module() |
                      | - compute_speculator_loss() |
                      | - compute_speculator_logits()|
                      +-------------+---------------+
                                    ^
                inherits            |            inherits
                                    |
        +---------------------------+---------------------------+
        |                                                           |
+-----------------------------+                       +-----------------------------+
|   LSTMSpeculatorAdapter     |                       |    MLPSpeculatorAdapter     |
| (trainer/speculators/...)   |                       | (trainer/speculators/...)   |
| implements:                 |                       | implements:                 |
| - build_speculator_module() |                       | - build_speculator_module() |
| - compute_speculator_loss() |                       | - compute_speculator_loss() |
| - compute_speculator_logits()|                      | - compute_speculator_logits()|
+--------------+--------------+                       +--------------+--------------+
               | builds                                        | builds
               v                                               v
   +-----------------------------+                 +-----------------------------+
   |     ArcticLSTMSpeculator    |                 |        MLPSpeculator        |
   | (models/speculator/...)     |                 | (models/speculator/...)     |
   | nn.Module interface:        |                 | nn.Module interface:        |
   | - forward()                 |                 | - forward()                 |
   | - reset_parameters()        |                 | - reset_parameters()        |
   +-----------------------------+                 +-----------------------------+

   +-----------------------------+                 +-----------------------------+
   |    SpeculatorConfigBase     |<----------------|     MLPSpeculatorConfig     |
   | (trainer/speculators/...)   |      inherits   | (models/speculator/...)     |
   | - save()                    |                 | - save()                    |
   +-----------------------------+                 +-----------------------------+

   +-----------------------------+                 +-------------------------------+
   |    SpeculatorConfigBase     |<----------------| ArcticLSTMSpeculatorConfig    |
   | (trainer/speculators/...)   |      inherits   | (trainer/speculators/...)     |
   | - save()                    |                 | - save()                      |
   +-----------------------------+                 +-------------------------------+
```



## 扩展新 Speculator 的流程

1) 在 `verl/models/speculator/` 实现新模型（仅 forward + reset_parameters）。
2) 在 `verl/trainer/speculators/` 新建 adapter：
   - 实现构建模型 + loss 逻辑 
   - 继承 `SpeculatorConfigBase` 并实现 `save()`
3) 在配置中设置：
   - `model.speculator`（超参）
   - `model.speculator_adapter.fqn` 指向新 adapter
