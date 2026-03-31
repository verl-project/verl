# 数学原理降维拆解：PPO、GRPO、KL 与 Rollout Correction

这篇的目标不是把公式抄一遍，而是做两件事：

1. 把公式和源码路径一一对上
2. 把那些“字母都认识，但脑子里没画面”的部分，硬拆成直觉

如果你愿意，完全可以一边看这篇，一边同时打开：

- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/ray_trainer.py`
- `docs/algo/ppo.md`
- `docs/algo/grpo.md`
- `docs/algo/rollout_corr_math.md`

## 1. 为什么 LLM 的 RL 数学一开始总让人别扭？

因为大模型训练里经常会出现一种“错位”：

- 模型是一个 token 一个 token 地在做决策
- reward 却经常像是对整段回答做一次总评分
- 但梯度更新又必须回到 token 级别去做

所以 `verl` 在做的一件核心事情，就是把“整段回答好不好”翻译成“哪些 token 值得鼓励、哪些 token 值得克制”。

```{mermaid}
flowchart LR
    A[Prompt] --> B[逐 token 生成]
    B --> C[完整回答]
    C --> D[Reward / Score]
    D --> E[Token 级 reward / advantage]
    E --> F[策略更新]
```

## 2. PPO：一句公式先记住

PPO 最核心的裁剪目标可以写成：

$$
L_{\text{PPO}}(\theta) = -\mathbb{E}\left[\min\left(r_t(\theta) A_t,\ \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]
$$

其中

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$

### 这些符号在现实里分别代表什么？

| 符号 | 现实含义 |
| --- | --- |
| $\pi_\theta$ | 当前正在被优化的策略 |
| $\pi_{\text{old}}$ | 这一批更新中固定住的“旧策略 / 近端锚点” |
| $a_t$ | 第 $t$ 个 token 的动作选择 |
| $s_t$ | 做出这个选择之前模型已经看到的上下文 |
| $A_t$ | 这个动作相对预期来说到底是“赚了”还是“亏了” |
| $\epsilon$ | 更新的安全绳，不让策略一步迈太大 |

### 先用超市店长类比一下

你可以把它想成店长在给店员打分：

- 昨天的服务习惯是旧策略
- 今天的新服务习惯是当前策略
- advantage 表示这次服务比预期更好还是更差
- PPO 的 clipping 像在说：**可以改进，但别一下午就把整个店的服务风格改得面目全非**

### 再算一个三秒钟就能看懂的小例子

假设：

- 旧策略下某个 token 的概率是 `0.20`
- 新策略下这个 token 的概率升到 `0.30`
- 该 token 的 advantage 是 `+2.0`
- clip 范围 $\epsilon = 0.2$

那就有：

$$
r_t = 0.30 / 0.20 = 1.5
$$

如果不裁剪，收益项就是：

`1.5 * 2.0 = 3.0`

但 PPO 会把比例裁到 `1.2`，于是实际采用的收益更接近：

$$
1.2 \times 2.0 = 2.4
$$

这背后的直觉非常朴素：

> “这个 token 的确做得好，可以奖励；但不能因为这一次好，就把整套策略猛拽过去。” 

**源码锚点**

- `docs/algo/ppo.md`
- `verl/trainer/ppo/core_algos.py`

## 3. GAE：怎么把奖励更平滑地往前传？

`verl` 在 `compute_gae_advantage_return(...)` 里实现了 GAE。

可以把它概括成两层递推：

$$
\delta_t = r_t + \gamma V_{t+1} - V_t
$$

$$
A_t = \delta_t + \gamma \lambda A_{t+1}
$$

### 每个字母的现实意义

| 符号 | 直觉含义 |
| --- | --- |
| $r_t$ | 第 $t$ 个位置的即时奖励 |
| $V_t$ | critic 对“从这里往后还值多少”的估计 |
| $\gamma$ | 你有多在意更后面的收益 |
| $\lambda$ | 你想把未来信息平滑回传多少 |

### 菜市场版小算例

假设你在菜市场买菜，一共发生了三步：

1. 先开口问价
2. 中间砍价
3. 最后成交

虽然“成交满意”这个结果是在最后才出现的，但你不会说前面两步完全没贡献。

假设：

- 三个 response token 的 reward 是 `[0.0, 0.0, 1.0]`
- critic 估计 value 是 `[0.3, 0.4, 0.2]`
- $\gamma = 1.0$，$\lambda = 0.9$

那你倒着看：

- 最后一个 token 的优势大概是 `1.0 - 0.2 = 0.8`
- 倒数第二个 token 会通过递推拿到一部分未来功劳
- 第一个 token 也会被“追责 / 追功”，只是更平滑一些

GAE 的本质就是：

> **别把功劳和锅都只甩给最后一个 token，要把信用沿着序列往前传，但传得别太躁。**

在代码里，你会看到它真的就是从 response 最后一个位置往前倒着累 `lastgaelam`。

**源码锚点**

- `verl/trainer/ppo/core_algos.py` -> `compute_gae_advantage_return`
- `verl/trainer/ppo/ray_trainer.py` -> `compute_advantage`

## 4. GRPO：不训练 critic，直接在同组答案里比高低

GRPO 的关键变化是：**不再依赖单独的 critic 来估值**。

它的玩法是：

1. 同一个 prompt 采样出多个答案
2. 给每个答案算 reward
3. 在这个组里做相对比较

一个常见写法是：

$$
A_i = \frac{R_i - \mu_g}{\sigma_g + \varepsilon}
$$

如果走 DrGRPO 风格、不做标准差归一化，则更接近：

$$
A_i = R_i - \mu_g
$$

其中：

- $R_i$ 是第 $i$ 个答案的分数
- $\mu_g$ 是这一组答案的平均分
- $\sigma_g$ 是这一组答案的标准差

### 这件事到底直觉在哪？

你可以想成同一道题，老师让 4 个学生都写一遍答案。

GRPO 不再问：

> “这个位置的精确 value 该是多少？”

而是问：

> “这 4 份答案里，谁明显比组平均更好，谁明显比组平均更差？”

### 一个四个答案的小例子

某个 prompt 生成 4 个回答，最终得分是：

- A：`1.0`
- B：`0.8`
- C：`0.2`
- D：`0.0`

平均分是 `0.5`。

如果先不做标准差归一化，只做中心化，那 advantage 就是：

- A：`+0.5`
- B：`+0.3`
- C：`-0.3`
- D：`-0.5`

解释立刻就清楚了：

- 高于平均分的答案应该被鼓励
- 低于平均分的答案应该被抑制

这就是为什么很多人会把 GRPO 理解成：

> **不用 critic，而是在同组 sibling responses 里做相对排名。**

### 为什么 `rollout.n > 1` 是硬前提？

因为你如果每个 prompt 只采样一个答案，那就根本没有“组内比较”可言。

**源码锚点**

- `docs/algo/grpo.md`
- `verl/trainer/ppo/core_algos.py` -> `compute_grpo_outcome_advantage`
- `verl/trainer/ppo/ray_trainer.py` -> `compute_advantage`

## 5. KL 控制：为什么要给策略拴一根绳？

在 RLHF / post-training 场景里，训练后的策略通常不能完全放飞，它还需要和一个 reference policy 保持大致接近。

`verl` 提供了多种 KL 机制，其中一个重要路径是 **把 KL 当作 reward penalty**。

高层上可以写成：

$$
\text{token\_level\_reward} = \text{token\_level\_score} - \beta \cdot \text{KL term}
$$

这意味着：

- 回答好，reward 会变大
- 但如果和 reference 偏得太远，reward 会被扣回来

### Adaptive KL Controller：把它想成空调温控

`AdaptiveKLController` 会动态调整系数 $\beta$。代码里的逻辑大致是：

$$
\beta \leftarrow \beta \cdot \left(1 + \operatorname{clip}(\frac{\text{current\_kl}}{\text{target\_kl}} - 1, -0.2, 0.2) \cdot \frac{n_{\text{steps}}}{\text{horizon}}\right)
$$

看起来很长，但直觉只有一句：

- 当前 KL 太高，就把惩罚系数调大
- 当前 KL 太低，就把惩罚系数调小
- 调的时候别一把拧到头

这不就是空调温控器吗？

### 一个小例子

假设：

- target KL = `0.10`
- current KL = `0.14`
- 当前系数 = `0.001`

那控制器就会判断“偏离有点大”，于是把系数往上抬一点。下一轮 reward penalty 更强，策略就会更保守一些。

**源码锚点**

- `verl/trainer/ppo/core_algos.py` -> `AdaptiveKLController`
- `verl/trainer/ppo/ray_trainer.py` -> `apply_kl_penalty`

## 6. Rollout Correction：为什么 `verl` 要搞三策略？

这是整个仓库里最容易劝退新人的部分之一，但它其实特别有必要。

在 rollout correction 的理论里，会区分三种策略：

- $\pi_{\text{rollout}}$ -- 真正拿来采样数据的行为策略
- $\pi_{\text{old}}$ -- 用来做 PPO clipping 的近端锚点策略
- $\pi_\theta$ -- 当前正在被训练更新的策略

```{mermaid}
flowchart LR
    A[pi_rollout<br/>行为策略] -->|负责采样| D[训练 batch]
    B[pi_old<br/>近端锚点] -->|负责 PPO 比例参考| D
    C[pi_theta<br/>当前策略] -->|被持续优化| D
```

为什么非要拆成三个？

因为真实系统里，“拿来采样数据的策略”和“你想拿来当 PPO 锚点的策略”，经常根本不是一个东西。

常见原因包括：

- rollout 和训练后端不同
- 数值精度不同
- 异步 worker 带来策略陈旧
- 使用 replay buffer / off-policy 数据

### 两个最关键的比值

#### 第一件事：修正行为策略偏差

$$
\rho_t = \frac{\pi_{\text{old}}(a_t \mid s_t)}{\pi_{\text{rollout}}(a_t \mid s_t)}
$$

它的含义是：

> “这条样本是由一个稍微不一样的行为策略采出来的，那我训练时要不要把它重新加权？”

#### 第二件事：控制参数更新幅度

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$

这就是标准 PPO 的近端比例。

### 一个最小数值例子

假设某个 token 的概率分别是：

- rollout 后端算出来：`0.25`
- proximal anchor 算出来：`0.20`
- 当前策略现在变成：`0.22`

那：

- 行为偏差修正比值是 `0.20 / 0.25 = 0.8`
- PPO 更新比值是 `0.22 / 0.20 = 1.1`

这意味着：

- 这条样本会因为 rollout 略微“高估”了它而被下调一点权重
- 但当前策略相对锚点只偏了一点点，所以 PPO 更新仍然是温和的

这正是三策略最漂亮的地方：

> **一个比值负责修正“数据从哪来”的偏差，另一个比值负责约束“参数往哪走”的步子。**

**源码锚点**

- 理论文档：`docs/algo/rollout_corr_math.md`
- 训练器逻辑：`verl/trainer/ppo/ray_trainer.py`
- 辅助实现：`verl/trainer/ppo/rollout_corr_helper.py`

## 7. 数学和代码到底怎么对齐？

| 数学概念 | 主要源码位置 |
| --- | --- |
| KL controller | `verl/trainer/ppo/core_algos.py` |
| in-reward KL | `verl/trainer/ppo/ray_trainer.py` |
| GAE | `verl/trainer/ppo/core_algos.py` |
| GRPO 组内 advantage | `verl/trainer/ppo/core_algos.py` |
| old / ref / rollout log-prob 路径 | `verl/trainer/ppo/ray_trainer.py` |
| rollout correction 理论 | `docs/algo/rollout_corr_math.md` |

## 8. 最短总结

如果你只想把主脉络背下来，那就记住这五句：

- **PPO**：好动作要鼓励，但更新别迈太大。
- **GAE**：把最终奖励平滑地往前传，不要只奖惩最后一个 token。
- **GRPO**：不训练 critic，直接在同组答案里做相对比较。
- **KL 控制**：别让新策略和 reference 偏离得太离谱。
- **Rollout correction**：把“谁采了这条数据”和“谁做 PPO 锚点”这两件事拆开。

这五句基本就是 `verl` 的数学脊梁。

如果这时你再回去对照 [`source-code-tour.md`](./source-code-tour.md) 和 `core_algos.py`，公式就不会再像天书，而会像“源码在执行的一套非常具体的账本规则”。
