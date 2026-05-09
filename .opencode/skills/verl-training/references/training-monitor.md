# 训练监控指南

## 实时日志

### 查看训练日志

```bash
# 实时查看训练日志
tail -f $LOG_FILE

# 查看最新日志
tail -100 $LOG_FILE
```

### 日志内容示例

```
[2024-03-15 10:23:45] ========== Epoch 1/5 ==========
[2024-03-15 10:23:46] Step 100/1000 | Loss: 1.234 | Reward: 0.65 | LR: 5.0e-07
[2024-03-15 10:23:47] Step 200/1000 | Loss: 1.123 | Reward: 0.68 | LR: 5.0e-07
[2024-03-15 10:23:48] Step 300/1000 | Loss: 0.987 | Reward: 0.72 | LR: 5.0e-07
...
```

## 性能汇总

训练完成后，自动生成性能汇总:

```
========================================
          训练性能汇总
========================================
模型: Qwen2.5-7B-Instruct
算法: GRPO
训练轮数: 5
GPU数量: 16

---------- 训练结果 ----------
最终 Loss: 0.3421
最终 Reward: 0.7823
训练时长: 2h 34m

---------- 性能指标 ----------
平均训练速度: 128 samples/s
GPU利用率: 92.3%
显存使用: 28.5GB/卡
```

## Loss曲线

### 使用TensorBoard查看

```bash
# 启动tensorboard
tensorboard --logdir ./logs/tb_logs

# 访问地址
# http://localhost:6006
```

### 生成Loss曲线图

训练过程中会生成以下曲线:

- `loss_curve.png`: 训练loss曲线
- `reward_curve.png`: 奖励曲线
- `learning_rate.png`: 学习率变化

### 手动绘图

```python
import matplotlib.pyplot as plt
import json

# 读取训练日志中的loss数据
losses = []
with open('./logs/training_log.txt') as f:
    for line in f:
        if 'Loss:' in line:
            loss = float(line.split('Loss:')[1].split('|')[0].strip())
            losses.append(loss)

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.savefig('./logs/loss_curve.png')
plt.show()
```

## 训练指标说明

| 指标 | 说明 |
|------|------|
| Loss | 训练损失值 |
| Reward | 奖励分数 |
| KL | KL散度 |
| Entropy | 熵值 |
| GPU利用率 | GPU使用百分比 |
| 显存使用 | 每卡显存使用量 |
| 训练速度 | samples/second |