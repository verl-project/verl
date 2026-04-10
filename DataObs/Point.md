
1.IFD (Instruction Following Difficulty) 指标：
定义：计算模型在“有 Prompt”条件下的 Loss 与“无 Prompt”条件下 Answer 的 Loss 之比。
逻辑：如果比值很高，说明这段 Answer 对模型来说很难通过指令直接预测，包含更多“学习价值”。


2.PPL (Perplexity) 困惑度：
定义：利用一个预训练好的高质量模型（如 Llama 3 或 GPT-4 蒸馏出的模型）计算 Answer 的平均负对数似然。
逻辑：极高 PPL 可能代表噪声，极低 PPL 可能代表过于平庸的废话（如 "I don't know"）。 计算

3.Informativeness (信息密度)：
定义：Answer 的长度与其实际携带的实体（Entity）、概念（Concept）数量之比。
计算：可以用 NLP 工具抽取关键字密度，或计算压缩比。


4.Logic Tree Depth (逻辑链深度)：
计算：对于带有 CoT（思维链）的数据，统计推理步骤的数量（Step count）或关键词（如 "Therefore", "Because", "Firstly"）的频率。


Response Diversity (回复多样性)：
计算：计算子集内所有 Answer 的 Self-BLEU 或相似度矩阵的平均值。
逻辑：如果 SFT 数据过于单一，RL 阶段模型容易陷入局部最优，失去探索能力。

Verbosity Bias (字数偏好)：
计算：统计 Answer 的平均字数。
逻辑：RL 阶段模型往往会通过“刷字数”来骗取 Reward。如果在 SFT 阶段就引入了过长或过短的偏好，会直接影响 RL 的稳定性。


这个熵什么的还有相似度，xx要改