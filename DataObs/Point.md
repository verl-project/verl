
1.IFD (Instruction Following Difficulty) 指标：
定义：计算模型在“有 Prompt”条件下的 Loss 与“无 Prompt”条件下 Answer 的 Loss 之比。
逻辑：如果比值很高，说明这段 Answer 对模型来说很难通过指令直接预测，包含更多“学习价值”。


3.Informativeness (信息密度)：
定义：Answer 的长度与其实际携带的实体（Entity）、概念（Concept）数量之比。
计算：可以用 NLP 工具抽取关键字密度，或计算压缩比。

4.Logic Tree Depth (逻辑链深度)：
计算：对于带有 CoT（思维链）的数据，统计推理步骤的数量（Step count）或关键词（如 "Therefore", "Because", "Firstly"）的频率。


这个熵什么的还有相似度，xx要改

换数据集

eval现在不太可以
