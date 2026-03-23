

## MedRAGChecker Follow-up Fork

This repository is the follow-up `VERL` codebase for our MedRAGChecker line of work.
It extends `verl` with the training and evaluation components needed for checker-centric,
medical multi-turn RL, where an agent first searches for evidence and then decides when and
how to invoke a checker.

Compared with upstream `verl`, this fork mainly adds:

- Medical multi-turn `search + checker` tool integration for SGLang rollouts
- A MedRAG-style checker service and the corresponding `verl` tool wrapper
- Guarded checker invocation logic so the checker complements retrieval instead of replacing it
- Reward / evaluation utilities for checker usage, support-contradiction signals, and tool statistics
- Training and ablation scripts for `search_r1_like` style medical checker experiments

If you are looking for the MedRAGChecker-specific workflow, the main entry points are:

- Training config: [`examples/sglang_multiturn/config/search_multiturn_grpo_explicitcheck.yaml`](examples/sglang_multiturn/config/search_multiturn_grpo_explicitcheck.yaml)
- Tool config: [`examples/sglang_multiturn/config/tool_config/medical_search_checker_tool_config.yaml`](examples/sglang_multiturn/config/tool_config/medical_search_checker_tool_config.yaml)
- Main training script: [`examples/sglang_multiturn/search_r1_like/run_qwen2.5-7b_search_checker_ablation_2gpu.sh`](examples/sglang_multiturn/search_r1_like/run_qwen2.5-7b_search_checker_ablation_2gpu.sh)
- Evaluation script: [`evaluate/evaluate_search_r1.py`](evaluate/evaluate_search_r1.py)
- Checker service: [`search_r1_preprocess/checker_medrag.py`](search_r1_preprocess/checker_medrag.py)

