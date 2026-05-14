
## 简介

昇腾全面支持 Verl 使用与开发，本文档全面介绍了如何在华为昇腾芯片 NPU 上使用 Verl。

Last updated: 05/14/2026.

## 目录结构

```
ascend_tutorial/
├── get_start/                     # 快速入门指南
├── feature_support/               # 特性支持说明
├── model_support/                 # 模型支持说明
├── dev_guide/                     # 开发指南
├── faq/                           # 常见问题解答
└── contribution_guide/            # 社区贡献指南
```
## 最新消息
- [verl-ascend-recipe 建仓](https://github.com/verl-project/verl-ascend-recipe) - 新增昇腾recipe
- [verl on ascend 2026Q2 roadmap](https://github.com/verl-project/verl/issues/5526) - 2026Q2 RoadMap 已发布

## 快速开始
- [Docker 构建使用指南](./get_start/dockerfile_build_guidance.rst) - 构建并使用昇腾环境的 Docker 镜像  
- [自定义环境安装](./get_start/install_guidance.rst) - 在昇腾 NPU 上自定义安装 Verl                                                    
- [快速上手](./get_start/quick_start.rst) - 快速上手在昇腾 NPU 上运行 Verl

## 特性支持说明

- [verl特性支持](./feature_support/support_verl_features.md) - 支持的verl框架特性/参数列表
- [NPU特性支持](./feature_support/npu_advance_features/npu_advance_features.md) - NPU相关常用特性/环境变量说明

## 模型支持说明

- [模型与算法支持说明](./model_support/model_and_algorithm_support.md) - 支持的模型/算法列表
- [最佳实践示例](./model_support/examples) - 最佳实践与模型部署示例


## 开发指南

- [模型开发](./dev_guide/model_dev) 
    - [模型迁移](./dev_guide/model_dev/transfer_to_npu_guide.md) - 模型迁移指南
    - [训练参数与指标](./dev_guide/model_dev/parameters_and_metrics.md) - 训练参数与指标
    - [模型评测](./dev_guide/model_dev/evaluation.md) - 模型评测指南
- [精度调试](./dev_guide/precision_analysis) 
    - [精度分析](./dev_guide/precision_analysis/precision_alignment.md) - 精度对齐指南
    - [精度调试器](./dev_guide/precision_analysis/precision_debugger.md) - 精度问题排查工具
- [性能调优](./dev_guide/performance) 
    - [性能分析](./dev_guide/performance/ascend_performance_analysis_guide.md) - 性能分析指南
    - [性能调优](./dev_guide/performance/perf_tuning_on_ascend.rst) - 性能调优指南
    - [profiling采集](./dev_guide/performance/ascend_profiling_zh.md) - profiling 工具使用指南


## 支持与反馈

如果您在使用过程中遇到问题，欢迎通过以下方式获取帮助：

1. 查看 [FAQ](./faq/faq.rst)
2. 在 GitHub Issues 中提交问题
3. 联系昇腾技术支持

## 贡献指南
- [verl 社区贡献](../contributing) - Verl 社区贡献指南
- [昇腾 CI 指南](./contribution_guide/ascend_ci_guide_zh.rst) - 昇腾环境 CI 配置与测试

## 相关资源

- [Verl 官方文档](https://verl.readthedocs.io/)
- [昇腾开发者社区](https://www.hiascend.com/)