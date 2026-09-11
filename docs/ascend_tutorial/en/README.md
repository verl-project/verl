# Ascend Tutorial
## Introduction

Ascend fully supports using and developing verl. This document comprehensively introduces how to use verl on Huawei Ascend chip NPUs.

Last updated: 05/14/2026.

## Directory structure

```
zh/
├── get_start/                     # Quick start guide
├── feature_support/               # Feature support description
├── model_support/                 # Model support description
├── dev_guide/                     # Development guide
├── faq/                           # Frequently asked questions
└── contribution_guide/            # Community contribution guide
```
## Latest news
- [verl-ascend-recipe repository created](https://github.com/verl-project/verl-ascend-recipe) - New Ascend recipe added
- [verl on Ascend 2026Q2 roadmap](https://github.com/verl-project/verl/issues/5526) - 2026Q2 roadmap released

## Quick Start
- [Ascend Image Guide](./get_start/dockerfile_build_guidance.rst) - Build and use Docker images for the Ascend environment  
- [Ascend Installation Guide](./get_start/install_guidance.rst) - Custom installation of verl on Ascend NPUs                                                    
- [Ascend Quick Start Guide](./get_start/quick_start.rst) - Quickly get started running verl on Ascend NPUs

## Feature Support

- [Training configuration parameters and metrics description](./dev_guide/model_dev/parameter_and_metrics.md) - List of supported verl framework features and parameters
- [NPU advanced features guide](./feature_support/npu_advance_features.md) - Description of common NPU-related features and environment variables

## Model Support

- [NPU Model and Algorithm Support](./model_support/model_and_algorithm_support.md) - List of supported models and algorithms
- [Best Practice Examples](./model_support/examples) - Examples of best practices and model deployment


## Developer Guide

- [Model Development](./dev_guide/model_dev) 
    - [Model Migration to NPU Guide](./dev_guide/model_dev/transfer_to_npu_guide.md) - Model migration guide
    - [Training Configuration Parameters and Metrics](./dev_guide/model_dev/parameter_and_metrics.md) - Training parameters and metrics
    - [Model Evaluation](./dev_guide/model_dev/evaluation.md) - Model evaluation guide
- [Precision Debugging](./dev_guide/precision_analysis) 
    - [Precision Alignment Guide](./dev_guide/precision_analysis/precision_alignment.md) - Precision alignment guide
    - [Precision Debugger](../en/dev_guide/precision_analysis/precision_debugger.md) - Precision issue troubleshooting tool
- [Performance Tuning](./dev_guide/performance) 
    - [Ascend Performance Analysis Guide](./dev_guide/performance/ascend_performance_analysis_guide.md) - Performance analysis guide
    - [Ascend Performance Tuning Guide](./dev_guide/performance/perf_tuning_on_ascend.rst) - Performance tuning guide
    - [Profiling Collection Guide](./dev_guide/performance/ascend_profiling.rst) - Profiling tool usage guide


## Support and feedback

If you encounter any issues during use, you can get help through the following methods:

1. View the [NPU FAQ](./faq/faq.rst)
2. Submit an issue in GitHub Issues
3. Contact Ascend technical support

## Contribution Guide
- [verl community contribution](../../contributing) - verl community contribution guide
- [NPU-CI addition guide](./contribution_guide/ascend_ci_guide.rst) - Ascend environment CI configuration and testing

## Related resources

- [verl official documentation](https://verl.readthedocs.io/)
- [Ascend Developer Community](https://www.hiascend.com/)
