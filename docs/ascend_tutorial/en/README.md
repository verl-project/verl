# Ascend Tutorial
## Introduction

Ascend fully supports the usage and development of verl. This documentation provides a comprehensive guide on how to use verl on Huawei Ascend NPUs.

Last updated: 05/14/2026.

## Directory Structure

```
en/
├── get_start/                     # Quick start guide
├── feature_support/               # Feature support descriptions
├── model_support/                 # Model support descriptions
├── dev_guide/                     # Developer guide
├── faq/                           # Frequently asked questions
└── contribution_guide/            # Community contribution guide
```
## Latest News
- [verl-ascend-recipe repository created](https://github.com/verl-project/verl-ascend-recipe) - New Ascend recipe added
- [verl on Ascend 2026Q2 roadmap](https://github.com/verl-project/verl/issues/5526) - 2026Q2 roadmap released

## Quick Start
- [Ascend Image Guide](./get_start/dockerfile_build_guidance.rst) - Build and use Docker images for the Ascend environment  
- [Ascend Installation Guide](./get_start/install_guidance.rst) - Custom installation of verl on Ascend NPUs                                                    
- [Ascend Quick Start Guide](./get_start/quick_start.rst) - Quick start for running verl on Ascend NPUs

## Feature Support

- [Training Configuration Parameters and Metrics](./dev_guide/model_dev/parameter_and_metrics.md) - List of supported verl framework features/parameters
- [NPU Advanced Features Guide](./feature_support/npu_advance_features.md) - Common NPU-related features/environment variables

## Model Support

- [NPU Model and Algorithm Support](./model_support/model_and_algorithm_support.md) - List of supported models/algorithms
- [Best Practice Examples](./model_support/examples) - Best practices and model deployment examples


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
    - [Profiling Data Collection Guide](./dev_guide/performance/ascend_profiling.rst) - Profiling tool usage guide


## Support and Feedback

If you encounter issues during use, you can get help through the following channels:

1. Check the [NPU FAQ](./faq/faq.rst)
2. Submit issues in GitHub Issues
3. Contact Ascend technical support

## Contributing
- [verl Community Contribution](../../contributing) - verl community contribution guide
- [NPU-CI Setup Guide](./contribution_guide/ascend_ci_guide.rst) - Ascend environment CI configuration and testing

## Related Resources

- [verl Official Documentation](https://verl.readthedocs.io/)
- [Ascend Developer Community](https://www.hiascend.com/)
