# 镜像版本参考

> 最后更新时间: 2026-04-14
> 数据来源: https://quay.io/repository/ascend/verl?tab=tags

## vLLM后端 - A2 (910b)

| CANN版本 | VERL版本 | 镜像标签 | 状态 |
|---------|---------|----------|------|
| 8.2.RC1 | latest | verl-8.2.rc1-910b-ubuntu22.04-py3.11-latest | |
| 8.2.RC1 | v0.6.1 | verl-8.2.rc1-910b-ubuntu22.04-py3.11-v0.6.1 | |
| 8.3.RC1 | latest | verl-8.3.rc1-910b-ubuntu22.04-py3.11-latest | |
| 8.3.RC1 | v0.7.0 | verl-8.3.rc1-910b-ubuntu22.04-py3.11-v0.7.0 | |
| 8.5.0 | latest | verl-8.5.0-910b-ubuntu22.04-py3.11-latest | 默认 |
| 8.5.0 | v0.7.1 | verl-8.5.0-910b-ubuntu22.04-py3.11-v0.7.1 | |

## vLLM后端 - A3

| CANN版本 | VERL版本 | 镜像标签 | 状态 |
|---------|---------|----------|------|
| 8.2.RC1 | latest | verl-8.2.rc1-a3-ubuntu22.04-py3.11-latest | |
| 8.2.RC1 | v0.6.1 | verl-8.2.rc1-a3-ubuntu22.04-py3.11-v0.6.1 | |
| 8.3.RC1 | latest | verl-8.3.rc1-a3-ubuntu22.04-py3.11-latest | |
| 8.3.RC1 | v0.7.0 | verl-8.3.rc1-a3-ubuntu22.04-py3.11-v0.7.0 | |
| 8.5.0 | latest | verl-8.5.0-a3-ubuntu22.04-py3.11-latest | |
| 8.5.0 | v0.7.1 | verl-8.5.0-a3-ubuntu22.04-py3.11-v0.7.1 | |

## SGLang后端 - A2 (910b)

| CANN版本 | 镜像标签 |
|---------|----------|
| 8.3.RC1 | verl-sglang-8.3.rc1-910b-ubuntu22.04-py3.11-latest |

## SGLang后端 - A3

| CANN版本 | 镜像标签 |
|---------|----------|
| 8.3.RC1 | verl-sglang-8.3.rc1-a3-ubuntu22.04-py3.11-latest |

## 拉取镜像命令

```bash
# 拉取镜像
docker pull quay.io/ascend/verl:<镜像标签>

# 完整示例
docker pull quay.io/ascend/verl:verl-8.5.0-910b-ubuntu22.04-py3.11-latest
docker pull quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest

# 如需代理
docker -e HTTP_PROXY=<代理地址> -e HTTPS_PROXY=<代理地址> pull quay.io/ascend/verl:<镜像标签>
```

## 查询最新镜像

如需查看最新可用镜像列表，可以访问：
- https://quay.io/repository/ascend/verl?tab=tags

或使用API（需要代理）：
```bash
export HTTP_PROXY=<代理地址>
export HTTPS_PROXY=<代理地址>
curl -s "https://quay.io/api/v1/repository/ascend/verl?public=true" | python3 -c "import sys,json; tags=json.load(sys.stdin)['tags']; print('\n'.join(sorted(tags.keys())))"
```