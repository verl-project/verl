---
name: image-list
description: 查询quay.io上可用的VERL镜像列表，支持过滤和版本更新
license: MIT
Compatibility: opencode
metadata:
  audience: developers
  workflow: image_list
---

# IMAGE-LIST Skill

查询quay.io上可用的VERL镜像列表。

## 前置条件

- 能够访问 quay.io API
- 如无法访问，可使用代理

## 功能

### 1. 查询可用镜像

调用Quay API获取镜像标签列表：

```bash
export HTTP_PROXY=<代理地址>
export HTTPS_PROXY=<代理地址>
curl -s "https://quay.io/api/v1/repository/ascend/verl?public=true" | \
  python3 -c "import sys,json; tags=json.load(sys.stdin)['tags']; print('\n'.join(sorted(tags.keys())))"
```

### 2. 过滤展示

支持按条件过滤：

| 过滤条件 | 示例 |
|---------|------|
| CANN版本 | 8.5.0 / 8.3.rc1 / 8.2.rc1 |
| 芯片类型 | 910b / a3 |
| 推理后端 | vllm / sglang |
| VERL版本 | latest / v0.7.1 / v0.7.0 |

### 3. 更新参考文档

如发现新镜像，提示更新 `references/image_versions.md`：

```
发现新镜像: verl-8.5.0-910b-ubuntu22.04-py3.11-v0.7.2
建议: 更新 references/image_versions.md
```

## 使用示例

### 示例1: 查询全部镜像

```
用户: 查看可用镜像
助手: 查询中...
展示:
| 标签 | 类型 |
|------|------|
| verl-8.2.rc1-910b-ubuntu22.04-py3.11-latest | vLLM A2 |
| verl-8.2.rc1-910b-ubuntu22.04-py3.11-v0.6.1 | vLLM A2 |
| verl-8.2.rc1-a3-ubuntu22.04-py3.11-latest | vLLM A3 |
| ... |
结果: 共14个镜像
```

### 示例2: 按条件过滤

```
用户: 查看8.5.0的镜像
助手: 过滤中...
展示:
| 标签 |
|------|
| verl-8.5.0-910b-ubuntu22.04-py3.11-latest |
| verl-8.5.0-910b-ubuntu22.04-py3.11-v0.7.1 |
| verl-8.5.0-a3-ubuntu22.04-py3.11-latest |
| verl-8.5.0-a3-ubuntu22.04-py3.11-v0.7.1 |
```

### 示例3: 使用代理查询

```
用户: 无法访问quay.io，使用代理查询
助手: 使用代理 http://127.0.0.1:58232 查询...
展示: (镜像列表)
```

### 示例4: 发现新版本

```
用户: 查看v0.7.2版本
助手: 查询中...
发现新镜像: verl-8.5.0-910b-ubuntu22.04-py3.11-v0.7.2
建议: 该版本不在references中，是否需要更新？
```

## 错误处理

| 错误 | 说明 | 处理方式 |
|------|------|----------|
| api_failed | API请求失败 | 尝试使用代理 |
| no_proxy | 无代理且无法访问 | 提示手动查看网页 |
| empty_result | 无匹配结果 | 提示放宽过滤条件 |