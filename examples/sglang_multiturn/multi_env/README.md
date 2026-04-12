# Multi-Environment Tool-Use Example

Demonstrates per-instance tool environment routing in `ToolAgentLoop`.

Each dataset row carries `extra_info.tool_env_id` and `extra_info.db_path`.
`ToolAgentLoop` resolves per-instance tools via a manifest and loads a
per-instance shared DB automatically — no custom subclass needed.

## Environments

| env_id | tool | DB schema |
|--------|------|-----------|
| `inventory_store` | `check_stock` | `{"products": {"laptop": {"stock": 10, "price": 999.99}}}` |
| `bank_account` | `check_balance` | `{"accounts": {"alice": {"balance": 5000.0}}}` |

## Quick Start

```bash
# From project root:
bash examples/sglang_multiturn/multi_env/run_multi_env_grpo.sh
```

## Key Files

- `config/tool_env_manifest.yaml` — Maps `tool_env_id` → tool config path
- `config/multi_env_grpo.yaml` — Training config with `tool_env_manifest_path` set
