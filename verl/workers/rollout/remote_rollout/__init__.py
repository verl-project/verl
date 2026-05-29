"""Rollout servers that delegate `generate` to a remote RL backend.

Each sub-package wraps an in-process rollout server (e.g.
``vLLMHttpServer``) and routes prompts to the backend's `generate`
endpoint instead of running the model locally. Used by adapters that
co-train sampling and training in a separate cluster.
"""
