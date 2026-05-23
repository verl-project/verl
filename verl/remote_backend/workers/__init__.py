"""Per-backend worker implementations.

Each sub-package houses the worker class that `RemoteBackendTrainer`
picks up when a given remote backend is selected. The split mirrors
the per-backend split of adapters under :mod:`verl.workers.remote_client`.
"""
