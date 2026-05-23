"""Adapter clients for remote RL backends.

Each module here implements a ``RemoteBackend`` (see :mod:`verl.remote_backend`)
that talks to an out-of-process training+rollout cluster owned by a
third-party library (e.g. ``arctic_training``). Importing the module
registers the adapter with :class:`verl.remote_backend.RemoteBackendRegistry`.
"""
