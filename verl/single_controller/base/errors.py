# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RPC and placement errors for the common Runtime surface."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from builtins import ExceptionGroup as ExceptionGroup
else:
    from exceptiongroup import ExceptionGroup as ExceptionGroup


class RPCError(RuntimeError):
    """Base class for RPC infrastructure failures."""


class RPCTransportError(RPCError):
    """Report that transport cannot determine the remote invocation outcome."""


class RPCUnavailableError(RPCTransportError):
    """Report that the backend confirmed the fixed remote target is unavailable."""


class RPCTimeoutError(RPCTransportError, TimeoutError):
    """Report that a submit deadline expired while the remote outcome is unknown."""


class RPCRemoteError(RPCError):
    """Fallback for an application exception that RPC cannot reconstruct."""


class PlacementUnavailableError(RuntimeError):
    """Report that the requested exact process topology cannot be allocated."""
