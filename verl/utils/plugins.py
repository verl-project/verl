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
"""Discover and load external verl plugins.

A plugin is any installed package that declares a ``verl.plugins`` entry point;
importing it applies its registrations (rollout adapters, model patches, ...).
``load_plugins`` runs on the driver and is installed as the Ray
``worker_process_setup_hook`` so plugins load in every process. It is a no-op
when no plugins are installed.
"""

import importlib.metadata
import logging

logger = logging.getLogger(__name__)

_PLUGIN_GROUP = "verl.plugins"
_loaded = False


def load_plugins() -> None:
    """Import every package registered under the ``verl.plugins`` entry point."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for ep in importlib.metadata.entry_points(group=_PLUGIN_GROUP):
        try:
            ep.load()
        except Exception:
            logger.exception("verl: failed to load plugin %r (%s)", ep.name, ep.value)
