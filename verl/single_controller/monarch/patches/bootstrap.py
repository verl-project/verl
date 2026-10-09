# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Temporary bootstrap compatibility patch installing the Monarch storage codec.

Remove this entry point when the supported SDK no longer needs that codec fix."""

from __future__ import annotations

import runpy

from .codec import install_monarch_storage_codec


def main() -> None:
    install_monarch_storage_codec()
    runpy.run_module("monarch._src.actor.bootstrap_main", run_name="__main__")


if __name__ == "__main__":
    main()
