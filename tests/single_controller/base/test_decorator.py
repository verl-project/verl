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

import pytest

import verl.single_controller.base.decorator as decorator_module
from verl.single_controller.base.decorator import (
    DISPATCH_MODE_FN_REGISTRY,
    Dispatch,
    _check_dispatch_mode,
    get_predefined_dispatch_fn,
    register_dispatch_mode,
    update_dispatch_mode,
)


@pytest.fixture
def reset_dispatch_registry():
    # Store original state
    original_registry = DISPATCH_MODE_FN_REGISTRY.copy()
    yield
    # Reset registry after test
    decorator_module.DISPATCH_MODE_FN_REGISTRY.clear()
    decorator_module.DISPATCH_MODE_FN_REGISTRY.update(original_registry)


def test_register_new_dispatch_mode(reset_dispatch_registry):
    # Test registration
    def dummy_dispatch(worker_group, *args, **kwargs):
        return args, kwargs

    def dummy_collect(worker_group, output):
        return output

    register_dispatch_mode("TEST_MODE", dummy_dispatch, dummy_collect)

    # Verify enum extension
    _check_dispatch_mode(Dispatch.TEST_MODE)

    # Verify registry update
    assert get_predefined_dispatch_fn(Dispatch.TEST_MODE) == {
        "dispatch_fn": dummy_dispatch,
        "collect_fn": dummy_collect,
    }
    # Clean up
    Dispatch.remove("TEST_MODE")


def test_update_existing_dispatch_mode(reset_dispatch_registry):
    # Store original implementation
    original_mode = Dispatch.ONE_TO_ALL

    # New implementations
    def new_dispatch(worker_group, *args, **kwargs):
        return args, kwargs

    def new_collect(worker_group, output):
        return output

    # Test update=
    update_dispatch_mode(original_mode, new_dispatch, new_collect)

    # Verify update
    assert get_predefined_dispatch_fn(original_mode)["dispatch_fn"] == new_dispatch
    assert get_predefined_dispatch_fn(original_mode)["collect_fn"] == new_collect
