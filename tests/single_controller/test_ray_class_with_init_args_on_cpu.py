# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Unit tests for RayClassWithInitArgs mixin injection.

Verifies that RayWorkerMixin is automatically injected into Worker subclasses,
pre-decorated classes are rejected, and non-Worker classes pass through unchanged.
Does not require a live Ray cluster or GPU — ray.remote is called only to create
decorated class objects, not to launch remote tasks.
"""

import unittest

import ray

from verl.single_controller.base import Worker
from verl.single_controller.ray.base import (
    RayClassWithInitArgs,
    RayWorkerMixin,
    unwrap_ray_remote,
)


class RayClassWithInitArgsMixinInjectionTest(unittest.TestCase):
    def test_mixin_placed_before_worker_in_mro(self):
        """RayWorkerMixin must appear before Worker in the MRO so its overrides take effect."""

        class Actor(Worker):
            pass

        cia = RayClassWithInitArgs(cls=Actor)
        raw = unwrap_ray_remote(cia.cls)
        mro = raw.__mro__

        mixin_idx = next(i for i, c in enumerate(mro) if c is RayWorkerMixin)
        worker_idx = next(i for i, c in enumerate(mro) if c is Worker)
        self.assertLess(mixin_idx, worker_idx)

    def test_pre_decorated_class_raises(self):
        """@ray.remote-decorated classes must not be passed; use plain classes instead."""

        @ray.remote
        class Actor(Worker):
            pass

        with self.assertRaises(ValueError):
            RayClassWithInitArgs(cls=Actor)

    def test_class_already_having_mixin_not_double_injected(self):
        """Worker subclass already inheriting RayWorkerMixin should not get it added again."""

        class Actor(RayWorkerMixin, Worker):
            pass

        cia = RayClassWithInitArgs(cls=Actor)
        raw = unwrap_ray_remote(cia.cls)

        mro_classes = raw.__mro__
        mixin_count = sum(1 for c in mro_classes if c is RayWorkerMixin)
        self.assertEqual(mixin_count, 1)

    def test_plain_non_worker_class_not_modified(self):
        """Plain classes that do not inherit Worker pass through without mixin injection."""

        class NotAWorker:
            pass

        cia = RayClassWithInitArgs(cls=NotAWorker)
        raw = unwrap_ray_remote(cia.cls)

        self.assertFalse(issubclass(raw, RayWorkerMixin))

    def test_kwargs_preserved_through_injection(self):
        """Constructor kwargs passed to RayClassWithInitArgs are preserved."""

        class Actor(Worker):
            def __init__(self, x=0):
                pass

        cia = RayClassWithInitArgs(cls=Actor, x=42)

        self.assertEqual(cia.kwargs, {"x": 42})

    def test_original_class_name_preserved(self):
        """The injected class should retain the original class name for observability."""

        class MySpecialWorker(Worker):
            pass

        cia = RayClassWithInitArgs(cls=MySpecialWorker)
        raw = unwrap_ray_remote(cia.cls)

        self.assertEqual(raw.__name__, "MySpecialWorker")

    def test_registered_methods_visible_on_injected_class(self):
        """Methods decorated with @register must still be accessible after injection."""
        from verl.single_controller.base.decorator import MAGIC_ATTR, Dispatch, register

        class Actor(Worker):
            @register(dispatch_mode=Dispatch.ONE_TO_ALL)
            def compute(self):
                pass

        cia = RayClassWithInitArgs(cls=Actor)
        raw = unwrap_ray_remote(cia.cls)

        method = getattr(raw, "compute", None)
        self.assertIsNotNone(method)
        self.assertTrue(hasattr(method, MAGIC_ATTR))


if __name__ == "__main__":
    unittest.main()
