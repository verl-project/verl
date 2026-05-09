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

from verl.utils.dataloader import shutdown_dataloader_workers, shutdown_dataloaders


class _FakeIterator:
    def __init__(self):
        self.shutdown_calls = 0

    def _shutdown_workers(self):
        self.shutdown_calls += 1


class _FailingIterator:
    def _shutdown_workers(self):
        raise RuntimeError("shutdown failed")


class _FakeDataLoader:
    def __init__(self, iterator=None):
        self._iterator = iterator


def test_shutdown_dataloader_workers_calls_iterator_shutdown():
    iterator = _FakeIterator()
    dataloader = _FakeDataLoader(iterator)

    assert shutdown_dataloader_workers(dataloader) is True

    assert iterator.shutdown_calls == 1
    assert dataloader._iterator is None


def test_shutdown_dataloader_workers_is_noop_without_iterator():
    dataloader = _FakeDataLoader()

    assert shutdown_dataloader_workers(dataloader) is False
    assert dataloader._iterator is None


def test_shutdown_dataloaders_skips_none_and_clears_iterators():
    iterator = _FakeIterator()
    dataloader = _FakeDataLoader(iterator)

    shutdown_dataloaders(None, dataloader)

    assert iterator.shutdown_calls == 1
    assert dataloader._iterator is None


def test_shutdown_dataloader_workers_does_not_raise_on_shutdown_error():
    dataloader = _FakeDataLoader(_FailingIterator())

    assert shutdown_dataloader_workers(dataloader) is False
    assert dataloader._iterator is None


def test_shutdown_stateful_dataloader_stops_worker_processes():
    from torchdata.stateful_dataloader import StatefulDataLoader

    dataloader = StatefulDataLoader(list(range(64)), batch_size=4, num_workers=2)
    iterator = iter(dataloader)
    next(iterator)
    workers = list(getattr(iterator, "_workers", []))

    assert workers
    assert any(worker.is_alive() for worker in workers)
    assert shutdown_dataloader_workers(dataloader) is True
    for worker in workers:
        worker.join(timeout=2)

    assert dataloader._iterator is None
    assert not any(worker.is_alive() for worker in workers)
