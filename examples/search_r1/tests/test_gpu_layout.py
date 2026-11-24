import importlib.util
from pathlib import Path

import pytest

from examples.search_r1.gpu_layout import retrieval_gpu_layout


def devices(count):
    return [{"uuid": f"GPU-{i:08x}-0000-0000-0000-000000000000"} for i in range(count)]


@pytest.mark.parametrize("count", [4, 8])
def test_retrieval_selects_four_uuids_without_changing_training_devices(count):
    gpus = devices(count)
    original = [g.copy() for g in gpus]
    layout = retrieval_gpu_layout(gpus, [3, 1, 0, 2])
    assert gpus == original
    assert layout["training_gpu_count"] == count
    assert layout["cuda_visible_devices"].split(",") == [gpus[i]["uuid"] for i in [3, 1, 0, 2]]
    assert [r["training_index"] for r in layout["retrieval_devices"]] == [3, 1, 0, 2]


@pytest.mark.parametrize("indices", [[0, 1, 2], [0, 1, 2, 2], [0, 1, 2, 8], [-1, 1, 2, 3]])
def test_retrieval_rejects_invalid_subsets(indices):
    with pytest.raises(ValueError):
        retrieval_gpu_layout(devices(8), indices)


def test_retrieval_rejects_missing_or_duplicate_uuid():
    for value in [None, "GPU-invalid", devices(4)[0]["uuid"]]:
        gpus = devices(4)
        gpus[3]["uuid"] = value
        with pytest.raises(ValueError):
            retrieval_gpu_layout(gpus, [0, 1, 2, 3])


@pytest.mark.parametrize("count", [4, 8])
def test_telemetry_tracks_all_allocated_gpus_in_cuda_order(count):
    script = Path(__file__).resolve().parents[4]/"scripts/collect_rollout_telemetry.py"
    spec = importlib.util.spec_from_file_location("rollout_telemetry_layout", script)
    telemetry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(telemetry)
    uuids = [g["uuid"] for g in devices(count)]
    # NVML can enumerate additional GPUs and use a different order from CUDA.
    body = "\n".join(f"{g['uuid']}, {15-i}, NVIDIA H200, {100+i}, 143771, 50"
                     for i, g in reversed(list(enumerate(devices(count+1)))))
    rows, mapping = telemetry.normalize_gpu_sample(body, uuids, count)
    assert [r["uuid"] for r in mapping] == uuids
    assert [int(row.split(",")[2]) for row in rows] == [100+i for i in range(count)]
    with pytest.raises(AssertionError):
        telemetry.normalize_gpu_sample(body, [], count)


def test_single_gpu_layout_preserves_uuid_and_bounds():
    gpus=devices(1)
    layout=retrieval_gpu_layout(gpus,[0])
    assert layout['cuda_visible_devices']==gpus[0]['uuid']
    assert len(layout['retrieval_devices'])==1
    with pytest.raises(ValueError):retrieval_gpu_layout(gpus,[1])
