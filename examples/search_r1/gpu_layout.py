"""Select the one- or four-GPU retrieval subset without remapping training's devices."""

import re


def retrieval_gpu_layout(gpus, indices):
    indices = list(indices)
    if len(indices) not in (1, 4) or len(set(indices)) != len(indices):
        raise ValueError("Retrieval requires one or four distinct GPU indices")
    if any(i < 0 or i >= len(gpus) for i in indices):
        raise ValueError("Retrieval GPU index is outside the training allocation")
    selected = []
    for service_index, training_index in enumerate(indices):
        uuid = gpus[training_index].get("uuid", "")
        if not isinstance(uuid, str) or not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid):
            raise ValueError("A CUDA GPU UUID is required for safe retrieval device selection")
        selected.append({"service_index": service_index, "training_index": training_index, "uuid": uuid})
    if len({row["uuid"] for row in selected}) != len(indices):
        raise ValueError("Retrieval GPU UUIDs must be distinct")
    return {"training_gpu_count": len(gpus), "retrieval_devices": selected,
            "cuda_visible_devices": ",".join(row["uuid"] for row in selected)}
