import asyncio
import multiprocessing as mp
import os
import uuid

import torch

from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver, BucketedWeightSender


def run_sender(zmq_handle, original_weights):
    # Set a tiny bucket size (1MB) to force chunking
    sender = BucketedWeightSender(zmq_handle=zmq_handle, bucket_size_mb=1, use_shm=True)
    
    async def weight_gen():
        for k, v in original_weights.items():
            yield k, v
            
    asyncio.run(sender.async_send_weights(weight_gen()))


def run_receiver(zmq_handle, original_weights, result_queue):
    # Force use CPU
    receiver = BucketedWeightReceiver(zmq_handle=zmq_handle, device=torch.device("cpu"), use_shm=True)
    
    received_weights = {}
    
    def on_received(weights):
        for name, tensor in weights:
            received_weights[name] = tensor.clone()
            
    receiver.receive_weights(on_received)
    
    all_matched = True
    for name, orig_tensor in original_weights.items():
        if name not in received_weights:
            print(f"Missing tensor: {name}")
            all_matched = False
            continue
            
        recv_tensor = received_weights[name]
        is_match = torch.allclose(orig_tensor, recv_tensor)
        print(f"Tensor {name}: shape={orig_tensor.shape}, Match = {is_match}")
        if not is_match:
            all_matched = False
            
    result_queue.put(all_matched)


def test_chunking_large_tensors():
    unique_id = uuid.uuid4().hex
    zmq_handle = f"ipc:///tmp/test-zmq-{unique_id}.sock"
    
    # 1MB bucket = 1048576 bytes ~ 262144 float32 numbers
    test_weights = {
        "small_1": torch.randn((128, 512), dtype=torch.float32),           # 0.25 MB, fits in 1 bucket
        "large_1": torch.randn((5 * 128, 512), dtype=torch.float32),       # 1.25 MB, spans 2 buckets
        "large_2": torch.ones((10 * 128, 512), dtype=torch.float32) * 5,   # 2.5 MB, spans 3 buckets
        "small_2": torch.randn((64, 256), dtype=torch.float32),            # 0.06 MB, fits in bucket
    }
    
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    
    p_sender = ctx.Process(target=run_sender, args=(zmq_handle, test_weights))
    p_receiver = ctx.Process(target=run_receiver, args=(zmq_handle, test_weights, result_queue))
    
    p_receiver.start()
    # give receiver a moment to bind/connect
    import time
    time.sleep(1)
    p_sender.start()
    
    p_sender.join()
    p_receiver.join()
    
    success = result_queue.get()
    assert success, "Weight transfer failed to match original tensors."
    print("All tests passed! Chunked weight transfer is successful.")


if __name__ == "__main__":
    test_chunking_large_tensors()