# BF16 control v15 - receiver drain fix

Same purpose as v14 (single-variable `PRECISION_MODE=bf16` control for the W4A4
response-length plateau), rebuilt because v15 changes runtime payload.

## What v14 found

`PRECISION_MODE=bf16` had never been executed. Both v14 attempts hung with no
traceback - the 8-node job burned its full 5h limit at step 0. py-spy showed the
cause exactly:

- all 4 senders frozen in `async_send_weights` at
  `bucketed_weight_transfer.py:138` (`socket.recv()`, waiting for a bucket ACK),
  all on the same tensor `model.layers.0.mlp.experts.77.gate_proj.weight`;
- all 4 receivers already **returned** from `update_weights_from_ipc`, back in
  vLLM's `worker_busy_loop`.

So the consumer failed, and `receive_weights` skipped the ACK and closed its
socket, stranding the sender. verl awaits the receiver future only *after* the
send loop, so the real exception could never surface. The iterator path used by
real NVFP4 already drained on failure; the legacy path did not. That asymmetry
is fixed, so the underlying consumer error now reports instead of hanging.

The consumer error itself is still unknown - v15 exists to surface it.

```bash
./submit.sh probe && ./submit.sh build && ./submit.sh preflight
./submit.sh smoke     # 1 node - prove the BF16 path before spending 8
./submit.sh control   # 8 nodes, 100 steps
```
