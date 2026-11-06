#!/bin/bash
# Print the shell exports that let RCCL use the host's AMD ionic (Pensando Pollara / AINIC) RDMA NICs.
#
# The image carries two ibverbs userspaces:
#   - the base image's rdma-core 39 + libionic 54 (ionic provider ABI 4: ionic drivers before 26.x), the default;
#   - upstream rdma-core in /opt/rdma-core (ionic provider ABI 1: ionic 26.x drivers, e.g. 26.08).
# A provider whose ABI does not match the host driver cannot open the device, and RCCL then silently falls back to
# TCP sockets. This script opens the first ionic device with each userspace and selects the one that works. With
# /opt/rdma-core it also enables dma-buf registration of GPU memory (NCCL_DMABUF_ENABLE=1, unless already set),
# which hosts without an amdgpu peer-memory module need for GPU Direct RDMA.
# Prints nothing when there is no ionic device or neither userspace can open it.
#
# Usage, on every node before `ray start` (the recipe also runs it): eval "$(ionic-rdma-env.sh)"
dev=$(ls /sys/class/infiniband 2>/dev/null | grep -m1 '^ionic') || exit 0
# probe <libdir>: can the libibverbs in <libdir> (and the provider it loads) open $dev? Runs in a fresh process.
probe() {
  LD_LIBRARY_PATH="$1" python3 - "$1/libibverbs.so.1" "$dev" >/dev/null 2>&1 <<'PY'
import ctypes
import sys

v = ctypes.CDLL(sys.argv[1])
v.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
v.ibv_get_device_name.restype = ctypes.c_char_p
v.ibv_open_device.restype = ctypes.c_void_p
n = ctypes.c_int(0)
devs = v.ibv_get_device_list(ctypes.byref(n))
for i in range(n.value):
    if v.ibv_get_device_name(ctypes.c_void_p(devs[i])).decode() == sys.argv[2]:
        sys.exit(0 if v.ibv_open_device(ctypes.c_void_p(devs[i])) else 2)
sys.exit(3)
PY
}
if probe /opt/rdma-core/lib; then
  echo "export LD_LIBRARY_PATH=/opt/rdma-core/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
  echo "export NCCL_DMABUF_ENABLE=\${NCCL_DMABUF_ENABLE:-1}"
  echo "ionic-rdma-env: $dev opens with /opt/rdma-core (ionic ABI 1)" >&2
elif probe /usr/lib/x86_64-linux-gnu; then
  echo "ionic-rdma-env: $dev opens with the base image's rdma-core (ionic ABI 4)" >&2
else
  echo "ionic-rdma-env: neither userspace can open $dev; RCCL will use TCP sockets" >&2
fi
