import subprocess, re, sys, json, datetime, argparse
parser=argparse.ArgumentParser()
parser.add_argument("--devices",default=None)
args=parser.parse_args()
selected=None if args.devices is None else {int(x) for x in args.devices.split(",")}
assert selected is None or (selected and selected.issubset(set(range(8))))
from pathlib import Path
result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True)
rows = re.findall(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|", result.stdout)
pids = [pid for device,pid in rows if selected is None or int(device) in selected]
# The observed Qwen accuracy launcher restarts its full-node server. An empty
# NPU table between restarts does not mean the machine has been released.
launchers = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        args = (entry / "cmdline").read_bytes().decode(errors="replace").strip("\0").split("\0")
        observed_launcher = "/workspace/qwen38_accuracy/run_sglang_acc.sh" in args
        server = any(args[i:i+2] == ["-m", "sglang.launch_server"] for i in range(len(args)-1))
        if observed_launcher or server:
            launchers.append({"pid": int(entry.name), "argv": args})
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
print(json.dumps({"time": datetime.datetime.now().astimezone().isoformat(), "selected_devices": sorted(selected) if selected is not None else "all", "observed_processes": rows, "active_npu_pids": pids, "other_workload_launchers": launchers}), flush=True)
if pids or launchers:
    sys.exit("Other NPU workload or launcher found; refusing to start another workload")
