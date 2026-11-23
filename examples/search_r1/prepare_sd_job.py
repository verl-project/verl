"""Prepare (never submit) an active two-step SD acceptance + fresh 200-step run."""

import datetime
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path(__file__).resolve().parents[3]
baseline = root / "outputs/search-r1-200step-20261122-203135-skill-split"
stamp = datetime.datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
common = [
    "python3",
    str(root / "scripts/prepare_five_step_job.py"),
    "--gpus",
    "8",
    "--actor-micro-batch",
    "32",
    "--old-log-prob-micro-batch",
    "32",
    "--ref-log-prob-micro-batch",
    "32",
    "--model-path",
    str(root / "models/Qwen2.5-3B-Instruct"),
    "--retrieval-encoder-device",
    "cpu",
    "--charged-group",
    "llmagent_gpu",
    "--telemetry",
    "--cuda-graphs",
    "--max-num-batched-tokens",
    "32768",
    "--comparison-baseline",
    str(baseline),
]
runs = {}
for kind, steps, freq in [("acceptance", 2, -1), ("formal", 200, 20)]:
    result = json.loads(
        subprocess.check_output(
            common + ["--steps", str(steps), "--save-frequency", str(freq), "--run-suffix", "sd-" + kind], text=True
        )
    )
    run = Path(result["run_dir"])
    shutil.copy2(root / "verl/examples/search_r1/collect_sd_telemetry.py", run / "collect_rollout_telemetry.py")
    runs[kind] = str(run)
    script = (
        (run / "run.sh")
        .read_text()
        .replace("bash examples/search_r1/run_grpo.sh", "bash examples/search_r1/run_sd_grpo.sh")
    )
    if kind == "acceptance":
        script = script.rstrip() + " sd_search.warmup_steps=0\n"
    (run / "run.sh").write_text(script)
    collect = (run / "collect_metrics.py").read_text().replace("+'\\\\n'", "+'\\n'")
    (run / "collect_metrics.py").write_text(collect)
    m = json.loads((run / "manifest.json").read_text())
    m.update(
        status="prepared-not-submitted",
        time_limit="awaiting current submission runtime choice",
        algorithm="SD-Search arXiv:2605.18299v1",
        task_reward="EM, retained for comparison; paper uses F1",
        paper_parameters={"alpha": 0.001, "warmup_steps": 0 if kind == "acceptance" else 50, "top_k": 50, "rho": 0},
        comparison_baseline=str(baseline),
    )
    m["recipe_sha256"] = {
        str(p.relative_to(run / "source")): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (run / "source/examples/search_r1").rglob("*")
        if p.is_file()
    }
    m["frozen_script_sha256"] = hashlib.sha256((run / "run.sh").read_bytes()).hexdigest()
    (run / "manifest.json").write_text(json.dumps(m, indent=2))
    subprocess.run(["bash", "-n", str(run / "run.sh")], check=True)
pipeline = root / "outputs" / ("search-r1-sd-pipeline-" + stamp)
pipeline.mkdir()
(pipeline / "run.sh").write_text(
    """#!/usr/bin/env bash
set -euo pipefail
PIPELINE_DIR="""
    + shlex.quote(str(pipeline))
    + """
exec > "$PIPELINE_DIR/console.log" 2>&1
trap 'code=$?; echo "$code" > "$PIPELINE_DIR/exit-code.txt"' EXIT
bash """
    + shlex.quote(runs["acceptance"] + "/run.sh")
    + """
/usr/bin/python3 - """
    + shlex.quote(runs["acceptance"])
    + """ <<'VERIFY'
import json,math,sys
from pathlib import Path
r=Path(sys.argv[1]);report=json.loads((r/'metrics.json').read_text())
assert report['verification_passed'],report
rows=[x.get('data',x) for x in report['records']]
values=[x['actor/sd/jsd'] for x in rows if 'actor/sd/jsd' in x]
assert len(values)==2 and all(math.isfinite(x) and x>0 for x in values),values
assert all(x['sd/alpha']==.001 for x in rows if 'sd/alpha' in x)
(r/'sd-acceptance.json').write_text(json.dumps(
    {'passed':True,'jsd':values,
     'scope':'active SD test; formal run starts fresh with 50-step warmup'},indent=2))
VERIFY
bash """
    + shlex.quote(runs["formal"] + "/run.sh")
    + "\n"
)
cmd = json.loads((Path(runs["formal"]) / "manifest.json").read_text())["command"]
cmd[cmd.index("--name") + 1] = pipeline.name
cmd[-1] = str(pipeline / "run.sh")
cmd[cmd.index("--mount") : cmd.index("--mount")] = [
    "--negative-tags",
    "node/gpu-lg-cmc-h-h200-0821.host.h.pjlab.org.cn",
]
manifest = {
    "status": "prepared-not-submitted",
    "runtime_authorization": "pending",
    "resources": {"gpu": 8, "cpu": 80, "memory_mib": 901120, "charged_group": "llmagent_gpu"},
    "command": cmd,
    "runs": runs,
    "baseline": str(baseline),
    "source_commit": subprocess.check_output(["git", "-C", str(root / "verl"), "rev-parse", "HEAD"], text=True).strip(),
}
(pipeline / "manifest.json").write_text(json.dumps(manifest, indent=2))
(pipeline / "submit-command.sh").write_text(shlex.join(cmd) + "\n")
subprocess.run(["bash", "-n", str(pipeline / "run.sh")], check=True)
pointer = root / "artifacts/search-r1-validation/latest-sd-search.json"
pointer.write_text(json.dumps({"pipeline_dir": str(pipeline), **runs}, indent=2))
print(json.dumps({"pipeline_dir": str(pipeline), "runs": runs, "command": cmd}, indent=2))
