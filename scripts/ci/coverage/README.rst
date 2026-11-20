Ascend PR workload coverage
===========================

``.github/workflows/coverage-collection-npu.yml`` calls the original Ascend
workflows with ``coverage_mode: true``. Those reusable workflows own their
jobs, environments and commands. There is no generated workflow, duplicated
job inventory or regeneration command. Editing a workload changes both normal
execution and coverage execution.

The collector runs at 05:00 UTC+8 on alternate days of the month
(``0 21 */2 * *``), rather than a strict 48-hour interval. Manual dispatch and
a temporary PR trigger are also available. No precision-test mapping, OBS
publication or PR test selection is performed.

Normal CI operation
-------------------

Ordinary PR events leave ``coverage_mode`` false, so coverage dependencies,
setup and packaging are skipped and workload steps use the original default
container shell (sh). Temporary trigger-disabling changes are maintained
separately from the coverage implementation.

Scope and maintenance
---------------------

The caller lists 14 Ascend PR workload workflows containing 19 declared jobs.
All four SGLang workflow calls are temporarily disabled in the collector with
``if: ${{ false }}`` while SGLang issues are resolved. Restore the commented
repository-owner condition on those calls to resume collection. The two
originally disabled Megatron/SGLang jobs also remain disabled. Nightly,
multi-node, image-build and documentation workloads are excluded. CPU test
steps inside an Ascend job are included.

Each workload job has direct setup and packaging ``run`` steps, plus an
``actions/upload-artifact@v4`` step. Setup runs after dependency installation;
packaging and upload use ``always()``. These steps own coverage installation,
output directories and artifact retention. Avoid local composite actions here:
the Kubernetes runner may not see files checked out inside the job container
when it resolves a local action manifest. Direct commands execute in that
container, where the repository is available.
Coverage-mode concurrency groups are unique to each reusable workflow and
separate from normal PR runs and the caller; timeouts are larger in coverage
mode. Original job dependencies, conditions and step ordering are preserved.

Workload steps use a shared shell wrapper::

    shell: python scripts/ci/coverage/collect.py run step-06 {0}

Each reusable workflow sets ``VERL_COVERAGE_ENABLED`` from
``inputs.coverage_mode``. In normal mode the wrapper immediately executes
``sh -e`` without importing coverage or PyYAML or stopping Ray. Expressions
are kept out of the step-level ``shell`` field.

When adding a workload step, use this shell with a unique step ID in that job.
The command itself stays in the step's single ``run`` block. Data preparation,
setup and cleanup steps must not use the coverage shell. Existing commands, environments and
parameters can be edited normally without changing the collector. A new job
needs the optional setup/package/upload steps and coverage shells; a completely
new workflow also needs one ``uses`` entry in the caller.

At setup time, ``collect.py`` reads the current job's coverage-marked steps
directly from its workflow YAML to construct the manifest. Names, entrypoints
and expected steps are not maintained in a separate JSON inventory. Missing
or duplicate step IDs fail initialization.

Collection behavior
-------------------

Setup installs coverage.py 7.8.2 and PyYAML 6.0.2 into the ambient Python
interpreter with uv. A dedicated opt-in ``.pth`` hook preserves the image's
existing ``sitecustomize`` and ``PYTHONPATH``. If a workload introduces another
Python environment, coverage and the hook must be installed there too.

Each coverage-marked Actions ``run`` block is one collection entry.
Multiple commands in a block share coverage,
preserving heredocs, shell state, checkpoint/resume ordering and validation
commands. Different steps invoking the same script remain separate entries.

The wrapper sets coverage identity before executing the entire block, including
any ``ray start`` command. It stops the container-local Ray cluster gracefully
before and after a block so subsequent steps do not reuse workers with stale
coverage identity. Original commands and their exit status are preserved;
cleanup failures also fail the collection step. Coverage uses branch data,
parallel process files, multiprocessing support and SIGTERM handling. It does
not instrument data preparation, installation or packaging commands. Data
preparation runs with the normal container shell and is excluded from the
coverage manifest and artifacts.

Original failure and dependency behavior is retained. Following a failure,
later dependent work may be skipped; its manifest entries remain explicitly
incomplete. Packaging and artifact retention run with ``always()``. Hard
cancellation or runner loss can still prevent archiving.

Artifacts
---------

Each executed job retains a GitHub Actions artifact for 14 days, named
``ascend-coverage-<workflow.yml>-<job-id>-<run-attempt>``. The tar contains::

    manifest.json
    packages.txt
    source-workflow.yml
    coveragerc
    coverage/step-NN/coverage.<host>.<pid>.<suffix>

The source SHA, run/attempt, original workflow/job, step identities, aggregate process
counts and per-step status allow later adapters to associate data correctly.
Process startup JSON files are used for validation but excluded from the tar.
Per-step ``command.sh`` copies are not created or included in archives; step
names and entrypoints remain in the manifest. Aggregation also excludes
``command.sh`` files from older job artifacts.
Valid coverage databases with no executed lines or arcs are also excluded;
``empty_coverage_files_removed`` records their count per step. The retained
SQLite files are unchanged; no coverage combine operation consumes them.
Corrupt databases remain available for diagnosis and still fail completeness.
Removing preprocessing instrumentation does not eliminate empty databases from
training subprocesses, so this filtering also runs during packaging.
Different workflows, jobs and run attempts have separate workspace directories.

``collection_complete`` requires all planned steps to pass, valid branch data
for test/training steps, and data files for the observed interpreter startups.
Missing data, hook errors, unrun steps and corrupt files cause finalization to
fail, but the partial tar is still uploaded. ``worker_coverage_verified`` and ``selection_ready`` remain
false: process counts cannot prove every Ray actor or engine subprocess was
instrumented. Forked processes do not necessarily produce a separate startup
record.

Combined archive
----------------

The final ``aggregate_coverage`` job waits for every workload workflow,
including failed or skipped calls, and downloads available job artifacts. It
publishes ``ascend-coverage-combined-<run-attempt>`` containing one
``coverage.tar.gz`` with this layout::

    index.json
    workflows/<workflow.yml>/<job-id>/manifest.json
    workflows/<workflow.yml>/<job-id>/coverage/step-NN/coverage.<host>.<pid>.<suffix>
    covstub/verl/...

The combined archive keeps one shared source snapshot at ``covstub/verl/``,
beside ``workflows/``. The top-level ``covstub`` field in ``index.json`` points
to it. Aggregation creates this snapshot once using ``git archive`` from the
run's exact source SHA in its checkout. Each selected job's manifest must match
that SHA. Individual job artifacts do not include source snapshots; redundant
snapshots from older attempts are ignored. The snapshot contains committed source,
not any uncommitted changes made by workloads. All retained job files live under
their workflow/job directory.
``index.json`` maps every called workflow filename to its caller result and
available jobs, then each job to its manifest and step directories. Failed or
skipped workflows with no artifacts remain in the index with
``artifacts_present: false``. Partial job manifests retain their original
completeness status; successfully bundling them does not certify coverage.

On a partial rerun, the latest available artifact for each job is used, including
previous attempts of this same run when a job was not rerun. Each selected
attempt is recorded. Run IDs and source SHAs must match; duplicate artifacts
for the same workflow/job/attempt are rejected. No available archives is an
error. The individual artifacts and combined artifact are retained for 14 days.
When adding a caller workflow, also add it to the aggregate job's ``needs`` list.

First CI inspection
-------------------

Inspect a pytest/torchrun job, an FSDP/vLLM job and a runnable SGLang job before
treating collection as reliable. Confirm rank/worker-side verl modules appear
in the raw data, paths refer to the intended checkout, and later entries do
not contain a previous entry's Ray workers. SIGKILL, abrupt engine shutdown
and overwritten child environments can lose coverage even when training passes.
No local PyTorch execution is needed for lint or packaging checks.
