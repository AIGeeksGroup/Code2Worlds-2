# SimCode4D DEV15 public runtime

This directory is the frozen 15-case development suite described in Sec. 4 of the Code2Worlds-2 manuscript. It includes three cases in each of five domains: indoor, industrial, natural outdoor, service, and urban. The implemented families are seven navigation, six dynamic-interaction, and two pick/transport cases.

## Contents

| Path | Purpose |
| --- | --- |
| `config/suite.json` and `config/freeze_lock.json` | Registered case grid and frozen hashes |
| `cases/` | Case specification, gold predicates, task confirmation, physics profile |
| `artifacts/` | Compiled M1 worlds, executable runtime MJCF, task overlays |
| `probes/` and `validators/` | Evaluator-owned robot policy and case-specific predicates |
| `tools/` | Runtime, isolated episode worker, matrix runner, method scorer, physical diagnostics, result aggregation |
| `vendor/` and `assets/nat01_meshes/` | Exact model meshes needed by the frozen runtime |

The 12 archived Blender authoring scenes are omitted from this GitHub code release. Three indoor authoring scenes were not archived in the original benchmark. All 15 compiled M1 worlds and task runtimes are included. Three cases required [path-only portability repairs](PORTABILITY.md), so their published hashes differ from the original server archive. The original `tools/validate_suite.py` checks source archives in addition to executable contracts, so it cannot pass on this compact release. Run the public integrity check below, then use `--skip-preflight` with the unchanged matrix runner. This still checks hashes for the registered toolchain, selected cases, models, and overlays during execution.

## Verify and run

From the repository root, with Python 3.10 and `requirements-benchmark.txt` installed:

```bash
python benchmark/simcode4d_unified/check_public_release.py

# Small smoke run.
python benchmark/simcode4d_unified/tools/run_matrix.py \
  --suite benchmark/simcode4d_unified \
  --output /tmp/simcode4d-smoke \
  --case-ids SER-03-hotel-drawer \
  --generation-seeds 1103 \
  --physics-seeds 401 \
  --skip-preflight

python benchmark/simcode4d_unified/tools/aggregate_results.py \
  --suite benchmark/simcode4d_unified \
  --run-root /tmp/simcode4d-smoke \
  --output /tmp/simcode4d-smoke-metrics \
  --allow-subset
```

For the full reference matrix, omit all case and seed selectors. Use fresh output paths:

```bash
python benchmark/simcode4d_unified/tools/run_matrix.py \
  --suite benchmark/simcode4d_unified \
  --output /tmp/simcode4d-dev15-matrix \
  --skip-preflight

python benchmark/simcode4d_unified/tools/aggregate_results.py \
  --suite benchmark/simcode4d_unified \
  --run-root /tmp/simcode4d-dev15-matrix \
  --output /tmp/simcode4d-dev15-metrics
```

The full run is 15 cases × 3 deterministic generation/rebuild slots × 5 physics seeds = 225 reference executions. The metrics in the paper establish reference-package self-consistency. They are not a score for a generated method. Keep a method's submitted world and task overlay separate from the evaluator-owned probes and validators.

The matrix runner starts a fresh worker process for each episode. This releases
MuJoCo allocations between cases and keeps the registered full matrix within
modest memory limits. If an external host terminates an episode, the run
manifest records the error and the incomplete matrix cannot be aggregated as
a passing result.

## Test a method world package

The public method scorer accepts one first-output package per selected case:

```text
SUBMISSION_ROOT/
  SER-03-hotel-drawer/
    runtime.xml
    task_overlay.json
    bindings.json
    assets/                 # if runtime.xml references local files
```

`runtime.xml` must carry the required bodies, joints, actuators, contact
groups, local assets, physical properties, and frozen 1 ms simulator setup.
`task_overlay.json` uses `code2worlds2.task-dynamics.v1` with `case_id`, the
SHA-256 of `runtime.xml` in `runtime_sha256`, all required dynamic objects in
`dynamic_bodies`, and typed `events` (`rigid_body`, `joint`, `visual_effect`).
`bindings.json` uses `code2worlds2.bindings.v1`, `case_id`, a `bindings` list
of `{entity_id, model_names}`, and an optional `support_relations` list of
`{child, parent}` using registered entity IDs. XML assets must be relative to
the package and remain inside it.

Build and overlay checks precede the frozen probe. Failed builds, overlays,
and executions remain zero in the registered episode denominator. For every
completed episode the same independent validators report task, mechanics,
kinematics, contact/friction, safety, integrity, and robust task success.
The scorer also isolates each case and episode in a fresh process to avoid
retaining large compiled worlds in memory.
`method_metrics.py` computes WorldBuildPass@1, TaskOverlayPass@1, pooled F1,
and RTS. Runtime entity-binding F1 checks the compiled model interface; the
paper's source-name binding F1 additionally needs the 12 authoring-source
catalogs, which are outside this compact release. Initial support F1 uses
only five annotated task-object relations; unannotated relations are not
treated as negatives.

To verify the submission API itself without preparing a new scene, create a
self-contained package from the included SER-03 reference world:

```bash
python benchmark/simcode4d_unified/tools/make_reference_smoke.py \
  --output /tmp/simcode4d-reference-package
python benchmark/simcode4d_unified/tools/score_submission.py \
  --submission-root /tmp/simcode4d-reference-package \
  --output /tmp/simcode4d-reference-score \
  --case-ids SER-03-hotel-drawer --physics-seeds 401
```

The resulting `method_score.json` is a reference-interface smoke test, not a
new generated-method result. Omit case and seed selectors to evaluate all 15
registered cases and five physics seeds of a complete submission.

This code release does not include ablation configurations, completed run logs, leaderboards, or demo videos. The case specifications for IND-02, NAT-01, and NAT-02 were rebound to relative asset paths; their physical parameters, probes, and validators were not changed. Provenance paths in a few source manifests refer to their historical authoring machines and are not used for runtime loading.
