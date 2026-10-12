# Architecture and paper correspondence

The repository has three layers. `third_party/Code2Worlds/` creates the visual
Blender world. `module1/` and `module2/` add physical compilation and simulated
dynamics. `benchmark/simcode4d_unified/` owns the frozen evaluation contracts,
robot probes, validators, runtime assets, and metric code. A method's world
package must stay separate from the evaluator-owned code.

| Paper component | Public implementation | What is executable here |
| --- | --- | --- |
| Dual-stream visual world generation | `third_party/Code2Worlds/agent/` | Upstream generation, with Infinigen and a model provider installed separately |
| M1 key-object selection and convex collision | `module1/module1_select_keys_v2.py`, `module1/module1_coacd.py` | Blender manifest to CoACD collision pieces |
| M1 terrain collision | `module1/module1_terrain_grid.py`, `module1/module1_terrain_hfield.py` | Local terrain cells or heightfield for outdoor scenes |
| M1 physical attributes | `module1/module1_vlm_render.py`, `module1/module1_vlm_attrs.py`, `module1/asset_registry.py` | Four-view estimates or local material fallback; mesh-derived mass and inertia |
| M1 support graph and gravity settling | `module1/support_graph.py`, `module1/settle_scene.py` | AABB relation proposal; independent MuJoCo settling and accepted-pose MJCF export |
| M1 contact, friction, paired exports | `module1/module1_emit_mjcf_v2.py` | Default and enriched MJCF with contact masks, friction and solver settings |
| M2 dynamics program and physical bodies | `module2/module2_dyn_program.py`, `module2/module2_emit_dynamic_v2.py` | Typed rigid/visual dynamics, mass/inertia, free bodies and robot model assembly |
| M2 articulated execution | `module2/module2_rollout_v2.py`, `module2/module2_rollout_jackal.py` | Go2, UR5e, and Jackal paths with differing scene requirements; NAT-03's amphibious task runs through the frozen benchmark runtime |
| M2 simulator and visual critic | `module2/module2_sim_critic.py` | Seven signals for the Go2/object interaction path; VLM verdict if configured |
| M2 bounded feedback | `module2/run_refinement.py` | Recompile, run, critique, and request a bounded physical-plan revision for the core Go2 path |
| SimCode4D case contracts | `benchmark/simcode4d_unified/cases/`, `config/`, `schemas/` | 15 frozen cases, five domains, three realized task families |
| SimCode4D reference validation | `benchmark/simcode4d_unified/tools/run_matrix.py`, `aggregate_results.py` | Independent robot probes, physical diagnostics, validators, per-predicate and seed aggregation |
| SimCode4D method test | `benchmark/simcode4d_unified/tools/score_submission.py`, `method_metrics.py` | First-output build/overlay gates, binding and annotated support F1, episode RTS |

## Boundaries of the public package

The 15 frozen compiled worlds are runnable without the original Blender
authoring scenes. The [`demo_scene/`](demo_scene/) directory supplies a small
editable Blender input and dynamics program; the bundled benchmark Go2 model
supports its offline M1/M2 physics run. A different scene needs its own
Blender input and compatible robot model, with an optional model API for
automated authoring and visual critique. The support graph is a geometric proposal; it must be checked
against simulator contacts. The standalone settling tool writes accepted
free-body poses and does not change the frozen benchmark worlds. The M2
automatic loop currently covers the Go2 interaction path; the specialized
robot paths have their own executors and are not claimed to use one generic
repair policy.

Method scoring uses the frozen runtime interface and checks runtime entity
bindings. The manuscript's *source-name* binding F1 requires the original
authoring-source catalogs for the 12 fixed-source cases. Those large source
bundles are absent here, so the public scorer labels its runtime binding F1
separately. Initial support F1 covers the five cases with published task-object
support labels; unannotated background relations are excluded. The metric
functions in `method_metrics.py` can pool source-name counts if separately
provided. A DEV15 reference-package success rate is not a generated-method
success rate.

No ablation runners, experiment configurations, completed result logs, or
demo videos are part of this release. Frozen provenance manifests
still mention the original authoring locations; those paths are not runtime
dependencies.
