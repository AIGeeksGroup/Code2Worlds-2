# Public runtime portability

The DEV15 runtime package was built on an internal server. Three frozen cases contained XML paths that resolved to that server, so a fresh GitHub clone could not load every model. This code release changes **asset paths only** for:

| Case | Rebound paths |
| --- | --- |
| `IND-02-go2-living-room` | Go2 mesh paths in `artifacts/IND-02-go2-living-room/go2_robot.xml` |
| `NAT-01-rock-kick` | Terrain/collision assets in `m1_world.xml` and Go2 meshes in `go2_runtime.xml` |
| `NAT-02-hills-turn-stop` | Jackal meshes in `jackal_runtime.xml` and its include path in `m1_world.xml` |

The Go2 and UR5e meshes already used by the suite remain in `vendor/handover9_20260921/`, with their license notices. The Jackal meshes required to load NAT-02 are included at `vendor/jackal_ugv/` with the Clearpath notice. The indoor Go2 historical source runtime was also made relative and points to the same shared Go2 meshes.

NAT-01's mesh export is stored under `assets/nat01_meshes/`. It was previously published as a top-level `export_scene1/` directory, but it is a benchmark runtime asset set, not a standalone editable Blender scene. Its historical manifest contains references to authoring files that are absent from the compact release. The runtime XML and dependency manifest now point to the relocated meshes.

The affected dependency manifests, M1 overlay hashes, case specifications, gold bindings, physics profiles, suite registry, and freeze lock were recalculated for the public tree. This **changes the file SHA-256 values** from the paper's internal archive without changing the registered physics parameters, task predicates, probes, or validators. The exact internal archive can be identified from the source handover; this repository is the portable publication package.

Twelve Blender authoring scenes are intentionally absent. Their historical source pointers remain in the case specifications and manifests; the executable MJCF and local runtime assets for all 15 cases are present. Use `check_public_release.py` to validate this publication package before running the matrix.
