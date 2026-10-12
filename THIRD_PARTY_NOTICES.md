# Third-party source and assets

The included Code2Worlds source is a snapshot of [`AIGeeksGroup/Code2Worlds`](https://github.com/AIGeeksGroup/Code2Worlds) at commit `03d7edb9242b4095bfb9b3315b1fd67a20d433a1`, with the local integration changes documented in [`third_party/Code2Worlds/UPSTREAM.md`](third_party/Code2Worlds/UPSTREAM.md). The inspected upstream snapshot contains no top-level license file; no license is invented here.

The benchmark retains only the robot meshes needed for its frozen runnable models. License notices are preserved at:

- [`benchmark/simcode4d_unified/vendor/handover9_20260921/unitree_go2/LICENSE`](benchmark/simcode4d_unified/vendor/handover9_20260921/unitree_go2/LICENSE)
- [`benchmark/simcode4d_unified/vendor/handover9_20260921/universal_robots_ur5e/LICENSE`](benchmark/simcode4d_unified/vendor/handover9_20260921/universal_robots_ur5e/LICENSE)
- [`benchmark/simcode4d_unified/vendor/jackal_ugv/LICENSE.clearpath`](benchmark/simcode4d_unified/vendor/jackal_ugv/LICENSE.clearpath)

Infinigen is not vendored in this release. Install it separately from [princeton-vl/infinigen](https://github.com/princeton-vl/infinigen) and follow its license.
