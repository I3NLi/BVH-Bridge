# BVH Bridge

`bvh_bridge` is a small standalone project for converting between the three BVH
families currently used in the motion pipeline:

- `lafan1`
- `nokov`
- `wbody`

The conversion path is always:

```text
source BVH -> canonical semantic motion -> target BVH template
```

This keeps format conversion separate from robot retargeting. The canonical
motion uses a shared axis convention and static rest-pose scale, so root
trajectory and joint rotations are not affected by frame-dependent poses such as
kicks or crouches.

Target BVH files keep the target template hierarchy and bone lengths. That means
pairwise position metrics include normal skeleton-shape differences. The primary
correctness check is closed-loop validation (`A -> B -> A`), which compares only
the joints expressible by every format in that path and records non-recoverable
intermediate joints in `lost_joints`.

## CLI

From `/home/hiyio/engineai`:

```bash
PYTHONPATH=tools/bvh_bridge python -m bvh_bridge.cli inspect input.bvh

PYTHONPATH=tools/bvh_bridge python -m bvh_bridge.cli convert \
  --input input.bvh \
  --target nokov \
  --output output_nokov.bvh

PYTHONPATH=tools/bvh_bridge python -m bvh_bridge.cli validate \
  --lafan /home/hiyio/GMR_hxl/Motion_data/lafan1/fight1_subject2.bvh \
  --nokov /home/hiyio/engineai/data/official/boxing_motions/raw_bvh/540huixuantitui_001.bvh \
  --wbody /home/hiyio/engineai/data/raw/xiayao2026-Wbody0.bvh \
  --out-dir /tmp/bvh_bridge_validate \
  --max-frames 300
```

Validation writes all pairwise conversions and A->B->A closed-loop files, then
prints canonical root-relative position, root trajectory, and rotation errors.
