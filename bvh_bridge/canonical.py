from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bvh import BvhMotion, forward_kinematics
from .formats import CANONICAL_ROOT_HEIGHT, FormatSpec, UNIT_TO_METERS


@dataclass
class CanonicalMotion:
    positions: dict[str, np.ndarray]
    rotations: dict[str, np.ndarray]
    frame_time: float
    source_format: str
    source_scale: float

    @property
    def frame_count(self) -> int:
        first = next(iter(self.positions.values()))
        return int(first.shape[0])


def _available_semantics(motion: BvhMotion, spec: FormatSpec) -> dict[str, str]:
    return {
        semantic: raw
        for semantic, raw in spec.semantic_to_raw.items()
        if raw in motion.nodes
    }


def _rest_positions(motion: BvhMotion) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}

    def visit(name: str) -> None:
        node = motion.nodes[name]
        if node.parent is None:
            positions[name] = np.zeros(3, dtype=np.float64)
        else:
            positions[name] = positions[node.parent] + node.offset
        for child in node.children:
            visit(child)

    visit(motion.root)
    return positions


def _rest_root_height(motion: BvhMotion, spec: FormatSpec) -> float:
    rest_raw = _rest_positions(motion)
    mapping = _available_semantics(motion, spec)
    raw_to_canonical = spec.raw_to_canonical_rot

    rest = {
        semantic: (rest_raw[raw_name] @ raw_to_canonical.T) * UNIT_TO_METERS
        for semantic, raw_name in mapping.items()
        if raw_name in rest_raw
    }
    if "Hips" not in rest:
        raise ValueError("Cannot estimate rest scale without Hips")

    foot_names = [name for name in ("LeftFoot", "RightFoot") if name in rest]
    if not foot_names:
        foot_names = [name for name in ("LeftToeBase", "RightToeBase") if name in rest]
    if not foot_names:
        raise ValueError("Cannot estimate rest scale without foot joints")

    hips_z = float(rest["Hips"][2])
    distances = [abs(hips_z - float(rest[name][2])) for name in foot_names]
    return max(float(np.mean(distances)), 1e-6)


def to_canonical(motion: BvhMotion, spec: FormatSpec, max_frames: int | None = None) -> CanonicalMotion:
    if max_frames is not None and motion.frame_count > max_frames:
        motion = motion.copy_with_frames(motion.frames[:max_frames].copy())

    world_pos, world_rot = forward_kinematics(motion)
    mapping = _available_semantics(motion, spec)
    raw_to_canonical = spec.raw_to_canonical_rot

    positions_m: dict[str, np.ndarray] = {}
    rotations: dict[str, np.ndarray] = {}
    for semantic, raw_name in mapping.items():
        raw_pos = world_pos[raw_name]
        raw_rot = world_rot[raw_name]
        positions_m[semantic] = (raw_pos @ raw_to_canonical.T) * UNIT_TO_METERS
        rotations[semantic] = np.einsum("ij,fjk,lk->fil", raw_to_canonical, raw_rot, raw_to_canonical)

    required = {"Hips", "LeftFoot", "RightFoot"}
    missing = sorted(required - set(positions_m))
    if missing:
        raise ValueError(f"Missing required canonical joints: {missing}")

    scale = CANONICAL_ROOT_HEIGHT / _rest_root_height(motion, spec)
    positions = {name: value * scale for name, value in positions_m.items()}
    return CanonicalMotion(positions, rotations, motion.frame_time, spec.name, scale)


def common_semantics(a: CanonicalMotion, b: CanonicalMotion) -> list[str]:
    names = sorted(set(a.positions) & set(b.positions))
    return [name for name in names if name in a.rotations and name in b.rotations]


def root_relative_positions(motion: CanonicalMotion, names: list[str]) -> np.ndarray:
    root = motion.positions["Hips"]
    return np.stack([motion.positions[name] - root for name in names], axis=1)


def rotation_geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rel = np.einsum("fji,fjk->fik", a, b)
    trace = np.trace(rel, axis1=1, axis2=2)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cos)


def compare_canonical(
    reference: CanonicalMotion,
    candidate: CanonicalMotion,
    names: list[str] | None = None,
) -> dict[str, float]:
    frame_count = min(reference.frame_count, candidate.frame_count)
    names = common_semantics(reference, candidate) if names is None else sorted(names)
    names = [name for name in names if name in reference.positions and name in candidate.positions]
    if not names:
        raise ValueError("No common semantic joints to compare")

    ref = CanonicalMotion(
        {k: v[:frame_count] for k, v in reference.positions.items()},
        {k: v[:frame_count] for k, v in reference.rotations.items()},
        reference.frame_time,
        reference.source_format,
        reference.source_scale,
    )
    cand = CanonicalMotion(
        {k: v[:frame_count] for k, v in candidate.positions.items()},
        {k: v[:frame_count] for k, v in candidate.rotations.items()},
        candidate.frame_time,
        candidate.source_format,
        candidate.source_scale,
    )

    pos_delta = root_relative_positions(ref, names) - root_relative_positions(cand, names)
    pos_error = np.linalg.norm(pos_delta, axis=-1)

    rot_errors = []
    for name in names:
        rot_errors.append(rotation_geodesic(ref.rotations[name], cand.rotations[name]))
    rot_error = np.stack(rot_errors, axis=1)

    root_delta = ref.positions["Hips"] - cand.positions["Hips"]
    root_error = np.linalg.norm(root_delta, axis=-1)

    return {
        "frames": float(frame_count),
        "joints": float(len(names)),
        "pos_mean": float(np.mean(pos_error)),
        "pos_p95": float(np.percentile(pos_error, 95)),
        "pos_max": float(np.max(pos_error)),
        "root_mean": float(np.mean(root_error)),
        "root_p95": float(np.percentile(root_error, 95)),
        "rot_mean_deg": float(np.degrees(np.mean(rot_error))),
        "rot_p95_deg": float(np.degrees(np.percentile(rot_error, 95))),
        "rot_max_deg": float(np.degrees(np.max(rot_error))),
    }
