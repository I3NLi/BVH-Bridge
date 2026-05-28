from __future__ import annotations

from pathlib import Path

import numpy as np

from .bvh import (
    BvhMotion,
    parse_bvh,
    rotation_channels,
    rotation_to_channel_angles,
    write_bvh,
)
from .canonical import CanonicalMotion, to_canonical
from .formats import CANONICAL_ROOT_HEIGHT, UNIT_TO_METERS, FormatSpec, detect_format, get_spec


def load_canonical(path: str | Path, source_format: str = "auto", max_frames: int | None = None) -> CanonicalMotion:
    motion = parse_bvh(path)
    fmt = detect_format(motion) if source_format == "auto" else source_format
    return to_canonical(motion, get_spec(fmt), max_frames=max_frames)


def _template_scale(template: BvhMotion, spec: FormatSpec) -> float:
    canonical = to_canonical(template, spec, max_frames=min(template.frame_count, 300))
    # to_canonical already normalized to CANONICAL_ROOT_HEIGHT. Its source_scale
    # is therefore the factor that maps target template meters into canonical units.
    return canonical.source_scale


def canonical_to_template_frames(
    canonical: CanonicalMotion,
    template: BvhMotion,
    target_spec: FormatSpec,
) -> np.ndarray:
    raw_to_can = target_spec.raw_to_canonical_rot
    can_to_raw = raw_to_can.T
    template_scale = _template_scale(template, target_spec)
    semantic_for_raw = target_spec.raw_to_semantic
    frame_count = canonical.frame_count
    output = np.zeros((frame_count, template.frames.shape[1]), dtype=np.float64)

    identity = np.eye(3, dtype=np.float64)

    for frame_idx in range(frame_count):
        target_global_rot: dict[str, np.ndarray] = {}
        target_root_pos_raw = np.zeros(3, dtype=np.float64)
        if "Hips" in canonical.positions:
            target_root_pos_raw = can_to_raw @ (canonical.positions["Hips"][frame_idx] / template_scale / UNIT_TO_METERS)

        for name in template.hierarchy_order:
            node = template.nodes[name]
            semantic = semantic_for_raw.get(name)
            if semantic in canonical.rotations:
                target_global_rot[name] = can_to_raw @ canonical.rotations[semantic][frame_idx] @ raw_to_can
            elif node.parent is None:
                target_global_rot[name] = identity
            else:
                target_global_rot[name] = target_global_rot[node.parent]

        for name in template.channel_order:
            node = template.nodes[name]
            cursor = node.channel_start
            if node.parent is None:
                local_rot = target_global_rot[name]
            else:
                local_rot = target_global_rot[node.parent].T @ target_global_rot[name]

            rot_channels = rotation_channels(node.channels)
            rot_angles = rotation_to_channel_angles(local_rot, rot_channels)
            rot_idx = 0
            for offset, channel in enumerate(node.channels):
                if channel == "Xposition":
                    output[frame_idx, cursor + offset] = target_root_pos_raw[0]
                elif channel == "Yposition":
                    output[frame_idx, cursor + offset] = target_root_pos_raw[1]
                elif channel == "Zposition":
                    output[frame_idx, cursor + offset] = target_root_pos_raw[2]
                elif channel.endswith("rotation"):
                    output[frame_idx, cursor + offset] = rot_angles[rot_idx]
                    rot_idx += 1

    return output


def convert_bvh(
    input_path: str | Path,
    output_path: str | Path,
    target_format: str,
    source_format: str = "auto",
    template_path: str | Path | None = None,
    max_frames: int | None = None,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    source_motion = parse_bvh(input_path)
    detected = detect_format(source_motion) if source_format == "auto" else source_format
    source_spec = get_spec(detected)
    target_spec = get_spec(target_format)

    if max_frames is not None and source_motion.frame_count > max_frames:
        source_motion = source_motion.copy_with_frames(source_motion.frames[:max_frames].copy())

    canonical = to_canonical(source_motion, source_spec)
    template = parse_bvh(template_path or target_spec.default_template)
    frames = canonical_to_template_frames(canonical, template, target_spec)
    write_bvh(output_path, template, frames, source_motion.frame_time)
    return output_path


def format_summary(path: str | Path) -> dict[str, object]:
    motion = parse_bvh(path)
    fmt = detect_format(motion)
    spec = get_spec(fmt)
    canonical = to_canonical(motion, spec, max_frames=min(motion.frame_count, 300))
    return {
        "file": str(Path(path)),
        "format": fmt,
        "frames": motion.frame_count,
        "fps": 1.0 / motion.frame_time,
        "joints": len(motion.nodes),
        "canonical_joints": sorted(canonical.positions),
        "canonical_root_height": CANONICAL_ROOT_HEIGHT,
        "source_scale": canonical.source_scale,
    }
