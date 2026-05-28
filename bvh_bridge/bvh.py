from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class BvhNode:
    name: str
    parent: Optional[str]
    offset: np.ndarray
    channels: list[str] = field(default_factory=list)
    channel_start: int = 0
    children: list[str] = field(default_factory=list)
    end_sites: list[np.ndarray] = field(default_factory=list)


@dataclass
class BvhMotion:
    nodes: dict[str, BvhNode]
    root: str
    hierarchy_order: list[str]
    channel_order: list[str]
    frames: np.ndarray
    frame_time: float

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    def copy_with_frames(self, frames: np.ndarray) -> "BvhMotion":
        return BvhMotion(
            nodes=self.nodes,
            root=self.root,
            hierarchy_order=self.hierarchy_order,
            channel_order=self.channel_order,
            frames=frames,
            frame_time=self.frame_time,
        )


def parse_bvh(path: str | Path) -> BvhMotion:
    lines = Path(path).read_text(errors="ignore").splitlines()
    nodes: dict[str, BvhNode] = {}
    hierarchy_order: list[str] = []
    channel_order: list[str] = []
    channel_cursor = 0
    idx = 0

    def parse_joint(parent: Optional[str]) -> str:
        nonlocal idx, channel_cursor
        parts = lines[idx].strip().split()
        if parts[0] not in {"ROOT", "JOINT"}:
            raise ValueError(f"Expected ROOT/JOINT at line {idx + 1}, got: {lines[idx]}")
        name = parts[1]
        hierarchy_order.append(name)
        idx += 1
        if idx >= len(lines) or lines[idx].strip() != "{":
            raise ValueError(f"Expected '{{' after {name} at line {idx + 1}")
        idx += 1

        offset = np.zeros(3, dtype=np.float64)
        channels: list[str] = []
        channel_start = 0
        children: list[str] = []
        end_sites: list[np.ndarray] = []

        while idx < len(lines):
            stripped = lines[idx].strip()
            parts = stripped.split()
            if not parts:
                idx += 1
                continue
            token = parts[0]
            if token == "OFFSET":
                offset = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                idx += 1
            elif token == "CHANNELS":
                count = int(parts[1])
                channels = parts[2 : 2 + count]
                channel_start = channel_cursor
                channel_cursor += count
                channel_order.append(name)
                idx += 1
            elif token == "JOINT":
                child = parse_joint(name)
                children.append(child)
            elif token == "End":
                idx += 1
                if idx >= len(lines) or lines[idx].strip() != "{":
                    raise ValueError(f"Malformed End Site near line {idx + 1}")
                idx += 1
                end_offset = np.zeros(3, dtype=np.float64)
                while idx < len(lines):
                    end_line = lines[idx].strip()
                    end_parts = end_line.split()
                    if end_parts and end_parts[0] == "OFFSET":
                        end_offset = np.asarray(
                            [float(end_parts[1]), float(end_parts[2]), float(end_parts[3])],
                            dtype=np.float64,
                        )
                    elif end_line == "}":
                        idx += 1
                        break
                    idx += 1
                end_sites.append(end_offset)
            elif token == "}":
                idx += 1
                nodes[name] = BvhNode(name, parent, offset, channels, channel_start, children, end_sites)
                return name
            else:
                raise ValueError(f"Unexpected BVH token at line {idx + 1}: {stripped}")
        raise ValueError("Unexpected end of BVH hierarchy")

    while idx < len(lines) and lines[idx].strip() != "HIERARCHY":
        idx += 1
    if idx == len(lines):
        raise ValueError("BVH file does not contain HIERARCHY")
    idx += 1
    root = parse_joint(None)

    while idx < len(lines) and lines[idx].strip() != "MOTION":
        idx += 1
    if idx == len(lines):
        raise ValueError("BVH file does not contain MOTION")
    idx += 1
    frame_count = int(lines[idx].split(":", 1)[1].strip())
    idx += 1
    frame_time = float(lines[idx].split(":", 1)[1].strip())
    idx += 1

    values: list[float] = []
    for line in lines[idx:]:
        stripped = line.strip()
        if stripped:
            values.extend(float(x) for x in stripped.split())
    expected = frame_count * channel_cursor
    if len(values) != expected:
        raise ValueError(f"BVH motion channel count mismatch: expected {expected}, got {len(values)}")
    frames = np.asarray(values, dtype=np.float64).reshape(frame_count, channel_cursor)
    return BvhMotion(nodes, root, hierarchy_order, channel_order, frames, frame_time)


def axis_rotation(channel: str, angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    if channel == "Xrotation":
        return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if channel == "Yrotation":
        return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if channel == "Zrotation":
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"Unsupported rotation channel: {channel}")


def compose_channel_rotation(channels: list[str], values: np.ndarray, channel_start: int) -> np.ndarray:
    rot = np.eye(3, dtype=np.float64)
    for offset, channel in enumerate(channels):
        if channel.endswith("rotation"):
            rot = rot @ axis_rotation(channel, values[channel_start + offset])
    return rot


def channel_translation(node: BvhNode, values: np.ndarray) -> np.ndarray:
    pos = np.zeros(3, dtype=np.float64)
    for offset, channel in enumerate(node.channels):
        value = values[node.channel_start + offset]
        if channel == "Xposition":
            pos[0] = value
        elif channel == "Yposition":
            pos[1] = value
        elif channel == "Zposition":
            pos[2] = value
    return pos


def forward_kinematics(motion: BvhMotion) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    frame_count = motion.frame_count
    world_pos = {name: np.zeros((frame_count, 3), dtype=np.float64) for name in motion.nodes}
    world_rot = {name: np.zeros((frame_count, 3, 3), dtype=np.float64) for name in motion.nodes}

    for frame_idx, frame in enumerate(motion.frames):
        def visit(name: str) -> None:
            node = motion.nodes[name]
            local_rot = compose_channel_rotation(node.channels, frame, node.channel_start)
            local_translation = node.offset.copy()
            if any(channel.endswith("position") for channel in node.channels):
                # Match the common BVH loader convention used in the GMR tree:
                # root position channels are absolute and replace ROOT OFFSET.
                local_translation = channel_translation(node, frame)

            if node.parent is None:
                world_rot[name][frame_idx] = local_rot
                world_pos[name][frame_idx] = local_translation
            else:
                parent_rot = world_rot[node.parent][frame_idx]
                world_rot[name][frame_idx] = parent_rot @ local_rot
                world_pos[name][frame_idx] = world_pos[node.parent][frame_idx] + parent_rot @ local_translation

            for child in node.children:
                visit(child)

        visit(motion.root)
    return world_pos, world_rot


def rotation_channels(channels: list[str]) -> list[str]:
    return [channel for channel in channels if channel.endswith("rotation")]


def rotation_to_channel_angles(rot: np.ndarray, rot_channels: list[str]) -> list[float]:
    if not rot_channels:
        return []
    seq = "".join(channel[0] for channel in rot_channels)
    best_angles: Optional[np.ndarray] = None
    best_error = np.inf
    for candidate_seq in (seq.lower(), seq.upper()):
        try:
            angles = R.from_matrix(rot).as_euler(candidate_seq, degrees=True)
        except ValueError:
            continue
        recon = np.eye(3, dtype=np.float64)
        for channel, angle in zip(rot_channels, angles, strict=False):
            recon = recon @ axis_rotation(channel, float(angle))
        error = float(np.linalg.norm(recon - rot))
        if error < best_error:
            best_error = error
            best_angles = np.asarray(angles, dtype=np.float64)
    if best_angles is None:
        raise ValueError(f"Could not decompose rotation for channels {rot_channels}")
    return [float(x) for x in best_angles]


def write_bvh(path: str | Path, template: BvhMotion, frames: np.ndarray, frame_time: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["HIERARCHY"]

    def write_joint(name: str, indent: int) -> None:
        node = template.nodes[name]
        prefix = "\t" * indent
        lines.append(f"{prefix}{'ROOT' if node.parent is None else 'JOINT'} {name}")
        lines.append(f"{prefix}{{")
        lines.append(
            f"{prefix}\tOFFSET {node.offset[0]:.6f} {node.offset[1]:.6f} {node.offset[2]:.6f}"
        )
        if node.channels:
            lines.append(f"{prefix}\tCHANNELS {len(node.channels)} " + " ".join(node.channels))
        for child in node.children:
            write_joint(child, indent + 1)
        for end_offset in node.end_sites:
            lines.append(f"{prefix}\tEnd Site")
            lines.append(f"{prefix}\t{{")
            lines.append(
                f"{prefix}\t\tOFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}"
            )
            lines.append(f"{prefix}\t}}")
        lines.append(f"{prefix}}}")

    write_joint(template.root, 0)
    lines.append("MOTION")
    lines.append(f"Frames: {frames.shape[0]}")
    lines.append(f"Frame Time: {frame_time:.8f}")
    for frame in frames:
        lines.append(" ".join(f"{value:.6f}" for value in frame))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
