from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bvh import BvhMotion


CANONICAL_ROOT_HEIGHT = 0.95
UNIT_TO_METERS = 0.01

FORMAT_NAMES = ("lafan1", "nokov", "wbody")


@dataclass(frozen=True)
class FormatSpec:
    name: str
    raw_to_canonical_rot: np.ndarray
    semantic_to_raw: dict[str, str]
    default_template: Path

    @property
    def raw_to_semantic(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for semantic, raw in self.semantic_to_raw.items():
            out.setdefault(raw, semantic)
        return out


ROOT = Path("/home/hiyio")

LAFAN1_TO_CANONICAL = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

NOKOV_TO_CANONICAL = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

WBODY_TO_CANONICAL = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


COMMON_SEMANTIC = {
    "Hips": "Hips",
    "Spine": "Spine",
    "Spine1": "Spine1",
    "Spine2": "Spine2",
    "Spine3": "Spine3",
    "Neck": "Neck",
    "Neck1": "Neck1",
    "Head": "Head",
    "LeftShoulder": "LeftShoulder",
    "LeftArm": "LeftArm",
    "LeftForeArm": "LeftForeArm",
    "LeftHand": "LeftHand",
    "RightShoulder": "RightShoulder",
    "RightArm": "RightArm",
    "RightForeArm": "RightForeArm",
    "RightHand": "RightHand",
    "LeftUpLeg": "LeftUpLeg",
    "LeftLeg": "LeftLeg",
    "LeftFoot": "LeftFoot",
    "LeftToeBase": "LeftToeBase",
    "RightUpLeg": "RightUpLeg",
    "RightLeg": "RightLeg",
    "RightFoot": "RightFoot",
    "RightToeBase": "RightToeBase",
}

LAFAN_SEMANTIC = dict(COMMON_SEMANTIC)
LAFAN_SEMANTIC.update(
    {
        "LeftToeBase": "LeftToe",
        "RightToeBase": "RightToe",
    }
)
LAFAN_SEMANTIC.pop("Spine3")
LAFAN_SEMANTIC.pop("Neck1")

NOKOV_SEMANTIC = dict(COMMON_SEMANTIC)

WBODY_SEMANTIC = dict(COMMON_SEMANTIC)
WBODY_SEMANTIC.pop("Neck1")

SPECS = {
    "lafan1": FormatSpec(
        "lafan1",
        LAFAN1_TO_CANONICAL,
        LAFAN_SEMANTIC,
        ROOT / "GMR_hxl/Motion_data/lafan1/fight1_subject2.bvh",
    ),
    "nokov": FormatSpec(
        "nokov",
        NOKOV_TO_CANONICAL,
        NOKOV_SEMANTIC,
        ROOT / "engineai/data/official/boxing_motions/raw_bvh/540huixuantitui_001.bvh",
    ),
    "wbody": FormatSpec(
        "wbody",
        WBODY_TO_CANONICAL,
        WBODY_SEMANTIC,
        ROOT / "engineai/data/raw/xiayao2026-Wbody0.bvh",
    ),
}


def detect_format(motion: BvhMotion) -> str:
    nodes = set(motion.nodes)
    if "Spine4" in nodes and "LeftToeBase" in nodes:
        return "wbody"
    if "Spine3" in nodes and "LeftToeBase" in nodes:
        return "nokov"
    if "LeftToe" in nodes and "RightToe" in nodes and "Spine2" in nodes:
        return "lafan1"
    raise ValueError("Unsupported BVH skeleton. Expected lafan1, nokov, or wbody.")


def get_spec(name: str) -> FormatSpec:
    if name == "auto":
        raise ValueError("'auto' is not a concrete format")
    if name not in SPECS:
        raise ValueError(f"Unknown BVH format {name!r}; expected one of {FORMAT_NAMES}")
    return SPECS[name]
