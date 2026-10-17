from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SIGNED_FIXED_2_RE = re.compile(r"[+-]\d+\.\d{2}")


class VLMParseError(ValueError):
    """Raised when a VLM motion response cannot be parsed safely."""


@dataclass(frozen=True)
class TranslationCommand:
    direction: np.ndarray
    distance_cm: float
    reason: str = ""


@dataclass(frozen=True)
class RotationCommand:
    axis: np.ndarray
    angle_deg: float
    reason: str = ""


@dataclass(frozen=True)
class GripperCommand:
    state: float = 1.0
    reason: str = ""
    provided: bool = True


@dataclass(frozen=True)
class MotionCommand:
    task_state: int
    translation: TranslationCommand
    rotation: RotationCommand
    gripper: GripperCommand = field(
        default_factory=lambda: GripperCommand(provided=False)
    )


def parse_motion_file(path: str | Path) -> MotionCommand:
    return parse_vlm_motion(Path(path).read_text())


def parse_vlm_motion(text: str) -> MotionCommand:
    """Parse the single supported pipe action format (signed, two decimals)."""
    if not isinstance(text, str) or not text.strip():
        raise VLMParseError("Empty VLM response.")
    text = text.strip()
    if any(character.isspace() for character in text):
        raise VLMParseError("Pipe motion response must not contain spaces or line breaks.")

    fields = text.split("|")
    if len(fields) != 4:
        raise VLMParseError(
            "Pipe motion response must contain exactly four fields separated by '|'."
        )

    task_state_text, translation_text, rotation_text, gripper_text = fields
    if task_state_text not in ("0", "1"):
        raise VLMParseError("Pipe task_state must be the integer 0 or 1.")

    translation_values = _parse_pipe_values(
        translation_text,
        expected_count=4,
        field_name="translation",
    )
    rotation_values = _parse_pipe_values(
        rotation_text,
        expected_count=4,
        field_name="rotation",
    )
    gripper_values = _parse_pipe_values(
        gripper_text,
        expected_count=1,
        field_name="gripper",
    )
    gripper_state = gripper_values[0]
    if not 0.0 <= gripper_state <= 1.0:
        raise VLMParseError(
            f"gripper state must be between 0.0 and 1.0, got {gripper_state}."
        )

    return MotionCommand(
        task_state=int(task_state_text),
        translation=TranslationCommand(
            direction=np.asarray(translation_values[:3], dtype=float),
            distance_cm=translation_values[3],
        ),
        rotation=RotationCommand(
            axis=np.asarray(rotation_values[:3], dtype=float),
            angle_deg=rotation_values[3],
        ),
        gripper=GripperCommand(state=gripper_state, provided=True),
    )


def _parse_pipe_values(
    text: str,
    expected_count: int,
    field_name: str,
) -> list[float]:
    values = text.split(",")
    if len(values) != expected_count:
        raise VLMParseError(
            f"Pipe {field_name} field must contain exactly {expected_count} values."
        )
    if any(SIGNED_FIXED_2_RE.fullmatch(value) is None for value in values):
        raise VLMParseError(
            f"Every pipe {field_name} value must include a sign and two decimals."
        )
    parsed = [float(value) for value in values]
    if not np.all(np.isfinite(parsed)):
        raise VLMParseError(f"Pipe {field_name} contains a non-finite value.")
    return parsed


def motion_to_dict(motion: MotionCommand) -> dict[str, Any]:
    return {
        "task_state": int(motion.task_state),
        "translation": {
            "direction": np.asarray(motion.translation.direction, dtype=float).tolist(),
            "distance_cm": float(motion.translation.distance_cm),
            "reason": motion.translation.reason,
        },
        "rotation": {
            "axis": np.asarray(motion.rotation.axis, dtype=float).tolist(),
            "angle_deg": float(motion.rotation.angle_deg),
            "reason": motion.rotation.reason,
        },
        "gripper": {
            "state": float(motion.gripper.state),
            "reason": motion.gripper.reason,
            "provided": bool(motion.gripper.provided),
        },
    }

def motion_as_pipe(motion):
    translation = [*motion.translation.direction, motion.translation.distance_cm]
    rotation = [*motion.rotation.axis, motion.rotation.angle_deg]
    return '|'.join([str(motion.task_state), ','.join(f'{x:+.2f}' for x in translation),
                     ','.join(f'{x:+.2f}' for x in rotation), f'{motion.gripper.state:+.2f}'])


