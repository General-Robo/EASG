"""Physical/task settings loaded by name from EASG/config, independent of cwd."""

from dataclasses import MISSING, asdict, dataclass, fields
import argparse
import secrets
import json
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config"
HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


@dataclass(frozen=True)
class SceneConfig:
    gripper_max_width_m: float
    max_translation_cm: float
    max_rotation_deg: float
    table_height_m: float
    action_duration_s: float
    arm_duration_s: float
    gripper_duration_s: float
    initial_pose_duration_s: float
    action_settle_s: float
    completion_settle_s: float
    force_backoff_duration_s: float
    workmat_thickness_m: float = 0.020
    workmat_contact_stiffness: float = 1000.0
    workmat_contact_damping: float = 10.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field.name} must be positive and finite")
        # This is an asset limit, not a configurable opening target.
        if self.gripper_max_width_m > 0.08:
            raise ValueError(
                "gripper_max_width_m exceeds the Franka asset limit of 0.08 m"
            )

    @property
    def workmat_top_m(self):
        return self.table_height_m + self.workmat_thickness_m

    def to_dict(self):
        return asdict(self)


def load_config(name="default.json"):
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or name in (".", "..")
    ):
        raise ValueError("--config must be a filename in EASG/config, not a path")
    path = CONFIG_DIR / name
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    expected = {field.name for field in fields(SceneConfig)}
    required = {field.name for field in fields(SceneConfig) if field.default is MISSING}
    if required - set(data) or set(data) - expected:
        raise ValueError(
            f"Invalid config fields: missing={sorted(required - set(data))}, unknown={sorted(set(data) - expected)}"
        )
    return SceneConfig(**data)


def config_from_args(parser, args):
    try:
        return load_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))


def add_position_arguments(parser):
    parser.add_argument(
        "--position-randomization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Randomize object/goal XY only; default off, 5 cm edge margin, 3 cm rectangle gap",
    )
    parser.add_argument(
        "--position-seed",
        type=int,
        help="Layout seed; generated and saved when omitted",
    )


def position_config_from_args(parser, args):
    from construct_scene.layout import PositionRandomization

    if not args.position_randomization:
        if args.position_seed is not None:
            parser.error("--position-seed requires --position-randomization")
        return None
    try:
        return PositionRandomization(
            args.position_seed
            if args.position_seed is not None
            else secrets.randbits(64)
        )
    except ValueError as error:
        parser.error(str(error))
