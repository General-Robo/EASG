#!/usr/bin/env python3
"""Convert LIBERO HDF5 demonstrations to the Qwen VLM action-SFT JSONL format.

Uses the legacy seven-field action schema for the single LIBERO Panda arm.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from PIL import Image


DEFAULT_INPUT_DIR = Path(os.environ.get("LIBERO_ROOT", Path(__file__).resolve().parents[1] / "third_party/LIBERO")) / "libero/datasets/libero_10"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "training_data" / "libero_10"
DEFAULT_PROMPT = Path(__file__).resolve().parents[1] / "prompt.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-file", type=Path, help="Convert one HDF5 task file.")
    source.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--task-pattern", default="*.hdf5", help="Glob used with --input-dir.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-template", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-demos-per-task", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--alignment", choices=("next", "same"), default="next")
    parser.add_argument("--image-format", choices=("png", "jpg"), default="png")
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--image-orientation",
        choices=("auto", "stored", "flip-vertical"),
        default="auto",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse already exported images.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sorted_demos(group: h5py.Group) -> list[str]:
    def index(name: str) -> int:
        match = re.search(r"(\d+)$", name)
        return int(match.group(1)) if match else 0

    return sorted(group.keys(), key=index)


def task_slug(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_demo")] if stem.endswith("_demo") else stem


def format_encoded_decimal(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"Non-finite action value: {value}")
    rounded = round(float(value), 2)
    if abs(rounded) < 0.005:
        rounded = 0.0
    sign = 1 if rounded < 0.0 else 0
    return f"{sign}:{abs(rounded):.2f}"


def format_vector(values: Iterable[float]) -> str:
    values = list(values)
    if len(values) != 3:
        raise ValueError(f"Expected a 3-vector, got {values}")
    return ",".join(format_encoded_decimal(value) for value in values)


def no_op_arm(gripper: float = 0.0) -> dict[str, Any]:
    return {
        "translation_cm": [0.0, 0.0, 0.0],
        "rotation_axis_angle_deg": [0.0, 0.0, 0.0],
        "gripper_state": float(gripper),
    }


def format_answer(task_state: int, action: dict[str, Any]) -> str:
    dummy = no_op_arm()
    return "|".join([
        str(task_state), format_vector(dummy["translation_cm"]),
        format_vector(dummy["rotation_axis_angle_deg"]), "0:0.00",
        format_vector(action["translation_cm"]),
        format_vector(action["rotation_axis_angle_deg"]),
        format_encoded_decimal(action["gripper_state"]),
    ])


def array_parameter(value: Any, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=np.float64)
    return np.broadcast_to(array, (size,)).astype(np.float64, copy=True)


def controller_config(data_group: h5py.Group) -> dict[str, Any]:
    env_args = json.loads(data_group.attrs["env_args"])
    config = env_args["env_kwargs"]["controller_configs"]
    return config


def scale_pose_action(raw_pose: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    size = len(raw_pose)
    input_min = array_parameter(config.get("input_min", -1.0), size)
    input_max = array_parameter(config.get("input_max", 1.0), size)
    output_min = array_parameter(config["output_min"], size)
    output_max = array_parameter(config["output_max"], size)
    clipped = np.clip(raw_pose, input_min, input_max)
    scale = np.abs(output_max - output_min) / np.abs(input_max - input_min)
    input_center = (input_max + input_min) / 2.0
    output_center = (output_max + output_min) / 2.0
    return (clipped - input_center) * scale + output_center


def converted_arm_action(raw_action: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    if raw_action.shape != (7,):
        raise ValueError(f"Expected a 7D LIBERO action, got {raw_action.shape}")
    scaled = scale_pose_action(raw_action[:6], config)
    return {
        "translation_cm": (scaled[:3] * 100.0).tolist(),
        # robosuite OSC_POSE interprets the last three values as a base-frame
        # axis-angle vector in radians and left-multiplies the current rotation.
        "rotation_axis_angle_deg": np.degrees(scaled[3:6]).tolist(),
        # LIBERO Panda convention: -1 opens and +1 closes the gripper.
        "gripper_state": float(np.clip((1.0 - raw_action[6]) / 2.0, 0.0, 1.0)),
    }


def observed_gripper_openness(gripper_qpos: np.ndarray) -> float:
    if gripper_qpos.shape != (2,):
        raise ValueError(f"Expected two gripper joint values, got {gripper_qpos.shape}")
    # Panda fingers are approximately +0.04 and -0.04 when fully open.
    return float(np.clip((gripper_qpos[0] - gripper_qpos[1]) / 0.08, 0.0, 1.0))


def format_prompt_vector(values: np.ndarray) -> str:
    result = []
    for value in np.asarray(values, dtype=np.float64).reshape(3):
        rounded = round(float(value), 4)
        if abs(rounded) < 0.00005:
            rounded = 0.0
        sign = 1 if rounded < 0.0 else 0
        result.append(f"{sign}:{abs(rounded):.4f}")
    return ",".join(result)


def image_needs_flip(mode: str, stored_convention: str) -> bool:
    if mode == "flip-vertical":
        return True
    if mode == "stored":
        return False
    return stored_convention.lower() == "opengl"


def save_image(array: np.ndarray, path: Path, args: argparse.Namespace, flip: bool) -> None:
    if args.resume and path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if flip:
        array = array[::-1]
    image = Image.fromarray(np.ascontiguousarray(array, dtype=np.uint8))
    if args.image_format == "png":
        image.save(path, format="PNG", compress_level=args.png_compress_level)
    else:
        image.save(path, format="JPEG", quality=args.jpeg_quality, subsampling=0)


def render_prompt(
    template: str,
    instruction: str,
    current_gripper: float,
    current_ee_position: np.ndarray,
    current_ee_axis_angle: np.ndarray,
) -> str:
    return template.format(
        instruction=instruction,
        current_gripper=f"{current_gripper:.2f}",
        current_ee_position=format_prompt_vector(current_ee_position),
        current_ee_axis_angle=format_prompt_vector(current_ee_axis_angle),
        active_arm="right",
    )


def source_files(args: argparse.Namespace) -> list[Path]:
    if args.input_file is not None:
        files = [args.input_file.expanduser().resolve()]
    else:
        files = sorted(args.input_dir.expanduser().resolve().glob(args.task_pattern))
    files = [path for path in files if path.is_file()]
    if not files:
        raise FileNotFoundError("No LIBERO HDF5 files matched the requested source.")
    return files


def validate_args(args: argparse.Namespace) -> None:
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.max_demos_per_task is not None and args.max_demos_per_task <= 0:
        raise ValueError("--max-demos-per-task must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if not 0 <= args.png_compress_level <= 9:
        raise ValueError("--png-compress-level must be in [0, 9]")


def main() -> int:
    args = parse_args()
    validate_args(args)
    files = source_files(args)
    output_dir = args.output_dir.expanduser().resolve()
    prompt_path = args.prompt_template.expanduser().resolve()
    prompt_template = prompt_path.read_text(encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "samples.jsonl"
    temporary_manifest = output_dir / "samples.jsonl.tmp"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")

    sample_count = 0
    source_frame_count = 0
    task_summaries: list[dict[str, Any]] = []
    with temporary_manifest.open("w", encoding="utf-8") as manifest:
        stop = False
        for input_path in files:
            slug = task_slug(input_path)
            with h5py.File(input_path, "r") as hdf5_file:
                data = hdf5_file["data"]
                config = controller_config(data)
                problem_info = json.loads(data.attrs["problem_info"])
                instruction = problem_info["language_instruction"]
                convention = str(data.attrs.get("macros_image_convention", "opengl"))
                flip = image_needs_flip(args.image_orientation, convention)
                demos = sorted_demos(data)
                if args.max_demos_per_task is not None:
                    demos = demos[: args.max_demos_per_task]
                task_sample_count = 0
                task_frame_count = 0

                for demo_name in demos:
                    demo = data[demo_name]
                    actions = np.asarray(demo["actions"][:], dtype=np.float64)
                    rewards = np.asarray(demo["rewards"][:], dtype=np.uint8)
                    dones = np.asarray(demo["dones"][:], dtype=np.uint8)
                    gripper_states = np.asarray(demo["obs/gripper_states"][:], dtype=np.float64)
                    ee_positions = np.asarray(demo["obs/ee_pos"][:], dtype=np.float64)
                    ee_orientations = np.asarray(demo["obs/ee_ori"][:], dtype=np.float64)
                    length = len(actions)
                    if not (
                        len(rewards)
                        == len(dones)
                        == len(gripper_states)
                        == len(ee_positions)
                        == len(ee_orientations)
                        == length
                    ):
                        raise ValueError(f"Length mismatch in {input_path}:{demo_name}")
                    source_frame_count += length
                    task_frame_count += length

                    frame_indices = list(range(0, length, args.frame_stride))
                    if frame_indices[-1] != length - 1:
                        frame_indices.append(length - 1)

                    for frame_index in frame_indices:
                        if args.max_samples is not None and sample_count >= args.max_samples:
                            stop = True
                            break

                        terminal_observation = frame_index == length - 1
                        if terminal_observation:
                            action_index = length - 1
                            active = no_op_arm(converted_arm_action(actions[-1], config)["gripper_state"])
                            task_state = int(max(rewards[-1], dones[-1]))
                        else:
                            action_index = frame_index + 1 if args.alignment == "next" else frame_index
                            active = converted_arm_action(actions[action_index], config)
                            task_state = int(rewards[action_index])

                        answer = format_answer(task_state, active)

                        sample_id = f"{slug}_{demo_name}_frame_{frame_index:06d}"
                        image_dir = output_dir / "images" / slug / demo_name
                        extension = args.image_format
                        agent_rel = Path("images") / slug / demo_name / f"frame_{frame_index:06d}_agentview.{extension}"
                        wrist_rel = Path("images") / slug / demo_name / f"frame_{frame_index:06d}_eye_in_hand.{extension}"
                        save_image(
                            demo["obs/agentview_rgb"][frame_index],
                            output_dir / agent_rel,
                            args,
                            flip,
                        )
                        save_image(
                            demo["obs/eye_in_hand_rgb"][frame_index],
                            output_dir / wrist_rel,
                            args,
                            flip,
                        )

                        current_gripper = observed_gripper_openness(gripper_states[frame_index])
                        current_ee_position = ee_positions[frame_index]
                        current_ee_axis_angle = ee_orientations[frame_index]
                        prompt = render_prompt(
                            prompt_template,
                            instruction,
                            current_gripper,
                            current_ee_position,
                            current_ee_axis_angle,
                        )
                        relative_images = [agent_rel.as_posix(), wrist_rel.as_posix()]
                        row = {
                            "id": sample_id,
                            "images": relative_images,
                            "prompt": prompt,
                            "answer": answer,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": relative_images[0]},
                                        {"type": "image", "image": relative_images[1]},
                                        {"type": "text", "text": prompt},
                                    ],
                                },
                                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                            ],
                            "metadata": {
                                "source_hdf5": str(input_path),
                                "task_slug": slug,
                                "instruction": instruction,
                                "demo": demo_name,
                                "observation_frame_index": frame_index,
                                "supervised_action_index": action_index,
                                "alignment": "terminal_no_op" if terminal_observation else args.alignment,
                                "terminal_observation": terminal_observation,
                                "task_state": task_state,
                                "current_gripper_openness": current_gripper,
                                "current_ee_position_m": current_ee_position.tolist(),
                                "current_ee_axis_angle_rad": current_ee_axis_angle.tolist(),
                                "raw_libero_action": actions[action_index].tolist(),
                                "action": active,
                                "action_coordinate_frame": "robosuite_base_frame",
                                "rotation_parameterization": "axis_angle_vector_degrees",
                                "translation_unit": "centimeters",
                                "source_image_convention": convention,
                                "exported_vertical_flip": flip,
                            },
                        }
                        manifest.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        sample_count += 1
                        task_sample_count += 1

                    print(
                        f"[{slug}] {demo_name}: frames={length}, "
                        f"total_samples={sample_count}",
                        flush=True,
                    )
                    if stop:
                        break

                task_summaries.append(
                    {
                        "source_hdf5": str(input_path),
                        "task_slug": slug,
                        "instruction": instruction,
                        "demo_count": len(demos),
                        "source_frame_count": task_frame_count,
                        "sample_count": task_sample_count,
                    }
                )
            if stop:
                break

    temporary_manifest.replace(manifest_path)
    dataset = {
        "format": "vlmasam_dual_arm_multimodal_sft_single_step_v3",
        "created_at_utc": utc_now(),
        "source": "LIBERO HDF5 demonstrations",
        "tasks": task_summaries,
        "source_frame_count": source_frame_count,
        "sample_count": sample_count,
        "frame_stride": args.frame_stride,
        "alignment": args.alignment,
        "include_terminal_no_op": True,
        "image_order": ["agentview", "eye_in_hand"],
        "prompt_state_inputs": [
            "current_gripper_openness",
            "current_ee_position_m",
            "current_ee_axis_angle_rad",
        ],
        "image_format": args.image_format,
        "image_orientation": args.image_orientation,
        "action_step_count": 1,
        "action_separator": ";",
        "action_coordinate_frame": "robosuite_base_frame",
        "translation_unit": "centimeters",
        "rotation_parameterization": "axis_angle_vector_degrees",
        "gripper_parameterization": "0_closed_1_open",
        "prompt_template": str(prompt_path),
        "conversion": {
            "translation": "normalized OSC input scaled by controller output limits, then meters to centimeters",
            "rotation": "normalized OSC input scaled to axis-angle radians, then converted to degrees",
            "gripper": "LIBERO -1=open,+1=close mapped to 1=open,0=closed",
        },
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {sample_count} samples to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
