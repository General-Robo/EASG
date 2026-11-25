#!/usr/bin/env python3
"""Validate a converted LIBERO VLM action-SFT directory."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image


ENCODED = re.compile(r"^[01]:\d+\.\d{2}$")
POSE_VECTOR_PATTERN = r"[01]:\d+\.\d{4}(?:,[01]:\d+\.\d{4}){2}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--skip-image-decode", action="store_true")
    return parser.parse_args()


def validate_vector(value: str, context: str) -> None:
    parts = value.split(",")
    if len(parts) != 3 or any(ENCODED.fullmatch(part) is None for part in parts):
        raise ValueError(f"{context}: invalid encoded 3-vector {value!r}")


def validate_answer(answer: str, context: str) -> int:
    if " " in answer or ";" in answer:
        raise ValueError(f"{context}: expected exactly one space-free action step")
    fields = answer.split("|")
    if len(fields) != 7:
        raise ValueError(f"{context}: expected seven fields, got {len(fields)}")
    task_state = int(fields[0])
    if task_state not in (0, 1):
        raise ValueError(f"{context}: task_state must be 0 or 1")
    for index in (1, 2, 4, 5):
        validate_vector(fields[index], context)
    for index in (3, 6):
        if ENCODED.fullmatch(fields[index]) is None:
            raise ValueError(f"{context}: invalid gripper field {fields[index]!r}")
    return task_state


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    manifest = dataset_dir / "samples.jsonl"
    descriptor = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    ids: set[str] = set()
    task_states: Counter[int] = Counter()
    tasks: Counter[str] = Counter()
    image_count = 0
    sample_count = 0

    with manifest.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            context = f"line {line_number}"
            sample_id = row["id"]
            if sample_id in ids:
                raise ValueError(f"{context}: duplicate id {sample_id}")
            ids.add(sample_id)
            if len(row.get("images", [])) != 2:
                raise ValueError(f"{context}: expected two images")
            for relative in row["images"]:
                path = dataset_dir / relative
                if not path.is_file():
                    raise FileNotFoundError(f"{context}: missing image {path}")
                if not args.skip_image_decode:
                    with Image.open(path) as image:
                        image.verify()
                image_count += 1
            answer = row.get("answer", "")
            task_states[validate_answer(answer, context)] += 1
            assistant_text = row["messages"][1]["content"][0]["text"]
            if assistant_text != answer:
                raise ValueError(f"{context}: answer/messages mismatch")
            metadata = row["metadata"]
            for key in ("current_ee_position_m", "current_ee_axis_angle_rad"):
                values = metadata.get(key)
                if not isinstance(values, list) or len(values) != 3:
                    raise ValueError(f"{context}: missing or invalid {key}")
            prompt = row["prompt"]
            for prefix in ("EEF_POSITION_M=", "EEF_AXIS_ANGLE_RAD="):
                if re.search(re.escape(prefix) + POSE_VECTOR_PATTERN, prompt) is None:
                    raise ValueError(f"{context}: missing or invalid {prefix} pose input")
            tasks[row["metadata"]["task_slug"]] += 1
            sample_count += 1

    if sample_count != descriptor["sample_count"]:
        raise ValueError(
            f"dataset.json sample_count={descriptor['sample_count']} but manifest has {sample_count}"
        )
    print(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "sample_count": sample_count,
                "image_count": image_count,
                "task_state_counts": dict(sorted(task_states.items())),
                "task_counts": dict(sorted(tasks.items())),
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
