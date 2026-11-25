#!/usr/bin/env python3
"""Evaluate a Qwen3.5 LoRA policy on LIBERO with parallel simulator workers.

Evaluate every task in the selected suite. Each simulator owns a persistent
model subprocess in --model-python.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = Path(os.environ.get("LIBERO_ROOT", PROJECT_ROOT / "third_party/LIBERO"))
DEFAULT_MODEL_PATH = Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / "models/Qwen3.5-0.8B"))
BUNDLED_ADAPTERS = {
    "libero_spatial": "spatial",
    "libero_goal": "goal",
    "libero_object": "object",
    "libero_10": "long",
}
DEFAULT_PROMPT = PROJECT_ROOT / "prompt.txt"
DEFAULT_MODEL_PYTHON = Path(os.environ.get("MODEL_PYTHON", sys.executable))
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/rollouts/evaluate"
DEFAULT_DEVICES = "cuda:0"
READY_PREFIX = "__QWEN_ROLLOUT_READY__"
RESPONSE_PREFIX = "__QWEN_ROLLOUT_RESPONSE__"
ENCODED_DECIMAL = re.compile(r"([01]):((?:\d+(?:\.\d*)?)|(?:\.\d+))")
ACTION_FIELD_ARITIES = (3, 3, 1, 3, 3, 1)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_common_args(args: argparse.Namespace) -> None:
    for name in ("model_path", "adapter_path"):
        path = Path(getattr(args, name)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{name.replace('_', ' ')} does not exist: {path}")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")


def resolve_adapter_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "adapter_config.json").is_file() and (path / "adapter_model.safetensors").is_file():
        return path
    candidates = []
    for checkpoint in path.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
        if match and (checkpoint / "adapter_config.json").is_file() and (checkpoint / "adapter_model.safetensors").is_file():
            candidates.append((int(match.group(1)), checkpoint))
    if not candidates:
        raise FileNotFoundError(f"No PEFT adapter found in {path}")
    return max(candidates, key=lambda item: item[0])[1]


def format_prompt_vector(values: Iterable[float]) -> str:
    result = []
    for value in values:
        rounded = round(float(value), 4)
        if abs(rounded) < 0.00005:
            rounded = 0.0
        sign = 1 if rounded < 0.0 else 0
        result.append(f"{sign}:{abs(rounded):.4f}")
    if len(result) != 3:
        raise ValueError(f"Expected a 3-vector, got {len(result)} values")
    return ",".join(result)


def observed_gripper_openness(gripper_qpos: Any) -> float:
    import numpy as np

    values = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
    if values.shape != (2,):
        raise ValueError(f"Expected robot0_gripper_qpos with shape (2,), got {values.shape}")
    return float(np.clip((values[0] - values[1]) / 0.08, 0.0, 1.0))


def render_prompt(template: str, instruction: str, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    import numpy as np
    from robosuite.utils import transform_utils as transform

    position = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
    quaternion = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(4)
    axis_angle = np.asarray(transform.quat2axisangle(quaternion), dtype=np.float64).reshape(3)
    gripper = observed_gripper_openness(obs["robot0_gripper_qpos"])
    prompt = template.format(
        instruction=instruction,
        current_gripper=f"{gripper:.2f}",
        current_ee_position=format_prompt_vector(position),
        current_ee_axis_angle=format_prompt_vector(axis_angle),
        active_arm="right",
    )
    state = {
        "gripper_openness": gripper,
        "eef_position_m": position.tolist(),
        "eef_quaternion_xyzw": quaternion.tolist(),
        "eef_axis_angle_rad": axis_angle.tolist(),
    }
    return prompt, state


def vertically_flipped_rgb(obs: dict[str, Any], key: str) -> Any:
    import numpy as np

    image = np.asarray(obs[key], dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image for {key}, got {image.shape}")
    return np.ascontiguousarray(image[::-1])


def encode_png(image: Any) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_encoded_decimal(text: str) -> float:
    match = ENCODED_DECIMAL.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"Invalid sign_bit:magnitude value: {text!r}")
    value = float(match.group(2))
    return -value if match.group(1) == "1" else value


def parse_vector(text: str) -> list[float]:
    parts = text.split(",")
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated values: {text!r}")
    return [parse_encoded_decimal(part) for part in parts]


class ActionTokenGrammar:
    """Token-level grammar for a complete seven-field action action."""

    _DIGITS = tuple(str(value) for value in range(10))
    _SPECIAL_CHARACTERS = (":", ".", ",", "|")

    def __init__(self, tokenizer: Any, eos_token_ids: Iterable[int]) -> None:
        characters = self._DIGITS + self._SPECIAL_CHARACTERS
        self.token_id_by_character = {
            character: self._single_token_id(tokenizer, character)
            for character in characters
        }
        self.character_by_token_id = {
            token_id: character
            for character, token_id in self.token_id_by_character.items()
        }
        self.eos_token_ids = tuple(dict.fromkeys(int(value) for value in eos_token_ids))
        if not self.eos_token_ids:
            raise ValueError("At least one EOS token id is required for constrained decoding")

    @staticmethod
    def _single_token_id(tokenizer: Any, character: str) -> int:
        token_ids = tokenizer(character, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(
                f"Action grammar character {character!r} is not a single token: {token_ids}"
            )
        return int(token_ids[0])

    def _token_ids(self, characters: Iterable[str]) -> list[int]:
        return [self.token_id_by_character[character] for character in characters]

    def allowed_next_token_ids(self, generated_token_ids: Iterable[int]) -> list[int]:
        state = "task"
        field_index = 0
        number_index = 0

        for position, raw_token_id in enumerate(generated_token_ids):
            token_id = int(raw_token_id)
            character = self.character_by_token_id.get(token_id)
            if character is None:
                raise ValueError(
                    f"Generated token {token_id} at action position {position} "
                    "is outside the constrained action alphabet"
                )

            if state == "task":
                if character not in ("0", "1"):
                    self._invalid_prefix(position, character, state)
                state = "task_pipe"
            elif state == "task_pipe":
                if character != "|":
                    self._invalid_prefix(position, character, state)
                state = "number_sign"
            elif state == "number_sign":
                if character not in ("0", "1"):
                    self._invalid_prefix(position, character, state)
                state = "number_colon"
            elif state == "number_colon":
                if character != ":":
                    self._invalid_prefix(position, character, state)
                state = "integer_first"
            elif state == "integer_first":
                if character not in self._DIGITS:
                    self._invalid_prefix(position, character, state)
                state = "integer_second_or_point"
            elif state == "integer_second_or_point":
                if character == ".":
                    state = "fraction_first"
                elif character in self._DIGITS:
                    state = "decimal_point"
                else:
                    self._invalid_prefix(position, character, state)
            elif state == "decimal_point":
                if character != ".":
                    self._invalid_prefix(position, character, state)
                state = "fraction_first"
            elif state == "fraction_first":
                if character not in self._DIGITS:
                    self._invalid_prefix(position, character, state)
                state = "fraction_second"
            elif state == "fraction_second":
                if character not in self._DIGITS:
                    self._invalid_prefix(position, character, state)
                if number_index + 1 < ACTION_FIELD_ARITIES[field_index]:
                    state = "number_comma"
                elif field_index + 1 < len(ACTION_FIELD_ARITIES):
                    state = "field_pipe"
                else:
                    state = "complete"
            elif state == "number_comma":
                if character != ",":
                    self._invalid_prefix(position, character, state)
                number_index += 1
                state = "number_sign"
            elif state == "field_pipe":
                if character != "|":
                    self._invalid_prefix(position, character, state)
                field_index += 1
                number_index = 0
                state = "number_sign"
            else:
                self._invalid_prefix(position, character, state)

        allowed_characters = {
            "task": ("0", "1"),
            "task_pipe": ("|",),
            "number_sign": ("0", "1"),
            "number_colon": (":",),
            "integer_first": self._DIGITS,
            "integer_second_or_point": self._DIGITS + (".",),
            "decimal_point": (".",),
            "fraction_first": self._DIGITS,
            "fraction_second": self._DIGITS,
            "number_comma": (",",),
            "field_pipe": ("|",),
        }
        if state == "complete":
            return list(self.eos_token_ids)
        return self._token_ids(allowed_characters[state])

    @staticmethod
    def _invalid_prefix(position: int, character: str, state: str) -> None:
        raise ValueError(
            f"Invalid constrained action prefix at position {position}: "
            f"character={character!r}, state={state}"
        )


@dataclass
class ParsedAction:
    task_state: int
    left_translation_cm: list[float]
    left_rotation_deg: list[float]
    left_gripper: float
    right_translation_cm: list[float]
    right_rotation_deg: list[float]
    right_gripper: float
    matched_text: str


def parse_action_text(prediction: str) -> ParsedAction:
    clean = prediction
    for marker in ("<|im_end|>", "<|endoftext|>"):
        clean = clean.split(marker, 1)[0]
    clean = clean.strip()
    candidates = [line.strip() for line in clean.splitlines() if line.strip()]
    candidates.insert(0, clean)
    errors = []
    for candidate in candidates:
        start = min((index for index in (candidate.find("0|"), candidate.find("1|")) if index >= 0), default=-1)
        if start < 0:
            continue
        candidate = candidate[start:].split(";", 1)[0].strip()
        fields = candidate.split("|")
        if len(fields) != 7:
            errors.append(f"expected 7 fields, got {len(fields)}")
            continue
        try:
            task_state = int(fields[0])
            if task_state not in (0, 1):
                raise ValueError("task_state must be 0 or 1")
            parsed = ParsedAction(
                task_state=task_state,
                left_translation_cm=parse_vector(fields[1]),
                left_rotation_deg=parse_vector(fields[2]),
                left_gripper=parse_encoded_decimal(fields[3]),
                right_translation_cm=parse_vector(fields[4]),
                right_rotation_deg=parse_vector(fields[5]),
                right_gripper=parse_encoded_decimal(fields[6]),
                matched_text="|".join(fields),
            )
            return parsed
        except ValueError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors[-3:]) if errors else "no action-shaped text found"
    raise ValueError(f"Could not parse model action: {detail}. Prediction={prediction!r}")


def array_parameter(value: Any, size: int) -> Any:
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=np.float64)
    return np.broadcast_to(array, (size,)).copy()


def inverse_scale_pose_action(physical_pose: Any, controller_config: dict[str, Any]) -> tuple[Any, Any]:
    import numpy as np

    physical_pose = np.asarray(physical_pose, dtype=np.float64).reshape(6)
    input_min = array_parameter(controller_config.get("input_min", -1.0), 6)
    input_max = array_parameter(controller_config.get("input_max", 1.0), 6)
    output_min = array_parameter(controller_config["output_min"], 6)
    output_max = array_parameter(controller_config["output_max"], 6)
    scale = np.abs(output_max - output_min) / np.abs(input_max - input_min)
    input_center = (input_max + input_min) / 2.0
    output_center = (output_max + output_min) / 2.0
    unclipped = (physical_pose - output_center) / scale + input_center
    return unclipped, np.clip(unclipped, input_min, input_max)


def parsed_to_libero_action(parsed: ParsedAction, controller_config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    physical_pose = np.concatenate(
        [
            np.asarray(parsed.right_translation_cm, dtype=np.float64) / 100.0,
            np.radians(np.asarray(parsed.right_rotation_deg, dtype=np.float64)),
        ]
    )
    unclipped_pose, pose = inverse_scale_pose_action(physical_pose, controller_config)
    gripper_unclipped = 1.0 - 2.0 * float(parsed.right_gripper)
    gripper = float(np.clip(gripper_unclipped, -1.0, 1.0))
    action = np.concatenate([pose, [gripper]]).astype(np.float64)
    conversion = {
        "physical_pose_m_rad": physical_pose.tolist(),
        "normalized_pose_unclipped": unclipped_pose.tolist(),
        "normalized_action": action.tolist(),
        "pose_was_clipped": bool(np.any(np.abs(unclipped_pose - pose) > 1e-12)),
        "gripper_was_clipped": bool(abs(gripper_unclipped - gripper) > 1e-12),
    }
    return action, conversion


class QwenPolicyWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        command = [
            str(self.args.model_python.expanduser().resolve()),
            str(Path(__file__).resolve()),
            "--model-worker",
            "--model-path", str(self.args.model_path.expanduser().resolve()),
            "--adapter-path", str(resolve_adapter_path(self.args.adapter_path)),
            "--device", self.args.device,
            "--torch-dtype", self.args.torch_dtype,
            "--attn-implementation", self.args.attn_implementation,
            "--max-new-tokens", str(self.args.max_new_tokens),
        ]
        command.append(
            "--constrained-decoding"
            if self.args.constrained_decoding
            else "--no-constrained-decoding"
        )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        event = self._read_event(READY_PREFIX)
        if event.get("status") != "ready":
            raise RuntimeError(f"Model worker did not become ready: {event}")

    def _read_event(self, prefix: str) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Model worker is not running")
        while True:
            line = self.process.stdout.readline()
            if not line:
                code = self.process.poll()
                raise RuntimeError(f"Model worker exited unexpectedly with code {code}")
            line = line.rstrip("\n")
            if line.startswith(prefix):
                return json.loads(line[len(prefix):])
            print(f"[model] {line}", flush=True)

    def predict(self, prompt: str, images: list[Any]) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Model worker is not running")
        request = {"prompt": prompt, "images_png_base64": [encode_png(image) for image in images]}
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response = self._read_event(RESPONSE_PREFIX)
        if response.get("error"):
            raise RuntimeError(f"Model inference failed: {response['error']}")
        return response

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None and self.process.stdin is not None:
            try:
                self.process.stdin.write('{"command":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
        self.process = None


def model_class(transformers_module: Any) -> Any:
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        cls = getattr(transformers_module, name, None)
        if cls is not None:
            return cls
    raise ImportError("Could not find a usable Transformers auto model class")


def decode_prediction(tokenizer: Any, token_ids: Any) -> tuple[str, str]:
    raw = tokenizer.decode(token_ids, skip_special_tokens=False).strip()
    clean = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    for marker in ("<|im_end|>", "<|endoftext|>"):
        clean = clean.replace(marker, "")
    return raw, clean.strip()


def run_model_worker(args: argparse.Namespace) -> int:
    import torch
    import transformers
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoProcessor

    validate_common_args(args)
    adapter_path = resolve_adapter_path(args.adapter_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the Qwen model worker")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]
    model_path = args.model_path.expanduser().resolve()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    base_model = model_class(transformers).from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model = PeftModel.from_pretrained(base_model, adapter_path).eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    print(READY_PREFIX + json.dumps({"status": "ready", "adapter_path": str(adapter_path)}), flush=True)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                return 0
            images = [Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB") for value in request["images_png_base64"]]
            messages = [
                {
                    "role": "user",
                    "content": [
                        *({"type": "image", "image": image} for image in images),
                        {"type": "text", "text": request["prompt"]},
                    ],
                }
            ]
            started = time.perf_counter()
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False,
                )
            except TypeError:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            inputs = {key: value.to(args.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            tokenizer = processor.tokenizer
            im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            endoftext_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            eos_token_ids = list(dict.fromkeys((im_end_token_id, endoftext_token_id)))
            generation_kwargs: dict[str, Any] = {}
            if args.constrained_decoding:
                grammar = ActionTokenGrammar(tokenizer, eos_token_ids)
                prompt_tokens = int(inputs["input_ids"].shape[1])

                def allowed_action_tokens(batch_id: int, token_ids: Any) -> list[int]:
                    del batch_id
                    generated_ids = token_ids[prompt_tokens:].detach().cpu().tolist()
                    return grammar.allowed_next_token_ids(generated_ids)

                generation_kwargs["prefix_allowed_tokens_fn"] = allowed_action_tokens
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=eos_token_ids,
                    pad_token_id=endoftext_token_id,
                    **generation_kwargs,
                )
            prompt_tokens = int(inputs["input_ids"].shape[1])
            new_ids = output_ids[0, prompt_tokens:]
            raw, prediction = decode_prediction(processor.tokenizer, new_ids)
            response = {
                "prediction": prediction,
                "prediction_raw": raw,
                "prompt_tokens": prompt_tokens,
                "output_tokens": int(new_ids.numel()),
                "latency_s": time.perf_counter() - started,
            }
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(RESPONSE_PREFIX + json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


def write_libero_config(config_dir: Path, libero_root: Path) -> None:
    benchmark_root = libero_root / "libero/libero"
    values = {
        "assets": benchmark_root / "assets",
        "bddl_files": benchmark_root / "bddl_files",
        "benchmark_root": benchmark_root,
        "datasets": libero_root / "libero/datasets",
        "init_states": benchmark_root / "init_files",
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{key}: {value.resolve()}\n" for key, value in values.items())
    (config_dir / "config.yaml").write_text(text, encoding="utf-8")


def configure_libero(args: argparse.Namespace, output_dir: Path) -> None:
    libero_root = args.libero_root.expanduser().resolve()
    config_dir = output_dir / ".libero_config"
    write_libero_config(config_dir, libero_root)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))


def json_safe(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_safe) + "\n", encoding="utf-8")


def create_video_writer(path: Path, fps: int) -> Any:
    import imageio.v2 as imageio

    return imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=None)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path, help="PEFT adapter directory (defaults to the bundled adapter for --benchmark)")
    parser.add_argument("--model-python", type=Path, default=DEFAULT_MODEL_PYTHON)
    parser.add_argument("--libero-root", type=Path, default=LIBERO_ROOT)
    parser.add_argument("--prompt-template", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--benchmark", default="libero_goal")
    parser.add_argument("--episodes-per-task", type=int, default=16)
    parser.add_argument("--start-state-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--devices", default=DEFAULT_DEVICES,
                        help="Comma-separated CUDA devices shared by the workers")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    parser.add_argument("--render-gpu-device-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--torch-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--constrained-decoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Constrain generation to the complete seven-field action action grammar.",
    )
    parser.add_argument("--parse-error-policy", choices=("stop", "noop"), default="stop")
    parser.add_argument("--stop-on-task-state", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--save-video", action="store_true", help="Also save a side-by-side MP4 per episode")
    parser.add_argument("--save-input-images", action="store_true", help="Save per-step policy input images")
    args = parser.parse_args()
    if args.adapter_path is None:
        if args.benchmark not in BUNDLED_ADAPTERS:
            parser.error("supply --adapter-path for a benchmark without a bundled adapter")
        args.adapter_path = PROJECT_ROOT / "checkpoints" / BUNDLED_ADAPTERS[args.benchmark] / "checkpoint"
    return args


def wprint(args: argparse.Namespace, message: str) -> None:
    print(f"[w{args.worker_id}] {message}", flush=True)


def load_benchmark_tasks(args: argparse.Namespace) -> tuple[Any, list[int]]:
    from libero.libero import benchmark

    benchmark_class = benchmark.get_benchmark_dict().get(args.benchmark.lower())
    if benchmark_class is None:
        raise ValueError(f"Unknown LIBERO benchmark: {args.benchmark}")
    benchmark_instance = benchmark_class()
    num_tasks = benchmark_instance.get_num_tasks()
    task_ids = list(range(num_tasks))
    return benchmark_instance, task_ids


def run_episode(
    args: argparse.Namespace,
    policy: QwenPolicyWorker,
    env: Any,
    init_states: Any,
    instruction: str,
    prompt_template: str,
    controller_config: dict[str, Any],
    task_dir: Path,
    episode_index: int,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    state_index = (args.start_state_index + episode_index) % len(init_states)
    rollout_dir = task_dir / f"rollout_{episode_index:03d}_state_{state_index:03d}"
    rollout_dir.mkdir(parents=True)
    images_dir = rollout_dir / "images"
    if args.save_input_images:
        images_dir.mkdir()
    steps_path = rollout_dir / "steps.jsonl"
    video_writer = None
    if args.save_video:
        try:
            video_writer = create_video_writer(rollout_dir / "rollout.mp4", args.video_fps)
        except Exception as exc:
            wprint(args, f"[warning] video disabled: {type(exc).__name__}: {exc}")

    env.seed(args.seed + episode_index)
    env.reset()
    obs = env.set_init_state(init_states[state_index])
    for _ in range(args.warmup_steps):
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float64))
    success = bool(env.check_success())
    stop_reason = "initial_success" if success else "max_steps"
    parse_errors = 0
    rollout_started = time.perf_counter()
    with steps_path.open("w", encoding="utf-8") as step_log:
        for step in range(args.max_steps):
            if success:
                break
            agent_image = vertically_flipped_rgb(obs, "agentview_image")
            wrist_image = vertically_flipped_rgb(obs, "robot0_eye_in_hand_image")
            if video_writer is not None:
                video_writer.append_data(np.concatenate([agent_image, wrist_image], axis=1))
            if args.save_input_images:
                Image.fromarray(agent_image).save(images_dir / f"step_{step:04d}_agentview.png", compress_level=1)
                Image.fromarray(wrist_image).save(images_dir / f"step_{step:04d}_eye_in_hand.png", compress_level=1)
            prompt, robot_state = render_prompt(prompt_template, instruction, obs)
            response = policy.predict(prompt, [agent_image, wrist_image])
            parse_error = None
            try:
                parsed = parse_action_text(response["prediction"])
                action, conversion = parsed_to_libero_action(parsed, controller_config)
            except ValueError as exc:
                parse_errors += 1
                parse_error = str(exc)
                if args.parse_error_policy == "stop":
                    record = {
                        "step": step,
                        "prompt": prompt,
                        "robot_state": robot_state,
                        "model": response,
                        "parse_error": parse_error,
                    }
                    step_log.write(json.dumps(record, ensure_ascii=False, default=json_safe) + "\n")
                    step_log.flush()
                    stop_reason = "parse_error"
                    wprint(args, f"episode={episode_index} step={step} parse_error={parse_error}")
                    break
                parsed = None
                action = np.zeros(7, dtype=np.float64)
                action[6] = float(np.clip(1.0 - 2.0 * robot_state["gripper_openness"], -1.0, 1.0))
                conversion = {"normalized_action": action.tolist(), "fallback": "noop"}

            obs, reward, done, info = env.step(action)
            success = bool(env.check_success())
            record = {
                "step": step,
                "prompt": prompt,
                "robot_state": robot_state,
                "model": response,
                "parsed_action": None if parsed is None else parsed.__dict__,
                "action_conversion": conversion,
                "parse_error": parse_error,
                "reward": float(reward),
                "done": bool(done),
                "success": success,
                "info": info,
            }
            step_log.write(json.dumps(record, ensure_ascii=False, default=json_safe) + "\n")
            step_log.flush()
            if success:
                stop_reason = "environment_success"
                break
            if parsed is not None and parsed.task_state == 1 and args.stop_on_task_state:
                stop_reason = "model_task_state"
                break
            if bool(done):
                stop_reason = "environment_done"
                break

    if video_writer is not None:
        video_writer.close()
    summary = {
        "episode_index": episode_index,
        "init_state_index": state_index,
        "success": success,
        "stop_reason": stop_reason,
        "parse_errors": parse_errors,
        "elapsed_s": time.perf_counter() - rollout_started,
        "steps_log": str(steps_path),
    }
    save_json(rollout_dir / "summary.json", summary)
    wprint(args, f"episode={episode_index} state={state_index} success={success} reason={stop_reason}")
    return summary


def run_task(
    args: argparse.Namespace,
    policy: QwenPolicyWorker,
    benchmark_instance: Any,
    task_id: int,
    prompt_template: str,
    controller_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.bddl_utils import get_problem_info

    task = benchmark_instance.get_task(task_id)
    bddl_file = benchmark_instance.get_task_bddl_file_path(task_id)
    # Training data uses the BDDL :language field (see convert_libero_hdf5.py),
    # which can differ from the filename-derived task.language.
    instruction = get_problem_info(str(bddl_file))["language_instruction"]
    init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    init_states = torch.load(init_states_path, weights_only=False)
    task_dir = output_dir / f"task_{task_id:02d}_{task.name}"
    task_dir.mkdir(parents=True, exist_ok=True)
    wprint(args, f"task={task_id} {task.name} instruction={instruction!r} episodes={args.episodes_per_task}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
        render_gpu_device_id=args.render_gpu_device_id,
        horizon=args.max_steps + args.warmup_steps + 1,
        ignore_done=False,
    )
    episodes = []
    try:
        for episode_index in range(args.episodes_per_task):
            episodes.append(
                run_episode(
                    args, policy, env, init_states, instruction, prompt_template,
                    controller_config, task_dir, episode_index,
                )
            )
    finally:
        env.close()

    num_success = sum(int(item["success"]) for item in episodes)
    summary = {
        "task_id": task_id,
        "task_name": task.name,
        "instruction": instruction,
        "bddl_file": str(bddl_file),
        "num_episodes": len(episodes),
        "num_success": num_success,
        "success_rate": num_success / len(episodes) if episodes else 0.0,
        "episodes": episodes,
    }
    save_json(task_dir / "summary.json", summary)
    wprint(args, f"task={task_id} done: {num_success}/{len(episodes)} success")
    return summary


def try_claim_task(claims_dir: Path, task_id: int) -> bool:
    """Atomically claim a task so exactly one worker runs it to completion."""
    try:
        fd = os.open(claims_dir / f"task_{task_id:02d}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.close(fd)
    return True


def main_worker(args: argparse.Namespace) -> int:
    if args.episodes_per_task < 1 or args.max_steps < 1 or args.camera_size < 1:
        raise ValueError("--episodes-per-task, --max-steps and --camera-size must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative")
    validate_common_args(args)
    prompt_path = args.prompt_template.expanduser().resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    output_dir = args.output_dir.expanduser().resolve()
    configure_libero(args, output_dir)

    import robosuite as suite

    benchmark_instance, task_ids = load_benchmark_tasks(args)
    prompt_template = prompt_path.read_text(encoding="utf-8")
    controller_config = suite.load_controller_config(default_controller="OSC_POSE")

    claims_dir = output_dir / ".claims"
    policy_args = types.SimpleNamespace(
        model_python=args.model_python,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        constrained_decoding=args.constrained_decoding,
    )
    policy = None
    claimed = 0
    wprint(args, f"starting: device={args.device} render_gpu={args.render_gpu_device_id}")
    try:
        # Dynamic work stealing: claim one task, run all its episodes, claim the next.
        for task_id in task_ids:
            if not try_claim_task(claims_dir, task_id):
                continue
            if policy is None:
                policy = QwenPolicyWorker(policy_args)
                policy.start()
                wprint(args, "model worker ready")
            claimed += 1
            run_task(args, policy, benchmark_instance, task_id, prompt_template, controller_config, output_dir)
    finally:
        if policy is not None:
            policy.close()
    wprint(args, f"worker finished ({claimed} tasks claimed)")
    return 0


def device_render_id(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return 0


def main_suite(args: argparse.Namespace) -> int:
    if args.episodes_per_task < 1 or args.max_steps < 1 or args.camera_size < 1:
        raise ValueError("--episodes-per-task, --max-steps and --camera-size must be positive")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")
    validate_common_args(args)
    prompt_path = args.prompt_template.expanduser().resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    output_dir = (args.output_dir or (DEFAULT_OUTPUT_ROOT / timestamp())).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    configure_libero(args, output_dir)

    benchmark_instance, task_ids = load_benchmark_tasks(args)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices cannot be empty")
    num_workers = len(devices) * args.workers_per_gpu
    (output_dir / ".claims").mkdir()

    run_config = {
        "created_at_utc": utc_now(),
        "benchmark": args.benchmark,
        "task_ids": task_ids,
        "num_tasks": len(task_ids),
        "episodes_per_task": args.episodes_per_task,
        "start_state_index": args.start_state_index,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "camera_size": args.camera_size,
        "devices": devices,
        "workers_per_gpu": args.workers_per_gpu,
        "num_workers": num_workers,
        "model_path": str(args.model_path.expanduser().resolve()),
        "adapter_path": str(resolve_adapter_path(args.adapter_path)),
        "prompt_template": str(prompt_path),
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "max_new_tokens": args.max_new_tokens,
        "constrained_decoding": args.constrained_decoding,
        "parse_error_policy": args.parse_error_policy,
        "stop_on_task_state": args.stop_on_task_state,
        "seed": args.seed,
        "save_video": args.save_video,
        "save_input_images": args.save_input_images,
        "instruction_source": "bddl :language field (matches convert_libero_hdf5.py training data)",
    }
    save_json(output_dir / "run_config.json", run_config)
    print(f"Output: {output_dir}", flush=True)
    print(f"Benchmark: {args.benchmark}, {len(task_ids)} tasks x {args.episodes_per_task} episodes", flush=True)
    print(f"Workers: {num_workers} on devices {devices}", flush=True)

    script_path = Path(__file__).resolve()
    processes: list[tuple[int, subprocess.Popen]] = []
    suite_started = time.perf_counter()
    for worker_id in range(num_workers):
        device = devices[worker_id % len(devices)]
        command = [
            sys.executable,
            str(script_path),
            "--worker",
            "--worker-id", str(worker_id),
            "--model-path", str(args.model_path.expanduser().resolve()),
            "--adapter-path", str(args.adapter_path.expanduser().resolve()),
            "--model-python", str(args.model_python.expanduser().resolve()),
            "--libero-root", str(args.libero_root.expanduser().resolve()),
            "--prompt-template", str(prompt_path),
            "--output-dir", str(output_dir),
            "--benchmark", args.benchmark,
            "--episodes-per-task", str(args.episodes_per_task),
            "--start-state-index", str(args.start_state_index),
            "--max-steps", str(args.max_steps),
            "--warmup-steps", str(args.warmup_steps),
            "--camera-size", str(args.camera_size),
            "--device", device,
            "--render-gpu-device-id", str(device_render_id(device)),
            "--torch-dtype", args.torch_dtype,
            "--attn-implementation", args.attn_implementation,
            "--max-new-tokens", str(args.max_new_tokens),
            "--parse-error-policy", args.parse_error_policy,
            "--seed", str(args.seed),
            "--video-fps", str(args.video_fps),
        ]
        command.append(
            "--constrained-decoding"
            if args.constrained_decoding
            else "--no-constrained-decoding"
        )
        if args.stop_on_task_state:
            command.append("--stop-on-task-state")
        if args.save_video:
            command.append("--save-video")
        if args.save_input_images:
            command.append("--save-input-images")
        processes.append((worker_id, subprocess.Popen(command)))

    failures: dict[int, int] = {}
    for worker_id, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures[worker_id] = return_code
            print(f"[suite] worker {worker_id} exited with code {return_code}", flush=True)

    task_summaries = []
    for task_id in task_ids:
        task = benchmark_instance.get_task(task_id)
        summary_path = output_dir / f"task_{task_id:02d}_{task.name}" / "summary.json"
        if summary_path.is_file():
            task_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        else:
            task_summaries.append({
                "task_id": task_id,
                "task_name": task.name,
                "status": "missing_or_failed",
                "num_episodes": 0,
                "num_success": 0,
                "success_rate": 0.0,
            })

    completed = [item for item in task_summaries if item.get("num_episodes")]
    total_episodes = sum(item["num_episodes"] for item in task_summaries)
    total_success = sum(item["num_success"] for item in task_summaries)
    suite_summary = {
        "completed_at_utc": utc_now(),
        "elapsed_s": time.perf_counter() - suite_started,
        "benchmark": args.benchmark,
        "num_tasks": len(task_ids),
        "num_tasks_completed": len(completed),
        "episodes_per_task": args.episodes_per_task,
        "total_episodes": total_episodes,
        "total_success": total_success,
        "overall_success_rate": total_success / total_episodes if total_episodes else 0.0,
        "mean_task_success_rate": (
            sum(item["success_rate"] for item in completed) / len(completed) if completed else 0.0
        ),
        "worker_failures": failures,
        "tasks": task_summaries,
    }
    save_json(output_dir / "summary.json", suite_summary)

    print("\n=== Suite summary ===", flush=True)
    for item in task_summaries:
        rate = f"{item['success_rate']:.2f}" if item.get("num_episodes") else "n/a"
        print(f"  task {item['task_id']:02d} {item['task_name']}: {rate}", flush=True)
    print(f"Overall success rate: {suite_summary['overall_success_rate']:.3f} "
          f"({total_success}/{total_episodes})", flush=True)
    if failures:
        print(f"WARNING: {len(failures)} worker(s) failed: {failures}", flush=True)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    if args.model_worker:
        return run_model_worker(args)
    if args.worker:
        return main_worker(args)
    return main_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
