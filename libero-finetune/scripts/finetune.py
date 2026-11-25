#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoProcessor, Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
ASSISTANT_START = "<|im_start|>assistant\n"
MESSAGE_END = "<|im_end|>"
MM_CAT_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "second_per_grid_ts",
)
# Every signed scalar is encoded as ``sign:magnitude``.  The sign bit is 0
# for non-negative values and 1 for negative values, e.g. 0:0.39 and 1:0.26.
ENCODED_DECIMAL_RE = re.compile(
    r"(?P<sign>[01]):(?P<magnitude>\d+\.\d{2})"
)
LOSS_COMPONENT_NAMES = (
    "ce_raw",
    "ce_weighted",
    "gripper_ce_raw",
    "gripper_ce_weighted",
    "sign_format_raw",
    "sign_format_weighted",
    "digit_format_raw",
    "digit_format_weighted",
    "total",
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("LoRA SFT for Qwen3.5 vision-language models.")
    parser.add_argument("--model-path", default=str(Path(__file__).resolve().parents[1] / "models/Qwen3.5-0.8B"))
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output-dir", default="./outputs/libero_10")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--init-adapter-from", help="Initialize matching LoRA tensors, leaving newly added modules initialized.")
    parser.add_argument("--adapter-path", default=None, help="Continue training an existing PEFT adapter.")

    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj,attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2,merger.linear_fc1,merger.linear_fc2",
    )
    parser.add_argument(
        "--ce-weight",
        type=float,
        default=1.0,
        help="Weight for standard causal LM cross-entropy on all assistant tokens, including digits.",
    )
    parser.add_argument(
        "--gripper-ce-weight",
        type=float,
        default=5.0,
        help=(
            "Weight for standard full-vocabulary cross-entropy on gripper "
            "field tokens, added to ordinary assistant CE."
        ),
    )
    parser.add_argument(
        "--sign-format-weight",
        type=float,
        default=1.0,
        help="Weight for -log(P(0)+P(1)) at every encoded sign-bit position.",
    )
    parser.add_argument(
        "--digit-format-weight",
        type=float,
        default=1.0,
        help="Weight for -log(sum P(0..9)) at every magnitude-digit position.",
    )
    parser.add_argument("--train-on-source", action="store_true")
    parser.add_argument("--enable-thinking-template", action="store_true")
    parser.add_argument(
        "--answer-only-logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Only run the LM head on the suffix needed to predict non-ignored labels. "
            "This is mathematically equivalent to full-sequence logits when source labels are -100."
        ),
    )
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument(
        "--fail-on-nonfinite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Abort before backward when the scalar training loss is NaN or infinite.",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def masked_token_group_cross_entropy(
    logits: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    """Compute token-group CE after selecting only supervised positions.

    At each selected position this is exactly -log(sum(p_allowed)) under the
    full vocabulary softmax. Selecting positions before the full-vocabulary
    logsumexp avoids reducing that partition for unrelated answer tokens.
    """
    if not mask.any():
        return None

    selected_logits = logits[mask].float()
    selected_allowed_logits = selected_logits.index_select(
        -1,
        allowed_token_ids.to(logits.device),
    )
    values = (
        torch.logsumexp(selected_logits, dim=-1)
        - torch.logsumexp(selected_allowed_logits, dim=-1)
    )
    selected_weights = weights[mask].to(values.dtype)
    return (
        (values * selected_weights).sum()
        / selected_weights.sum().clamp_min(1e-6)
    )


def import_qwen_model_class():
    import transformers

    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        model_cls = getattr(transformers, name, None)
        if model_cls is not None:
            return model_cls
    raise ImportError("Could not find a usable AutoModel class in transformers.")


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and (item.get("type") == "text" or "text" in item):
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    raise ValueError(f"Unsupported assistant content type: {type(content)!r}")


def normalize_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept the image/text messages emitted by the LIBERO converter."""
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a nonempty messages list from the LIBERO converter.")
    for message in messages:
        if message.get("role") not in ("user", "assistant", "system"):
            raise ValueError("Unsupported message role.")
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or any(item.get("type") not in ("text", "image") for item in content):
            raise ValueError("Only image/text message content is supported.")
    return messages


def is_relative_media_path(value: str) -> bool:
    lowered = value.lower()
    return not os.path.isabs(value) and "://" not in lowered and not lowered.startswith("data:")


def resolve_media_paths(messages: list[dict[str, Any]], image_root: Path) -> list[dict[str, Any]]:
    resolved = []
    for message in messages:
        content = message["content"]
        if not isinstance(content, list):
            resolved.append(message)
            continue
        items = []
        for item in content:
            new_item = dict(item)
            for key in ("image", "video", "path"):
                value = new_item.get(key)
                if isinstance(value, str) and is_relative_media_path(value):
                    new_item[key] = str((image_root / value).resolve())
            items.append(new_item)
        resolved.append({"role": message["role"], "content": items})
    return resolved


def find_subsequence(values: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern:
        return -1
    for index in range(start, len(values) - len(pattern) + 1):
        if values[index : index + len(pattern)] == pattern:
            return index
    return -1


def assistant_only_labels(tokenizer: Any, input_ids: list[int]) -> list[int]:
    labels = [IGNORE_INDEX] * len(input_ids)
    start_ids = tokenizer(ASSISTANT_START, add_special_tokens=False)["input_ids"]
    end_ids = tokenizer(MESSAGE_END, add_special_tokens=False)["input_ids"]
    cursor = 0

    while True:
        start = find_subsequence(input_ids, start_ids, cursor)
        if start < 0:
            break
        label_start = start + len(start_ids)
        end = find_subsequence(input_ids, end_ids, label_start)
        label_end = len(input_ids) if end < 0 else end + len(end_ids)
        labels[label_start:label_end] = input_ids[label_start:label_end]
        cursor = label_end
    return labels


def answer_logit_start(labels: torch.Tensor) -> int:
    """Return the first sequence position whose logits can affect causal LM loss.

    A target at position ``t`` is predicted by logits at position ``t - 1``.
    Positions whose target is ``IGNORE_INDEX`` can be omitted from the LM head.
    """
    if labels.ndim != 2:
        raise ValueError(f"Expected rank-2 labels, got shape {tuple(labels.shape)}")
    if labels.shape[1] < 2:
        raise ValueError("Causal LM training requires a sequence of at least two tokens.")
    valid_prediction_positions = labels[:, 1:].ne(IGNORE_INDEX).nonzero(as_tuple=False)
    if valid_prediction_positions.numel() == 0:
        raise ValueError("Batch contains no causal LM targets after shifting labels.")
    # Column indices in labels[:, 1:] are already the corresponding logit positions.
    return int(valid_prediction_positions[:, 1].min().item())


def causal_target_window(
    values: torch.Tensor,
    logit_start: int,
    prediction_length: int,
) -> torch.Tensor:
    """Select target-aligned labels or annotations for suffix logits."""
    target_start = logit_start + 1
    target_end = target_start + prediction_length
    if target_end > values.shape[1]:
        raise ValueError(
            f"Target window [{target_start}:{target_end}] exceeds shape {tuple(values.shape)}."
        )
    return values[:, target_start:target_end]


def causal_prediction_window(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logit_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align suffix-only logits with their next-token labels."""
    expected_logit_length = labels.shape[1] - logit_start
    if logits.shape[:2] != (labels.shape[0], expected_logit_length):
        raise ValueError(
            "Suffix logits do not match labels: "
            f"got logits {tuple(logits.shape[:2])}, labels {tuple(labels.shape)}, "
            f"logit_start={logit_start}."
        )
    # The final sequence position has no next token in this training example.
    prediction_logits = logits[:, :-1, :]
    shifted_labels = causal_target_window(
        labels,
        logit_start,
        prediction_logits.shape[1],
    )
    return prediction_logits, shifted_labels


def action_character_annotations(text: str) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    sign_format, digit_format, gripper_ce = {}, {}, {}
    step_start = 0
    for step in text.split(";"):
        fields = step.split("|")
        if len(fields) != 7:
            raise ValueError("Every assistant action step must contain seven pipe-separated fields.")
        field_start = step_start
        for index, field in enumerate(fields):
            if index in (3, 6):
                gripper_ce.update({i: 1.0 for i in range(field_start, field_start + len(field))})
            field_start += len(field) + 1
        step_start += len(step) + 1
    for match in ENCODED_DECIMAL_RE.finditer(text):
        sign_format[match.start("sign")] = 1.0
        digit_format.update({match.start("magnitude") + i: 1.0
                             for i, c in enumerate(match.group("magnitude")) if c.isdigit()})
    return sign_format, digit_format, gripper_ce


def action_token_annotations(
    tokenizer: Any,
    input_ids: list[int],
    messages: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float]]:
    sign_format_weights = [0.0] * len(input_ids)
    digit_format_weights = [0.0] * len(input_ids)
    gripper_ce_weights = [0.0] * len(input_ids)
    start_ids = tokenizer(ASSISTANT_START, add_special_tokens=False)["input_ids"]
    cursor = 0

    for message in messages:
        if message["role"] != "assistant":
            continue
        answer_text = content_to_text(message["content"])
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        if not answer_ids:
            continue

        start = find_subsequence(input_ids, start_ids, cursor)
        search_from = cursor if start < 0 else start + len(start_ids)
        answer_start = find_subsequence(input_ids, answer_ids, search_from)
        if answer_start < 0:
            cursor = search_from
            continue

        char_sign_format, char_digit_format, char_gripper_ce = action_character_annotations(answer_text)
        tokenized = tokenizer(
            answer_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        for token_index, (char_start, char_end) in enumerate(tokenized["offset_mapping"]):
            input_index = answer_start + token_index
            if input_index >= len(input_ids):
                continue
            if any(
                char_gripper_ce.get(char_index, 0.0) > 0
                for char_index in range(char_start, char_end)
            ):
                gripper_ce_weights[input_index] = 1.0
            if char_end - char_start != 1:
                continue

            sign_format_weights[input_index] = char_sign_format.get(char_start, 0.0)
            digit_format_weights[input_index] = char_digit_format.get(char_start, 0.0)
        cursor = answer_start + len(answer_ids)

    return sign_format_weights, digit_format_weights, gripper_ce_weights


def squeeze(feature: dict[str, Any]) -> dict[str, Any]:
    squeezed = {}
    for key, value in feature.items():
        if key in {"input_ids", "attention_mask", "mm_token_type_ids"} and isinstance(value, list):
            squeezed[key] = value[0] if value and isinstance(value[0], list) else value
        else:
            squeezed[key] = value
    return squeezed


def mask_visual_labels(labels: list[int], input_ids: list[int], feature: dict[str, Any], processor: Any) -> list[int]:
    visual_ids = {
        getattr(processor, "image_token_id", None),
        getattr(processor, "video_token_id", None),
        getattr(processor, "vision_start_token_id", None),
        getattr(processor, "vision_end_token_id", None),
    }
    visual_ids.discard(None)
    mm_types = feature.get("mm_token_type_ids")

    masked = list(labels)
    for index, token_id in enumerate(input_ids):
        if token_id in visual_ids:
            masked[index] = IGNORE_INDEX
        if mm_types is not None and index < len(mm_types) and mm_types[index] != 0:
            masked[index] = IGNORE_INDEX
    return masked


class JsonlSftDataset(Dataset):
    def __init__(
        self,
        path: str,
        processor: Any,
        image_root: Path,
        max_seq_length: int,
        train_on_source: bool,
        enable_thinking: bool,
    ) -> None:
        self.path = path
        self.processor = processor
        self.image_root = image_root
        self.max_seq_length = max_seq_length
        self.train_on_source = train_on_source
        self.enable_thinking = enable_thinking
        with open(path, "r", encoding="utf-8") as f:
            self.rows = [(line_no, json.loads(line)) for line_no, line in enumerate(f, 1) if line.strip()]
        if not self.rows:
            raise ValueError(f"No training examples found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def get_row(self, index: int) -> tuple[int, dict[str, Any]]:
        return self.rows[index]

    def resolve_example_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        return resolve_media_paths(normalize_messages(example), self.image_root)

    def __getitem__(self, index: int) -> dict[str, Any]:
        line_no, example = self.get_row(index)
        try:
            messages = self.resolve_example_messages(example)
            feature = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_tensors=None,
                processor_kwargs={"truncation": True, "max_length": self.max_seq_length},
                enable_thinking=self.enable_thinking,
            )
        except Exception as exc:
            raise ValueError(f"Failed to preprocess {self.path}:{line_no}: {exc}") from exc

        feature = squeeze(dict(feature))
        input_ids = list(feature["input_ids"])
        labels = list(input_ids) if self.train_on_source else assistant_only_labels(self.processor.tokenizer, input_ids)
        labels = mask_visual_labels(labels, input_ids, feature, self.processor)
        if not any(label != IGNORE_INDEX for label in labels):
            raise ValueError(f"Example produced no trainable labels: {self.path}:{line_no}")
        feature["labels"] = labels
        (
            feature["sign_format_weights"],
            feature["digit_format_weights"],
            feature["gripper_ce_weights"],
        ) = action_token_annotations(
            self.processor.tokenizer,
            input_ids,
            messages,
        )
        return feature


@dataclass
class SftCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features = [dict(feature) for feature in features]
        labels = [feature.pop("labels") for feature in features]
        extra_specs = {
            "sign_format_weights": (0.0, torch.float32),
            "digit_format_weights": (0.0, torch.float32),
            "gripper_ce_weights": (0.0, torch.float32),
        }
        extras = {
            key: [feature.pop(key, None) for feature in features]
            for key in extra_specs
        }
        text = [
            {
                key: feature[key]
                for key in ("input_ids", "attention_mask")
                if key in feature
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(
            text,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]

        batch["labels"] = torch.tensor(
            [label + [IGNORE_INDEX] * (max_len - len(label)) for label in labels],
            dtype=torch.long,
        )
        # Keep this as a Python integer so Trainer does not copy it to CUDA and
        # compute_loss does not need a device-synchronizing .item() every step.
        batch["answer_logit_start"] = answer_logit_start(batch["labels"])
        for key, (pad_value, dtype) in extra_specs.items():
            values = extras[key]
            batch[key] = torch.tensor(
                [
                    (
                        list(value)
                        if value is not None
                        else [pad_value] * len(feature["input_ids"])
                    )
                    + [pad_value]
                    * (
                        max_len
                        - len(
                            value
                            if value is not None
                            else feature["input_ids"]
                        )
                    )
                    for feature, value in zip(features, values)
                ],
                dtype=dtype,
            )

        if any("mm_token_type_ids" in feature for feature in features):
            padded_mm_token_type_ids = []
            for feature in features:
                values = list(feature.get("mm_token_type_ids", [0] * len(feature["input_ids"])))
                padded_mm_token_type_ids.append(values + [0] * (max_len - len(values)))
            batch["mm_token_type_ids"] = torch.tensor(padded_mm_token_type_ids, dtype=torch.long)

        for key in MM_CAT_KEYS:
            values = [feature[key] for feature in features if key in feature and feature[key] is not None]
            if values:
                tensors = []
                for value in values:
                    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
                    tensors.append(tensor if tensor.dim() > 0 else tensor.unsqueeze(0))
                batch[key] = torch.cat(tensors, dim=0)
        return batch


def trainer_processing_kw(processor: Any) -> dict[str, Any]:
    key = "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
    return {key: processor}


def make_training_args(**kwargs: Any) -> TrainingArguments:
    supported = inspect.signature(TrainingArguments.__init__).parameters
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in supported})


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class FinetuneMetadataCallback(TrainerCallback):
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            save_json(
                checkpoint_dir / "finetune_metadata.json",
                {
                    **self.metadata,
                    "status": "checkpoint",
                    "global_step": state.global_step,
                },
            )
        return control


class ActionTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        digit_token_ids: list[int],
        ce_weight: float,
        gripper_ce_weight: float,
        sign_format_weight: float,
        digit_format_weight: float,
        answer_only_logits: bool = True,
        fail_on_nonfinite: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.digit_token_ids = torch.tensor(digit_token_ids, dtype=torch.long)
        self.ce_weight = ce_weight
        self.gripper_ce_weight = gripper_ce_weight
        self.sign_format_weight = sign_format_weight
        self.digit_format_weight = digit_format_weight
        self.answer_only_logits = answer_only_logits
        self.fail_on_nonfinite = fail_on_nonfinite
        self._loss_component_sums: dict[str, dict[str, torch.Tensor]] = {
            "train": {},
            "eval": {},
        }
        self._loss_component_counts = {"train": 0, "eval": 0}


    def _record_loss_components(
        self,
        model: torch.nn.Module,
        components: dict[str, torch.Tensor],
    ) -> None:
        phase = "train" if model.training else "eval"
        sums = self._loss_component_sums[phase]
        for name in LOSS_COMPONENT_NAMES:
            value = components[name].detach().float()
            sums[name] = value if name not in sums else sums[name] + value
        self._loss_component_counts[phase] += 1

    def _flush_loss_components(self, phase: str) -> dict[str, float]:
        count = self._loss_component_counts[phase]
        if count == 0:
            return {}

        sums = self._loss_component_sums[phase]
        values = torch.stack(
            [sums[name] for name in LOSS_COMPONENT_NAMES]
            + [torch.tensor(float(count), device=next(iter(sums.values())).device)]
        )
        values = self.accelerator.reduce(values, reduction="sum")
        averages = values[:-1] / values[-1].clamp_min(1.0)
        prefix = "loss_component" if phase == "train" else "eval_loss_component"
        result = {
            f"{prefix}/{name}": value.item()
            for name, value in zip(LOSS_COMPONENT_NAMES, averages)
        }
        self._loss_component_sums[phase] = {}
        self._loss_component_counts[phase] = 0
        return result

    def log(
        self,
        logs: dict[str, float],
        start_time: float | None = None,
    ) -> None:
        phase = "eval" if any(key.startswith("eval_") for key in logs) else "train"
        component_logs = self._flush_loss_components(phase)
        logs.update(component_logs)
        super().log(logs, start_time)
        if component_logs and self.is_world_process_zero():
            log_path = Path(self.args.output_dir) / "loss_components.jsonl"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.state.log_history[-1]) + "\n")

    def _check_finite_loss(self, loss: torch.Tensor) -> None:
        if getattr(self, "fail_on_nonfinite", False) and not torch.isfinite(loss.detach()).item():
            raise FloatingPointError(
                "Non-finite training loss detected before backward at "
                f"global_step={self.state.global_step}. Aborting to protect checkpoints."
            )

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        sign_format_weights = inputs.pop("sign_format_weights", None)
        digit_format_weights = inputs.pop("digit_format_weights", None)
        gripper_ce_weights = inputs.pop("gripper_ce_weights", None)
        batch_logit_start = inputs.pop("answer_logit_start", None)
        labels = inputs.get("labels")

        if self.answer_only_logits and labels is not None:
            logit_start = (
                int(batch_logit_start)
                if batch_logit_start is not None
                else answer_logit_start(labels)
            )
            model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            outputs = model(
                **model_inputs,
                logits_to_keep=labels.shape[1] - logit_start,
            )
            logits, shifted_labels = causal_prediction_window(
                outputs.logits,
                labels,
                logit_start,
            )
            ce_loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                shifted_labels.reshape(-1).to(logits.device),
                ignore_index=IGNORE_INDEX,
            )
        else:
            logit_start = 0
            outputs = model(**inputs)
            ce_loss = outputs.loss
            if labels is not None and outputs.logits is not None:
                logits, shifted_labels = causal_prediction_window(
                    outputs.logits,
                    labels,
                    logit_start,
                )

        loss = self.ce_weight * ce_loss
        zero = ce_loss.detach().new_zeros((), dtype=torch.float32)
        loss_components = {name: zero for name in LOSS_COMPONENT_NAMES}
        loss_components["ce_raw"] = ce_loss
        loss_components["ce_weighted"] = loss

        needs_auxiliary_loss = (
            self.gripper_ce_weight > 0
            or self.sign_format_weight > 0
            or self.digit_format_weight > 0
        )
        if not needs_auxiliary_loss or labels is None or outputs.logits is None:
            loss_components["total"] = loss
            self._check_finite_loss(loss)
            self._record_loss_components(model, loss_components)
            return (loss, outputs) if return_outputs else loss

        valid_labels = shifted_labels.ne(IGNORE_INDEX)

        if self.gripper_ce_weight > 0 and gripper_ce_weights is not None:
            shifted_weights = causal_target_window(
                gripper_ce_weights, logit_start, logits.shape[1]
            ).to(logits.device)
            gripper_mask = shifted_weights.gt(0) & valid_labels
            if gripper_mask.any():
                gripper_ce_loss = F.cross_entropy(
                    logits[gripper_mask].float(),
                    shifted_labels[gripper_mask],
                )
                loss = loss + self.gripper_ce_weight * gripper_ce_loss
                loss_components["gripper_ce_raw"] = gripper_ce_loss
                loss_components["gripper_ce_weighted"] = (
                    self.gripper_ce_weight * gripper_ce_loss
                )

        if self.sign_format_weight > 0 and sign_format_weights is not None:
            shifted_weights = causal_target_window(
                sign_format_weights, logit_start, logits.shape[1]
            ).to(logits.device)
            sign_mask = shifted_weights.gt(0) & valid_labels
            sign_format_loss = masked_token_group_cross_entropy(
                logits,
                self.digit_token_ids[:2],
                shifted_weights,
                sign_mask,
            )
            if sign_format_loss is not None:
                loss = loss + self.sign_format_weight * sign_format_loss
                loss_components["sign_format_raw"] = sign_format_loss
                loss_components["sign_format_weighted"] = (
                    self.sign_format_weight * sign_format_loss
                )

        if self.digit_format_weight > 0 and digit_format_weights is not None:
            shifted_weights = causal_target_window(
                digit_format_weights, logit_start, logits.shape[1]
            ).to(logits.device)
            digit_mask = shifted_weights.gt(0) & valid_labels
            digit_format_loss = masked_token_group_cross_entropy(
                logits,
                self.digit_token_ids,
                shifted_weights,
                digit_mask,
            )
            if digit_format_loss is not None:
                loss = loss + self.digit_format_weight * digit_format_loss
                loss_components["digit_format_raw"] = digit_format_loss
                loss_components["digit_format_weighted"] = (
                    self.digit_format_weight * digit_format_loss
                )

        loss_components["total"] = loss
        self._check_finite_loss(loss)
        self._record_loss_components(model, loss_components)
        return (loss, outputs) if return_outputs else loss


def initialize_adapter(model: Any, path: str) -> None:
    """Warm-start an expanded LoRA adapter without restoring optimizer/step state."""
    from peft.utils.save_and_load import load_peft_weights, get_peft_model_state_dict, set_peft_model_state_dict
    source = load_peft_weights(path, device="cpu")
    target = get_peft_model_state_dict(model)
    if not source or any(k not in target or target[k].shape != v.shape for k, v in source.items()):
        raise ValueError("Initialization adapter contains incompatible tensor names or shapes.")
    # PEFT's default LoRA initialization makes the extra modules no-ops initially.
    set_peft_model_state_dict(model, source)
    print(f"Initialized {len(source)} adapter tensors; {len(target) - len(source)} new tensors.")


def main() -> None:
    args = parse_args()
    if min(
        args.ce_weight,
        args.gripper_ce_weight,
        args.sign_format_weight,
        args.digit_format_weight,
    ) < 0:
        raise ValueError("Loss weights must be non-negative.")
    if args.init_adapter_from and (args.adapter_path or args.resume_from_checkpoint):
        raise ValueError("--init-adapter-from cannot be combined with adapter loading or full checkpoint resume.")
    set_seed(args.seed)
    random.seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code, use_fast=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_path = Path(args.train_file)
    image_root = Path(args.image_root) if args.image_root else train_path.parent
    train_dataset = JsonlSftDataset(
        args.train_file,
        processor,
        image_root,
        args.max_seq_length,
        args.train_on_source,
        args.enable_thinking_template,
    )
    eval_dataset = None
    if args.eval_file:
        eval_dataset = JsonlSftDataset(
            args.eval_file,
            processor,
            image_root,
            args.max_seq_length,
            args.train_on_source,
            args.enable_thinking_template,
        )

    digit_token_ids = []
    for digit in range(10):
        ids = tokenizer(str(digit), add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"Digit {digit!r} is not a single token: {ids}")
        digit_token_ids.append(ids[0])

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float32,
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": args.attn_implementation,
    }
    model = import_qwen_model_class().from_pretrained(args.model_path, **model_kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=split_csv(args.target_modules),
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    if args.init_adapter_from:
        initialize_adapter(model, args.init_adapter_from)
    model.print_trainable_parameters()

    training_args = make_training_args(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=eval_dataset is not None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=False,
        optim=args.optim,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = ActionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SftCollator(
            tokenizer,
        ),
        digit_token_ids=digit_token_ids,
        ce_weight=args.ce_weight,
        gripper_ce_weight=args.gripper_ce_weight,
        sign_format_weight=args.sign_format_weight,
        digit_format_weight=args.digit_format_weight,
        answer_only_logits=args.answer_only_logits,
        fail_on_nonfinite=args.fail_on_nonfinite,
        **trainer_processing_kw(processor),
    )
    metadata = {
        "base_model": str(Path(args.model_path).resolve()),
        "adapter_path": str(Path(args.adapter_path).resolve()) if args.adapter_path else None,
        "train_file": str(Path(args.train_file).resolve()),
        "eval_file": str(Path(args.eval_file).resolve()) if args.eval_file else None,
        "image_root": str(image_root.resolve()),
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": split_csv(args.target_modules),
        "ce_weight": args.ce_weight,
        "ce_scope": "all assistant tokens",
        "answer_only_logits": args.answer_only_logits,
        "fail_on_nonfinite": args.fail_on_nonfinite,
        "gripper_ce_weight": args.gripper_ce_weight,
        "gripper_ce_scope": "standard full-vocabulary CE on both gripper fields",
        "sign_format_weight": args.sign_format_weight,
        "digit_format_weight": args.digit_format_weight,
        "encoded_decimal_format": "sign:magnitude",
        "encoded_decimal_sign_bits": {
            "0": "non-negative",
            "1": "negative",
        },
        "script_args": vars(args),
        "training_arguments": training_args.to_dict(),
        "loss_component_log": {
            "path": "loss_components.jsonl",
            "frequency": "every Trainer logging event",
            "aggregation": "mean across microbatches and distributed processes",
            "names": list(LOSS_COMPONENT_NAMES),
            "weighted_suffix": "actual contribution added to total loss",
        },
    }
    trainer.add_callback(FinetuneMetadataCallback(metadata))
    if trainer.is_world_process_zero():
        save_json(
            Path(args.output_dir) / "finetune_metadata.json",
            {**metadata, "status": "running", "global_step": 0},
        )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output_dir)
        save_json(
            Path(args.output_dir) / "finetune_metadata.json",
            {
                **metadata,
                "status": "completed",
                "global_step": trainer.state.global_step,
            },
        )


if __name__ == "__main__":
    main()
