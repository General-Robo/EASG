"""Replay saved VLM actions, optionally continuing the online loop."""

import json
import shutil
from pathlib import Path

from run import (
    build_parser,
    configure_args,
    simulation_session,
    run_loop,
    finish_run,
    prompt_for,
    execute,
)
from observation import capture
import math
import numpy as np
from parse_util import (
    MotionCommand,
    TranslationCommand,
    RotationCommand,
    GripperCommand,
    motion_to_dict,
    motion_as_pipe,
)
from motion import TCP_FRAME, LEGACY_TCP_FRAME, legacy_to_paper_motion


def replay_parser():
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--resume-from",
        type=Path,
        required=True,
        help="Saved run to replay before continuing with fresh VLM actions",
    )
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Only replay and evaluate; no API call or key required",
    )
    return parser


def validate_source(parser, args):
    if args.resume_from and args.resume_from.resolve() == args.output.resolve():
        parser.error("--output must differ from --resume-from")
    prior_frame = TCP_FRAME
    if args.resume_from:
        if not args.resume_from.is_dir():
            parser.error("--resume-from must be an existing run directory")
        config_path = args.resume_from / "run_config.json"
        # Before task selection was added, all saved runs were pick_place.
        prior_config = (
            json.loads(config_path.read_text()) if config_path.exists() else {}
        )
        prior_task = prior_config.get("task", "pick_place")
        prior_frame = prior_config.get("motion_frame", LEGACY_TCP_FRAME)
        if prior_frame not in (TCP_FRAME, LEGACY_TCP_FRAME):
            parser.error("Unknown saved motion frame: " + prior_frame)
        if (
            "physics_config" in prior_config
            and prior_config["physics_config"] != args.scene_config.to_dict()
        ):
            parser.error(
                "--config must match the physical configuration of the saved run"
            )
        if prior_task != args.task:
            parser.error("--task must match the saved run")
        if args.task == "pick_place":
            if (
                prior_config.get("object") != args.object
                or prior_config.get("goal_asset") != args.target
            ):
                parser.error(
                    "Saved object/goal does not match this pick_place scene. Use the matching --object and --target; replay older scene formats with their frozen runtime."
                )
        saved_layout = prior_config.get("position_randomization", {"enabled": False})
        config = getattr(args, "position_config", None)
        requested_layout = config.to_dict() if config else {"enabled": False}
        if saved_layout != requested_layout:
            parser.error(
                "Position randomization must match the saved run; pass the same --position-randomization "
                "and --position-seed from run_config.json"
            )
    return prior_frame


def load_action_record(path):
    """Read our structured action log, not a model response; retain full precision."""

    def number(value):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Action log values must be finite numbers")
        return float(value)

    def vector(value):
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError("Action log vectors must contain three numbers")
        return np.array([number(item) for item in value])

    try:
        data = json.loads(Path(path).read_text())
        state = number(data["task_state"])
        translation, rotation = data["translation"], data["rotation"]
        gripper = number(data["gripper"]["state"])
        if state not in (0, 1) or not 0 <= gripper <= 1:
            raise ValueError("Invalid task/gripper state in action log")
        return MotionCommand(
            int(state),
            TranslationCommand(
                vector(translation["direction"]), number(translation["distance_cm"])
            ),
            RotationCommand(vector(rotation["axis"]), number(rotation["angle_deg"])),
            GripperCommand(state=gripper),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid action log {path}: {error}") from error


def replay_actions(args, scene, client, prior_frame):
    start_index = 1
    for prior in sorted(args.resume_from.glob("step_*")):
        if not (prior / "action.json").exists():
            break
        action = load_action_record(prior / "action.json")
        if prior_frame == LEGACY_TCP_FRAME:
            action = legacy_to_paper_motion(action)
        if action.task_state:
            break
        print("Replaying saved VLM action:", prior.name, flush=True)
        replay_folder = args.output / prior.name
        replay_info, replay_images = capture(scene, replay_folder)
        replay_prompt = prompt_for(scene, replay_info, args.instruction)
        (replay_folder / "prompt.txt").write_text(replay_prompt)
        # Preserve the original response, but store actions in the current frame.
        shutil.copy2(prior / "response.txt", replay_folder / "response.txt")
        (replay_folder / "action.json").write_text(
            json.dumps(motion_to_dict(action), indent=2)
        )
        (replay_folder / "replay_source.json").write_text(
            json.dumps(
                {
                    "source": str(prior),
                    "source_motion_frame": prior_frame,
                    "saved_action_frame": TCP_FRAME,
                    "response_is_original": True,
                }
            )
        )
        try:
            execute(scene, action)
        except (ValueError, RuntimeError) as error:
            if not (prior / "execution_error.txt").exists():
                raise
            (replay_folder / "execution_error.txt").write_text(str(error))
            print("Reproduced recorded partial-execution feedback.", flush=True)
        (replay_folder / "evaluation.json").write_text(
            json.dumps(scene.evaluate(), indent=2)
        )
        if client is not None:
            history_response = (
                motion_as_pipe(action)
                if prior_frame == LEGACY_TCP_FRAME
                else (prior / "response.txt").read_text()
            )
            context_response = replay_folder / "context_response.txt"
            context_response.write_text(history_response)
            client.remember_turn(
                replay_prompt,
                replay_images,
                history_response,
                context_path=replay_folder / "context.json",
                step_index=int(prior.name.split("_")[-1]),
                response_path=context_response,
            )
        start_index += 1
    (args.output / "resume_source.json").write_text(
        json.dumps({"source": str(args.resume_from), "replayed_steps": start_index - 1})
    )
    return start_index


def main():
    parser = replay_parser()
    args = parser.parse_args()
    templates = configure_args(parser, args)
    prior_frame = validate_source(parser, args)
    metadata = {
        "resume_from": str(args.resume_from),
        "mode": "replay"
        if args.replay_only
        else ("observation" if args.observe_only else "closed_loop"),
    }
    with simulation_session(
        args, templates, use_client=not args.replay_only, metadata=metadata
    ) as (scene, client):
        start_index = replay_actions(args, scene, client, prior_frame)
        if args.replay_only:
            scene.step_seconds(args.scene_config.completion_settle_s)
            capture(scene, args.output / "final")
        else:
            run_loop(args, scene, client, start_index=start_index)
        finish_run(args, scene)


if __name__ == "__main__":
    main()
