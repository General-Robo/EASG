"""Closed-loop VLMasAM on EASG. No scripted grasp trajectory or object-pose oracle."""

import json
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from settings import PROJECT_ROOT, add_position_arguments, position_config_from_args
from simulation import arguments, start_app
from settings import config_from_args
from client import (
    CHAT_CONNECTION_MODES,
    VisionClient,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_IMAGE_ENCODING,
    DEFAULT_RECENT_TURNS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
)
from parse_util import parse_vlm_motion, motion_to_dict
from motion import (
    TCP_FRAME,
    build_target_pose,
    validate_motion_limits,
    paper_to_hand_motion,
)
from observation import capture
from process_prompt import PROMPT_PATH, load_prompt
from construct_scene.base import TCP_OFFSET
from construct_scene.assets import OBJECTS
from construct_scene.pick_place import TARGET_NAMES, make_target


def prompt_for(scene, info, instruction):
    return scene.prompt_template.render(info, instruction)


def execute(scene, motion):
    from isaacsim.core.utils.rotations import rot_matrix_to_quat

    config = scene.config
    validate_motion_limits(motion, config.max_translation_cm, config.max_rotation_deg)
    target = build_target_pose(
        scene.hand_matrix(), paper_to_hand_motion(motion), tcp_z_offset_m=TCP_OFFSET
    )
    width = motion.gripper.state * scene.config.gripper_max_width_m
    # Uniform action order: move/rotate first, then set the gripper opening.
    if motion.translation.distance_cm or motion.rotation.angle_deg:
        scene.controller.move_pose(
            target[:3, 3],
            rot_matrix_to_quat(target[:3, :3]),
            duration=config.action_duration_s,
        )
    scene.controller.gripper(width)
    scene.step_seconds(config.action_settle_s)
    return target


def build_parser():
    parser = arguments(__doc__)
    parser.set_defaults(output=None)
    parser.add_argument(
        "--task", choices=["pick_place", "close_laptop"], default="pick_place"
    )
    parser.add_argument(
        "--object",
        choices=tuple(OBJECTS),
        default="banana",
        help="Object for pick_place (default banana)",
    )
    parser.add_argument(
        "--target",
        choices=TARGET_NAMES,
        default=None,
        help="Placement target; default plate_white; pick_place only",
    )
    parser.add_argument(
        "--instruction",
        help="Override the language instruction; evaluation remains task-specific",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        help="Optional fixed policy sent once before conversation history",
    )
    parser.add_argument(
        "--label-images",
        action="store_true",
        help="Attach source step and camera names to each image without changing pixels",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=PROMPT_PATH,
        help="Prompt template to snapshot for this run",
    )
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=DEFAULT_REASONING_EFFORT,
        help="Default xhigh; must be supported by the model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Response budget including reasoning; default 4096",
    )
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument(
        "--keys-file",
        type=Path,
        help="Optional private JSON list containing exactly one API key, never copied into output",
    )
    parser.add_argument(
        "--image-encoding",
        choices=["png", "jpeg"],
        default=DEFAULT_IMAGE_ENCODING,
        help="JPEG quality 90 at original resolution avoids the provider image-size cap for full history",
    )
    parser.add_argument("--max-task-steps", type=int, default=65)
    parser.add_argument(
        "--chat-connection-mode",
        choices=CHAT_CONNECTION_MODES,
        default="continuous",
        help="separate: current step only; continuous: all history; first-recent: first plus latest N turns, with images",
    )
    parser.add_argument(
        "--chat-recent-turns",
        type=int,
        default=DEFAULT_RECENT_TURNS,
        help="Recent completed turns for first-recent (default 2; at least 1)",
    )
    parser.add_argument("--observe-only", action="store_true")
    add_position_arguments(parser)
    return parser


def configure_args(parser, args):
    if args.output is None:
        label = args.object if args.task == "pick_place" else "laptop"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        args.output = PROJECT_ROOT / "output" / f"{args.task}-{label}-{stamp}"
    if args.task == "close_laptop" and args.target is not None:
        parser.error("--target is only supported by pick_place")
    if args.task == "pick_place" and args.target is None:
        args.target = "plate_white"
    args.scene_config = config_from_args(parser, args)
    args.position_config = position_config_from_args(parser, args)
    if args.position_config is not None and args.task == "close_laptop":
        parser.error(
            "position randomization requires a pick/place object and goal; close_laptop is not supported"
        )
    try:
        template = load_prompt(
            args.prompt_template, args.scene_config, system_path=args.system_prompt
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.label_images = args.label_images or template.system is not None
    if args.chat_recent_turns < 1:
        parser.error("--chat-recent-turns must be at least 1")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.instruction is None:
        args.instruction = (
            "Close the laptop lid completely, then withdraw the robot hand."
            if args.task == "close_laptop"
            else make_target(args.target).describe_instruction(args.object)
        )
    return template


def create_task_scene(app, assets, args, **scene_options):
    if args.task == "close_laptop":
        from construct_scene.laptop import LaptopScene

        return LaptopScene(app, assets, **scene_options)
    from construct_scene.pick_place import PickPlaceScene, RigidGraspObject

    return PickPlaceScene(
        app,
        assets,
        grasp_object=RigidGraspObject(OBJECTS[args.object]),
        placement_target=make_target(args.target),
        **scene_options,
    )


@contextmanager
def simulation_session(args, template, *, use_client=True, metadata=None):
    """Own simulation lifetime and shared setup for task entry points."""
    client = (
        None
        if not use_client or args.observe_only
        else VisionClient(
            args.model,
            args.base_url,
            args.keys_file,
            max_tokens=args.max_tokens,
            chat_connection_mode=args.chat_connection_mode,
            chat_recent_turns=args.chat_recent_turns,
            image_encoding=args.image_encoding,
            reasoning_effort=args.reasoning_effort,
            system_prompt=template.system,
            label_images=args.label_images,
        )
    )
    app = start_app(args)
    scene = None
    try:
        scene_options = {"config": args.scene_config}
        if args.position_config is not None:
            scene_options["position_randomization"] = args.position_config
        scene = create_task_scene(app, args.assets, args, **scene_options)
        scene.prepare(args.output)
        (args.output / "physics_config.json").write_text(
            json.dumps(args.scene_config.to_dict(), indent=2)
        )
        scene.prompt_template = template
        (args.output / "prompt_template.txt").write_text(template.observation)
        (args.output / "prompt_source.txt").write_text(template.source)
        if template.system is not None:
            (args.output / "system_prompt.txt").write_text(template.system)
        (args.output / "run_config.json").write_text(
            json.dumps(
                {
                    "system_prompt_file": str(args.system_prompt.resolve())
                    if args.system_prompt
                    else (
                        str(args.prompt_template.resolve()) if template.system else None
                    ),
                    "prompt_source_snapshot": str(
                        (args.output / "prompt_source.txt").resolve()
                    ),
                    "system_prompt_snapshot": str(
                        (args.output / "system_prompt.txt").resolve()
                    )
                    if template.system
                    else None,
                    "image_source_labels": args.label_images,
                    "prompt_file": str(args.prompt_template.resolve()),
                    "prompt_snapshot": str(
                        (args.output / "prompt_template.txt").resolve()
                    ),
                    "motion_format": "pipe",
                    "motion_frame": TCP_FRAME,
                    "config": args.config,
                    "physics_config": args.scene_config.to_dict(),
                    "position_randomization": args.position_config.to_dict()
                    if args.position_config
                    else {"enabled": False},
                    "evaluation_version": (
                        "pick_place_no_retreat_v1" if args.target == "blue_pad"
                        else "pick_place_plate_vertices75_no_margin_v4"
                    )
                    if args.task == "pick_place"
                    else "close_laptop_v1",
                    "schema_version": 3,
                    "object": args.object if args.task == "pick_place" else None,
                    "goal_asset": args.target if args.task == "pick_place" else None,
                    "temperature": DEFAULT_TEMPERATURE,
                    "task": args.task,
                    "model": args.model,
                    "base_url": args.base_url,
                    "instruction": args.instruction,
                    "gpu": args.gpu,
                    "runtime": scene.runtime,
                    "assets": str(scene.assets),
                    "mode": "observation" if args.observe_only else "closed_loop",
                    "chat_connection_mode": args.chat_connection_mode,
                    "chat_recent_turns": args.chat_recent_turns,
                    "image_encoding": args.image_encoding,
                    "jpeg_quality": 90 if args.image_encoding == "jpeg" else None,
                    "image_resize": False,
                    "reasoning_effort": args.reasoning_effort,
                    "max_tokens": args.max_tokens,
                    "physics_dt": 1 / 120,
                    "tcp_offset_m": TCP_OFFSET,
                    "max_task_steps": args.max_task_steps,
                    **(metadata or {}),
                },
                indent=2,
            )
        )
        yield scene, client
    except BaseException:
        import traceback

        traceback.print_exc()
        app.app.post_quit(1)
        raise
    finally:
        try:
            if scene is not None and getattr(scene, "recording", False):
                scene.finish()
        finally:
            app.close()


def run_loop(args, scene, client, *, start_index=1):
    for index in range(start_index, args.max_task_steps + 1):
        if (args.output / "STOP").exists():
            print(
                "Trial stopped by output/STOP; saving current physical state.",
                flush=True,
            )
            capture(scene, args.output / "final")
            break
        folder = args.output / f"step_{index:03d}"
        info, images = capture(scene, folder)
        prompt = prompt_for(scene, info, args.instruction)
        (folder / "prompt.txt").write_text(prompt)
        if args.observe_only:
            break
        print(f"VLM STEP {index}: querying {args.model}", flush=True)
        response = client.chat(
            prompt, images, context_path=folder / "context.json", step_index=index
        )
        (folder / "response.txt").write_text(response)
        (args.output / "api_usage.json").write_text(json.dumps(client.usage, indent=2))
        print(response, flush=True)
        try:
            motion = parse_vlm_motion(response)
            validate_motion_limits(
                motion, scene.config.max_translation_cm, scene.config.max_rotation_deg
            )
            (folder / "action.json").write_text(
                json.dumps(motion_to_dict(motion), indent=2)
            )
            if motion.task_state == 1:
                scene.step_seconds(scene.config.completion_settle_s)
            else:
                target = execute(scene, motion)
                (folder / "target_hand_world.json").write_text(
                    json.dumps(target.tolist())
                )
        except (ValueError, RuntimeError) as error:
            print("EXECUTION FEEDBACK:", str(error), flush=True)
            (folder / "execution_error.txt").write_text(str(error))
        evaluation = scene.evaluate()
        (folder / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
        if evaluation["success"]:
            print("PHYSICAL TASK SUCCESS:", args.task, flush=True)
            capture(scene, args.output / "final")
            break


def finish_run(args, scene):
    # Isaac Sim fast shutdown can terminate the process in app.close().
    # Save and validate the result while simulation_session still handles errors.
    if getattr(scene, "recording", False):
        scene.finish()
    result = scene.result
    if not args.observe_only and not result["success"]:
        completed_turns = len(list(args.output.glob("step_*/response.txt")))
        raise RuntimeError(
            f"Task did not succeed; {completed_turns} replies recorded (limit {args.max_task_steps}); inspect {args.output}"
        )
    print("RESULT", json.dumps(result), flush=True)


def main():
    parser = build_parser()
    args = parser.parse_args()
    templates = configure_args(parser, args)
    with simulation_session(args, templates) as (scene, client):
        run_loop(args, scene, client)
        finish_run(args, scene)


if __name__ == "__main__":
    main()
