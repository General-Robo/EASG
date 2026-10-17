"""Isaac application startup and standalone preview helpers."""

import argparse
import os
from pathlib import Path
from settings import PROJECT_ROOT

DEFAULT_ASSETS = PROJECT_ROOT / "Asset"


def arguments(description, *, demo=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", default="default.json", help="JSON filename in EASG/config"
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path(
            os.environ.get(
                "EASG_ASSETS", os.environ.get("ROBOSCENE_ASSETS", DEFAULT_ASSETS)
            )
        ),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--gpu",
        type=int,
        choices=range(8),
        default=0,
        help="GPU index for rendering and physics (0-7 on eight-GPU DLC workers)",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument(
        "--path-tracing", action="store_true", help="Slower, higher quality rendering"
    )
    if demo:
        parser.add_argument(
            "--steps",
            type=int,
            default=None,
            help="Extra simulation steps; GUI defaults to unlimited, headless to 120",
        )
        parser.add_argument("--screenshot", action="store_true")
    return parser


def start_app(args):
    # Kit must start before importing Isaac APIs.
    from isaacsim import SimulationApp

    return SimulationApp(
        {
            "headless": args.headless,
            "width": 1280,
            "height": 960,
            "renderer": "PathTracing" if args.path_tracing else "RayTracedLighting",
            "samples_per_pixel_per_frame": 32,
            "multi_gpu": False,
            "active_gpu": args.gpu,
            "physics_gpu": args.gpu,
        }
    )


def keep_running(scene, args):
    steps = args.steps if args.steps is not None else (120 if args.headless else None)
    if steps is not None:
        if steps < 0:
            raise ValueError("--steps must be nonnegative")
        scene.step(steps)
    else:
        while scene.app.is_running():
            scene.step()
