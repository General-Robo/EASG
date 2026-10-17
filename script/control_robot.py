"""Control Franka using joint targets or world-frame hand poses (quaternion wxyz)."""

import json
import math
import numpy as np
from pathlib import Path

from settings import config_from_args
from settings import SCRIPT_DIR as HERE


class RobotController:
    """Small synchronous API: move_joints, move_pose, gripper, run_task."""

    def __init__(self, scene):
        self.scene, self.robot = scene, scene.robot
        self.ik = None  # Joint-only tasks never load an IK solver.

    @staticmethod
    def vector(values, count, label):
        import numpy as np

        value = np.asarray(values, dtype=float)
        if value.shape != (count,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{label} needs {count} finite numbers")
        return value

    def _move(self, target, indices, duration, tolerance):
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction

        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be positive and finite")
        limits = self.robot.dof_properties
        if np.any(target < limits["lower"][indices] - 1e-6) or np.any(
            target > limits["upper"][indices] + 1e-6
        ):
            raise ValueError(f"Joint target outside USD limits: {target}")
        target = np.clip(target, limits["lower"][indices], limits["upper"][indices])
        start = self.robot.get_joint_positions()[indices]
        # Cubic easing, with duration extended to cap peak target speed to 0.6 rad/s.
        duration = max(duration, float(1.5 * np.max(np.abs(target - start)) / 0.6))
        steps = max(1, math.ceil(duration / self.scene.world.get_physics_dt()))
        for i in range(1, steps + 1):
            t = i / steps
            q = start + (target - start) * (3 * t * t - 2 * t * t * t)
            self.robot.apply_action(
                ArticulationAction(joint_positions=q, joint_indices=indices)
            )
            self.scene.step()
        for _ in range(240):
            error = float(
                np.max(np.abs(self.robot.get_joint_positions()[indices] - target))
            )
            if error < tolerance:
                return error
            self.scene.step()
        raise RuntimeError(f"Joint target not reached: max error {error:.5f}")

    def move_joints(self, positions, duration=None):
        duration = self.scene.config.arm_duration_s if duration is None else duration
        target = self.vector(positions, 7, "joints (radians)")
        error = self._move(target, self.scene.arm_indices, duration, 0.015)
        return {"type": "joint", "max_error_rad": error}

    def move_pose(self, position, orientation=None, duration=None):
        import numpy as np
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        from isaacsim.core.utils.extensions import get_extension_path_from_name
        from isaacsim.robot_motion.motion_generation import (
            LulaKinematicsSolver,
            ArticulationKinematicsSolver,
        )

        duration = self.scene.config.arm_duration_s if duration is None else duration
        position = self.vector(position, 3, "position (world metres)")
        if orientation is not None:
            orientation = self.vector(orientation, 4, "orientation (wxyz)")
            norm = np.linalg.norm(orientation)
            if norm < 1e-8:
                raise ValueError("Quaternion must have nonzero norm")
            orientation /= norm
        if self.ik is None:
            root = (
                Path(
                    get_extension_path_from_name(
                        "isaacsim.robot_motion.motion_generation"
                    )
                )
                / "motion_policy_configs/franka"
            )
            self.ik = LulaKinematicsSolver(
                str(root / "rmpflow/robot_descriptor.yaml"),
                str(root / "lula_franka_gen.urdf"),
            )
            self.solver = ArticulationKinematicsSolver(
                self.robot, self.ik, "panda_hand"
            )
        self.ik.set_robot_base_pose(*self.scene.base.get_world_pose())
        action, success = self.solver.compute_inverse_kinematics(
            position, orientation, position_tolerance=0.001, orientation_tolerance=0.01
        )
        if not success:
            raise RuntimeError(f"IK failed: target {position.tolist()}")
        self._move(action.joint_positions, action.joint_indices, duration, 0.008)
        # Check actual simulated hand transform, independently of Lula FK.
        actual, quaternion = self.scene.hand.get_world_pose()
        error = float(np.linalg.norm(actual - position))
        angle = None
        if orientation is not None:
            rotation = quat_to_rot_matrix(orientation).T @ quat_to_rot_matrix(
                quaternion
            )
            angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
        if error > 0.015 or (angle is not None and angle > 0.06):
            raise RuntimeError(
                f"Pose not reached: position error={error:.4f} m, orientation error={angle} rad"
            )
        return {
            "type": "pose",
            "position_error_m": error,
            "orientation_error_rad": angle,
        }

    def gripper(self, width, duration=None):
        import numpy as np

        duration = (
            self.scene.config.gripper_duration_s if duration is None else duration
        )
        maximum = self.scene.config.gripper_max_width_m
        if not math.isfinite(width) or not 0 <= width <= maximum:
            raise ValueError(f"gripper width must be between 0 and {maximum} metres")
        error = self._move(
            np.array([width / 2, width / 2]), self.scene.finger_indices, duration, 0.002
        )
        return {"type": "gripper", "max_error_m": error}

    def run_task(self, steps):
        report = []
        for i, step in enumerate(steps):
            kind = step["type"]
            print(f"Task step {i + 1}: {kind}", flush=True)
            if kind == "joint":
                result = self.move_joints(step["positions"], step.get("duration"))
            elif kind == "pose":
                result = self.move_pose(
                    step["position"], step.get("orientation"), step.get("duration")
                )
            elif kind == "gripper":
                result = self.gripper(step["width"], step.get("duration"))
            elif kind == "wait":
                seconds = step["duration"]
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("wait duration must be nonnegative and finite")
                self.scene.step(math.ceil(seconds / self.scene.world.get_physics_dt()))
                result = {"type": "wait"}
            else:
                raise ValueError(f"Unknown task type: {kind}")
            report.append(result)
            print(result, flush=True)
        return report


def main():
    from construct_scene.tabletop import TabletopScene
    from simulation import arguments, keep_running, start_app

    parser = arguments(__doc__, demo=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--joints", type=float, nargs=7, metavar="RAD")
    mode.add_argument(
        "--pose",
        type=float,
        nargs=7,
        metavar="VALUE",
        help="World hand pose: x y z qw qx qy qz",
    )
    mode.add_argument(
        "--task", type=Path, help="JSON list of joint/pose/gripper/wait steps"
    )
    args = parser.parse_args()
    config = config_from_args(parser, args)
    if args.joints is not None:
        task = [
            {
                "type": "joint",
                "positions": args.joints,
                "duration": config.arm_duration_s,
            }
        ]
    elif args.pose is not None:
        task = [
            {
                "type": "pose",
                "position": args.pose[:3],
                "orientation": args.pose[3:],
                "duration": config.arm_duration_s,
            }
        ]
    else:
        task = json.loads((args.task or HERE / "demo_task.json").read_text())
    app = start_app(args)
    try:
        scene = TabletopScene(app, args.assets, config=config)
        report = RobotController(scene).run_task(task)
        scene.save(args.output, args.screenshot)
        (args.output / "task_report.json").write_text(json.dumps(report, indent=2))
        print("Task completed.", flush=True)
        keep_running(scene, args)
    except Exception:
        import traceback

        traceback.print_exc()
        app.app.post_quit(1)
        raise
    finally:
        app.close()


class GraspController(RobotController):
    """Position-drive grasp: allow fingers to stop at contact while retaining force."""

    def gripper(self, width, duration=None):
        from isaacsim.core.utils.types import ArticulationAction

        maximum = self.scene.config.gripper_max_width_m
        duration = (
            self.scene.config.gripper_duration_s if duration is None else duration
        )
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be positive and finite")
        if not np.isfinite(width) or not 0 <= width <= maximum:
            raise ValueError(f"width must be in [0, {maximum}]")
        indices = self.scene.finger_indices
        self.robot.get_articulation_controller().set_max_efforts(
            np.array([35.0, 35.0]), joint_indices=indices
        )
        if abs(width - getattr(self, "desired_width", maximum)) < 1e-6:
            # Re-issuing close must NOT ramp from measured contact width: that
            # temporarily removes the spring force and drops the held object.
            self.scene.step_seconds(duration * 0.25)
            return {
                "requested_width": width,
                "measured_width": float(sum(self.robot.get_joint_positions()[indices])),
            }
        self.desired_width = width
        start = self.robot.get_joint_positions()[indices]
        target = np.array([width / 2, width / 2])
        steps = max(1, math.ceil(duration / self.scene.world.get_physics_dt()))
        for i in range(steps):
            q = start + (target - start) * min(1.0, (i + 1) / (steps * 0.75))
            self.robot.apply_action(
                ArticulationAction(joint_positions=q, joint_indices=indices)
            )
            self.scene.step()
        return {
            "requested_width": width,
            "measured_width": float(sum(self.robot.get_joint_positions()[indices])),
        }


if __name__ == "__main__":
    main()
