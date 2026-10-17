"""InternDataEngine articulated laptop; close its lid through robot contact."""

import json
import numpy as np
from construct_scene.base import BaseTaskScene
from control_robot import GraspController


class ContactForceStop(RuntimeError):
    pass


class LaptopController(GraspController):
    def _move(self, *args, **kwargs):
        self.scene.force_guard = True
        if max(self.scene.contacts()) < 1.0:
            self.scene.last_free_joints = self.robot.get_joint_positions().copy()
        try:
            return super()._move(*args, **kwargs)
        except ContactForceStop as error:
            # Return the drive target to the last unloaded motion-start configuration.
            # This releases stored contact force before the next VLM action.
            self.scene.force_guard = False
            safe = self.scene.last_free_joints.copy()
            super()._move(
                safe,
                np.arange(len(safe)),
                self.scene.config.force_backoff_duration_s,
                0.015,
            )
            self.scene.step_seconds(self.scene.config.action_settle_s)
            raise RuntimeError(
                str(error) + " Backed off to the last unloaded arm target."
            ) from error
        finally:
            self.scene.force_guard = False


class LaptopScene(BaseTaskScene):
    # Keep the articulated-task layout unchanged when PnP props are restored.
    include_tabletop_props = False
    controller_class = LaptopController

    @property
    def initial_hand_position(self):
        return [-0.10, -0.03, self.table_height + 0.48]

    # Back of the lid faces the robot; the keyboard extends away from it.
    # Keep the complete base on the workmat, close to the initial wrist view.
    laptop_xy_m = (-0.05, -0.03)
    laptop_yaw_deg = 180.0

    def build_task(self):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.prims import SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage

        prim = add_reference_to_stage(
            str(self.asset("objects/articulated/laptop/laptop9968/instance.usd")),
            "/World/Laptop",
        )
        scale = 0.28
        position = np.array(
            [*self.laptop_xy_m, self.config.workmat_top_m + 0.2232700486 * scale]
        )
        half_yaw = np.radians(self.laptop_yaw_deg) / 2
        orientation = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])
        self.layout = {
            "laptop_root_position_m": position.tolist(),
            "laptop_root_orientation_wxyz": orientation.tolist(),
            "laptop_yaw_deg": self.laptop_yaw_deg,
            "initial_hand_position_m": self.initial_hand_position,
            "initial_lid_target_deg": 65.0,
        }
        SingleXFormPrim("/World/Laptop").set_local_scale(np.full(3, scale))
        SingleXFormPrim("/World/Laptop").set_world_pose(
            position=position, orientation=orientation
        )
        base_path = "/World/Laptop/instance/base"
        self.lid_path = "/World/Laptop/instance/contact_link"
        joint_path = base_path + "/contact_link_revolute"
        self.lid_joint = UsdPhysics.RevoluteJoint(self.stage.GetPrimAtPath(joint_path))
        self.lid_joint.CreateLowerLimitAttr(0.0)
        self.lid_joint.CreateUpperLimitAttr(110.0)
        self.drive = UsdPhysics.DriveAPI(self.lid_joint.GetPrim(), "angular")
        self.drive.CreateStiffnessAttr(80.0)
        self.drive.CreateDampingAttr(3.0)
        self.drive.CreateMaxForceAttr(10.0)
        self.drive.CreateTargetPositionAttr(65.0)
        PhysxSchema.PhysxJointAPI.Apply(
            self.lid_joint.GetPrim()
        ).CreateJointFrictionAttr(1.0)
        fixed = UsdPhysics.FixedJoint.Define(self.stage, "/World/Laptop/FixedBase")
        fixed.CreateBody1Rel().SetTargets([base_path])
        fixed.CreateLocalPos0Attr(Gf.Vec3f(*position))
        fixed.CreateLocalRot0Attr(
            Gf.Quatf(float(orientation[0]), Gf.Vec3f(*orientation[1:]))
        )
        self.laptop = self.world.scene.add(
            SingleArticulation("/World/Laptop/instance", name="laptop")
        )
        self.lid = SingleXFormPrim(self.lid_path)
        self.laptop_base = SingleXFormPrim(base_path)
        self.contact_targets = [self.lid_path]
        material = UsdPhysics.MaterialAPI.Apply(
            self.stage.DefinePrim("/World/Looks/LaptopPhysics", "Material")
        )
        material.CreateStaticFrictionAttr(0.6)
        material.CreateDynamicFrictionAttr(0.5)
        material.CreateRestitutionAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(material.GetPrim()),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        shell = self.material(
            "LaptopAluminum", (0.16, 0.18, 0.20), roughness=0.38, metallic=0.55
        )
        screen = self.material("LaptopDisplay", (0.008, 0.014, 0.022), roughness=0.18)
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(
                    p
                ).CreateSolverPositionIterationCountAttr(64)
            if p.IsA(UsdGeom.Mesh):
                if "/contact_link/visuals/" in str(p.GetPath()):
                    visual = screen if p.GetName() == "screen_11" else shell
                    UsdShade.MaterialBindingAPI.Apply(p).Bind(
                        visual, bindingStrength=UsdShade.Tokens.strongerThanDescendants
                    )
                is_collider = "/collisions/" in str(p.GetPath())
                UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(is_collider)
                if is_collider:
                    UsdGeom.Imageable(p).MakeInvisible()
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr(
                    "convexHull"
                )
                collision = PhysxSchema.PhysxCollisionAPI.Apply(p)
                collision.CreateContactOffsetAttr(0.001)
                collision.CreateRestOffsetAttr(0.0)
        # A laptop resting on a high-friction mat is fixed at its base for this demo.
        # Only its passive hinge moves during task execution.

    def initialize_task(self):
        self.initial_angle = self.angle()
        self.initial_base = self.laptop_base.get_world_pose()[0].copy()
        self.contact_seen = False
        self.closed_stable_samples = 0
        self.peak_contact = 0.0
        self.drive.GetStiffnessAttr().Set(0.0)
        self.drive.GetDampingAttr().Set(0.05)
        self.drive.GetMaxForceAttr().Set(0.0)
        (self.output / "laptop_layout.json").write_text(
            json.dumps(self.layout, indent=2) + "\n"
        )
        print("Laptop initial joint angle:", self.initial_angle, flush=True)

    def angle(self):
        return float(np.degrees(self.laptop.get_joint_positions()[0]))

    def task_sample(self):
        forces = self.contacts()
        self.peak_contact = max(self.peak_contact, max(forces))
        self.contact_seen |= max(forces) > 0.1
        if self.closed_and_clear():
            self.closed_stable_samples += 1
        else:
            self.closed_stable_samples = 0
        return {
            "time": self.tick / 120,
            "lid_joint_deg": self.angle(),
            "tcp": self.tcp_matrix()[:3, 3].tolist(),
            "finger_contact_N": forces,
            "lid_position": self.lid.get_world_pose()[0].tolist(),
            "gripper_width": float(
                sum(self.robot.get_joint_positions()[self.finger_indices])
            ),
        }

    def step(self, count=1):
        for _ in range(count):
            super().step()
            if getattr(self, "recording", False):
                force = max(self.contacts())
                self.peak_contact = max(self.peak_contact, force)
                # Latch contact at the physics rate, including brief force-guard hits
                # between the 15 Hz video/telemetry samples.
                self.contact_seen |= force > 0.1
                if getattr(self, "force_guard", False) and force > 25.0:
                    from isaacsim.core.utils.types import ArticulationAction

                    self.robot.apply_action(
                        ArticulationAction(
                            joint_positions=self.robot.get_joint_positions()
                        )
                    )
                    raise ContactForceStop(
                        f"Lid contact force {force:.1f} N exceeded 25 N; stopped arm target. Retract and inspect before pressing again."
                    )

    def closed_and_clear(self):
        # Joint zero is the packaged laptop's closed configuration.
        # Require the TCP to be at least 0.19 m above the table.
        return (
            abs(self.angle()) < 3.0
            and abs(float(self.laptop.get_joint_velocities()[0])) < 0.03
            and self.tcp_matrix()[2, 3] > self.table_height + 0.19
            and max(self.contacts()) < 0.1
        )

    def evaluate(self):
        angle = self.angle()
        velocity = float(self.laptop.get_joint_velocities()[0])
        drift = float(
            np.linalg.norm(self.laptop_base.get_world_pose()[0] - self.initial_base)
        )
        success = (
            self.initial_angle > 45.0
            and self.contact_seen
            and self.closed_and_clear()
            and self.closed_stable_samples >= 15
            and drift < 0.002
        )
        return {
            "success": bool(success),
            "task": "close_laptop",
            "initial_joint_deg": self.initial_angle,
            "final_joint_deg": angle,
            "joint_velocity_rad_s": velocity,
            "robot_lid_contact_seen": bool(self.contact_seen),
            "peak_finger_contact_N": self.peak_contact,
            "closed_stable_seconds": self.closed_stable_samples / 15,
            "hand_clear": bool(
                self.tcp_matrix()[2, 3] > self.table_height + 0.19
                and max(self.contacts()) < 0.1
            ),
            "base_drift_m": drift,
            "base_fixed": True,
            "physics_only": True,
            "hinge_drive_enabled_during_task": False,
            "object_poses_sent_to_vlm": False,
        }
