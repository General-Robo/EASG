"""Task-independent sensors, lifecycle, synchronous recording and output."""

import json
from pathlib import Path
import numpy as np
from construct_scene.tabletop import TabletopScene
from control_robot import GraspController

TCP_OFFSET = 0.095


class BaseTaskScene(TabletopScene):
    controller_class = GraspController

    def add_task_objects(self):
        from pxr import Usd, UsdPhysics, UsdShade, PhysxSchema
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.utils.rotations import rot_matrix_to_quat
        from isaacsim.sensors.camera import Camera

        for name in ("Floor", "BackWall", "SideWall", "Skirting"):
            UsdPhysics.CollisionAPI(
                self.stage.GetPrimAtPath("/World/" + name)
            ).CreateCollisionEnabledAttr(False)
        # A fixed table is a kinematic rigid body with colliders, not a falling dynamic body.
        body = UsdPhysics.RigidBodyAPI.Apply(self.stage.GetPrimAtPath("/World/Table"))
        body.CreateKinematicEnabledAttr(True)

        self.build_task()
        rubber = UsdPhysics.MaterialAPI.Apply(
            self.stage.DefinePrim("/World/Looks/FingerPhysics", "Material")
        )
        rubber.CreateStaticFrictionAttr(1.2)
        rubber.CreateDynamicFrictionAttr(1.0)
        rubber.CreateRestitutionAttr(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(
            rubber.GetPrim()
        ).CreateFrictionCombineModeAttr("max")
        self.finger_contacts = []
        for side in ("left", "right"):
            path = f"/World/Robot/fr3/panda_{side}finger"
            for p in Usd.PrimRange(self.stage.GetPrimAtPath(path)):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(
                    UsdShade.Material(rubber.GetPrim()),
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    collision = PhysxSchema.PhysxCollisionAPI.Apply(p)
                    collision.CreateContactOffsetAttr(0.002)
                    collision.CreateRestOffsetAttr(0.0)
            view = RigidPrim(
                path,
                name=side + "_contact",
                track_contact_forces=True,
                contact_filter_prim_paths_expr=self.contact_targets,
            )
            self.finger_contacts.append(self.world.scene.add(view))
        self.wrist = self.world.scene.add(
            Camera(
                "/World/Robot/fr3/panda_hand/WristCamera",
                name="wrist",
                resolution=(640, 480),
                frequency=30,
            )
        )
        # ROS optical frame in panda_hand coordinates: forward along the fingers,
        # image right along jaw separation, image down toward hand -X.
        # Side offset frames the jaws below centre; the wide lens keeps both visible.
        # Default: +TCP Y 10 mm from the fixed baseline recorded in note.txt.
        eye = np.array([0.07, 0.0, 0.030])
        x = np.array([0.0, 1.0, 0.0])
        y = np.array([-1.0, 0.0, 0.0])
        z = np.array([0.0, 0.0, 1.0])
        self.wrist.set_local_pose(
            eye, rot_matrix_to_quat(np.column_stack([x, y, z])), camera_axes="ros"
        )
        self.wrist.prim.GetAttribute("focalLength").Set(5.0)
        self.wrist.prim.GetAttribute("horizontalAperture").Set(20.955)
        self.wrist.prim.GetAttribute("verticalAperture").Set(15.71625)
        self.wrist.set_clipping_range(0.015, 10)
        self.add_wrist_fill_light()

    def add_wrist_fill_light(self):
        """Shadowless fill follows the wrist camera, including installation changes."""
        from pxr import Gf, UsdGeom, UsdLux

        light = UsdLux.RectLight.Define(
            self.stage, str(self.wrist.prim.GetPath()) + "/ShadowlessFill"
        )
        light.CreateWidthAttr(1.2)
        light.CreateHeightAttr(1.2)
        light.CreateIntensityAttr(1500.0)
        light.CreateColorAttr(Gf.Vec3f(1, 1, 1))
        # USD cameras and rect lights face local -Z. +Z is behind the lens;
        # parenting preserves this offset and direction when the camera moves.
        xform = UsdGeom.Xformable(light)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.01))
        UsdLux.LightAPI(
            light.GetPrim()
        ).GetShadowLinkCollectionAPI().CreateIncludeRootAttr(False)
        UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(False)

    @property
    def initial_hand_position(self):
        return [-0.06, -0.03, self.config.workmat_top_m + 0.376]

    def prepare(self, output):
        self.wrist.add_distance_to_image_plane_to_frame()
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.controller = getattr(self, "controller_class", GraspController)(self)
        self.controller.move_pose(
            self.initial_hand_position,
            [0, 1, 0, 0],
            duration=self.config.initial_pose_duration_s,
        )
        self.initialize_task()
        self.telemetry = []
        self.recording = True
        self.video = None
        import imageio.v2 as imageio

        self.video = imageio.get_writer(
            str(self.output / "execution.mp4"), fps=15, codec="libx264", quality=7
        )
        self.record_sample()

        self.after_prepare()

    def hand_matrix(self):
        from isaacsim.core.utils.rotations import quat_to_rot_matrix

        position, quat = self.hand.get_world_pose()
        T = np.eye(4)
        T[:3, :3] = quat_to_rot_matrix(quat)
        T[:3, 3] = position
        return T

    def tcp_matrix(self):
        from motion import paper_tcp_pose

        return paper_tcp_pose(self.hand_matrix(), TCP_OFFSET)

    def contacts(self):
        return [
            float(np.linalg.norm(v.get_contact_force_matrix(dt=1 / 120)))
            for v in self.finger_contacts
        ]

    def step(self, count=1):
        for _ in range(count):
            super().step()
            if getattr(self, "recording", False) and self.tick % 8 == 0:
                self.record_sample()

    def finish(self):
        if not getattr(self, "recording", False):
            return self.result
        self.recording = False
        if self.video is not None:
            self.video.close()
            self.video = None
        (self.output / "telemetry.json").write_text(
            json.dumps(self.telemetry, indent=2)
        )
        self.result = self.evaluate()
        (self.output / "result.json").write_text(json.dumps(self.result, indent=2))
        return self.result

    def build_task(self):
        raise NotImplementedError

    def initialize_task(self):
        raise NotImplementedError

    def task_sample(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def after_prepare(self):
        """Optional task-specific metadata/initial observation hook."""

    def record_sample(self):
        self.telemetry.append(self.task_sample())
        if self.video is not None:
            rgba = self.camera.get_rgba()
            if rgba is not None and rgba.size:
                self.video.append_data(rgba[:, :, :3].astype("uint8"))
