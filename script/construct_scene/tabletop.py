"""Shared tabletop environment, robot and wood workmat."""

import json
import math
import sys
from pathlib import Path
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import HOME, load_config
from construct_scene.assets import (
    resolve_asset,
    reference_asset as add_reference_to_stage,
    normalize_mesh_normals,
)


def build_workmat(scene, rubber):
    from pxr import UsdPhysics

    thickness = scene.config.workmat_thickness_m
    # Approved 2026-10-16: expand both axes 5 cm, preserving the board centre.
    scene.box(
        "WorkMat",
        (0.05, 0, scene.table_height + thickness / 2),
        (0.60, 0.59, thickness),
        rubber,
    )
    from pxr import PhysxSchema, UsdShade

    mat_contact = UsdShade.Material.Define(scene.stage, "/World/Looks/WorkMatPhysics")
    mat_physics = UsdPhysics.MaterialAPI.Apply(mat_contact.GetPrim())
    mat_physics.CreateStaticFrictionAttr(0.5)
    mat_physics.CreateDynamicFrictionAttr(0.5)
    mat_physics.CreateRestitutionAttr(0.0)
    compliant = PhysxSchema.PhysxMaterialAPI.Apply(mat_contact.GetPrim())
    compliant.CreateCompliantContactStiffnessAttr(
        scene.config.workmat_contact_stiffness
    )
    compliant.CreateCompliantContactDampingAttr(scene.config.workmat_contact_damping)
    compliant.CreateCompliantContactAccelerationSpringAttr(False)
    UsdShade.MaterialBindingAPI.Apply(scene.stage.GetPrimAtPath("/World/WorkMat")).Bind(
        mat_contact,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    add_workmat_appearance(scene)


def add_workmat_appearance(scene):
    """Keep WorkMat's placement and physics material; split collision and visuals.

    The unchanged unit-box collider becomes an invisible child of WorkMat.
    Its visible sibling follows the same transform and supplies workspace bounds.
    Texture paths resolve through the selected Asset root, never output or /tmp.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    root = scene.stage.GetPrimAtPath("/World/WorkMat")
    size = np.asarray(root.GetAttribute("xformOp:scale").Get(), dtype=float)
    cube_size = root.GetAttribute("size").Get()
    root.SetTypeName("Xform")
    root.RemoveAPI(UsdPhysics.CollisionAPI)
    root.RemoveProperty("size")
    collider = UsdGeom.Cube.Define(scene.stage, "/World/WorkMat/Collider")
    collider.CreateSizeAttr(cube_size)
    collider.CreateVisibilityAttr("invisible")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    half = size / 2
    corner = min(0.012, float(min(half[:2])) * 0.5)
    bevel = min(0.002, float(half[2]) * 0.5, corner * 0.5)
    arc = np.concatenate(
        [np.linspace(k * np.pi / 2, (k + 1) * np.pi / 2, 17) for k in range(4)]
    )
    centres = np.repeat(
        np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * (half[:2] - corner), 17, axis=0
    )
    dirs = np.column_stack([np.cos(arc), np.sin(arc)])
    points, normals = [], []
    rings = [(half[2] - bevel, phi) for phi in np.linspace(np.pi / 2, 0, 9)]
    rings += [(-half[2] + bevel, phi) for phi in np.linspace(0, -np.pi / 2, 9)]
    for zbase, phi in rings:
        xy = centres + (corner - bevel + bevel * np.cos(phi)) * dirs
        points.extend(
            np.column_stack([xy, np.full(len(arc), zbase + bevel * np.sin(phi))])
        )
        normals.extend(
            np.column_stack([dirs * np.cos(phi), np.full(len(arc), np.sin(phi))])
        )
    n = len(arc)
    counts, indices = [], []
    for ring in range(len(rings) - 1):
        for k in range(n):
            counts.append(4)
            indices.extend(
                [
                    ring * n + k,
                    (ring + 1) * n + k,
                    (ring + 1) * n + (k + 1) % n,
                    ring * n + (k + 1) % n,
                ]
            )
    for is_top in (True, False):
        ci = len(points)
        points.append([0.0, 0.0, half[2] if is_top else -half[2]])
        normals.append([0.0, 0.0, 1.0 if is_top else -1.0])
        offset = 0 if is_top else (len(rings) - 1) * n
        for k in range(n):
            counts.append(3)
            tri = [ci, offset + k, offset + (k + 1) % n]
            indices.extend(tri if is_top else tri[::-1])
    pts = np.asarray(points)
    visual = UsdGeom.Mesh.Define(scene.stage, "/World/WorkMat/Visual")
    visual.CreatePointsAttr(pts.tolist())
    visual.CreateFaceVertexCountsAttr(counts)
    visual.CreateFaceVertexIndicesAttr(indices)
    visual.CreateSubdivisionSchemeAttr("none")
    visual.CreateNormalsAttr(np.asarray(normals).tolist())
    visual.SetNormalsInterpolation("vertex")
    # Cancel the unit-cube scale, while inheriting its placement/rotation.
    UsdGeom.Xformable(visual).AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*(1 / size))
    )
    # Match approved F preview: grain along local X, 65 cm texture period.
    # Per-face box projection avoids stretching top UVs down thin side faces.
    # Side faces reuse the source grain; this is not reconstructed end grain.
    render_points = np.asarray(visual.GetPointsAttr().Get())
    render_normals = np.asarray(visual.GetNormalsAttr().Get())
    face_indices = np.asarray(indices)
    uv, offset = [], 0
    for count in counts:
        ids = face_indices[offset : offset + count]
        offset += count
        p = render_points[ids]
        axis = int(np.argmax(np.abs(render_normals[ids].mean(axis=0))))
        if axis == 2:
            a, b = p[:, 0], p[:, 1]
        elif axis == 1:
            a, b = p[:, 0], p[:, 2]
        else:
            a, b = p[:, 2], p[:, 1]
        uv.extend((np.column_stack([a, b]) / 0.65 + 0.5).tolist())
    UsdGeom.PrimvarsAPI(visual).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying"
    ).Set(uv)
    visual.CreatePurposeAttr("render")

    material = UsdShade.Material.Define(scene.stage, "/World/Looks/WorkMatLocalAsh0111")
    shader = UsdShade.Shader.Define(
        scene.stage, material.GetPath().AppendChild("Surface")
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    reader = UsdShade.Shader.Define(scene.stage, material.GetPath().AppendChild("UV"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for channel, filename, field, kind, output in [
        (
            "color",
            "Ash_BaseColor.png",
            "diffuseColor",
            Sdf.ValueTypeNames.Color3f,
            "rgb",
        ),
        ("roughness", "Ash_ORM.png", "roughness", Sdf.ValueTypeNames.Float, "g"),
        ("normal", "Ash_Normal.png", "normal", Sdf.ValueTypeNames.Normal3f, "rgb"),
    ]:
        texture = scene.asset(f"environment/workmat/local_ash_0111/{filename}")
        tex = UsdShade.Shader.Define(
            scene.stage, material.GetPath().AppendChild(channel)
        )
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
            "sRGB" if channel == "color" else "raw"
        )
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        if channel == "normal":
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(2, 2, 2, 1)
            )
            tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(-1, -1, -1, 0)
            )
        tex.CreateOutput(
            output,
            Sdf.ValueTypeNames.Float3 if output == "rgb" else Sdf.ValueTypeNames.Float,
        )
        shader.CreateInput(field, kind).ConnectToSource(tex.ConnectableAPI(), output)
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)


class TabletopScene:
    """Own the World, robot and camera; no imports from InternDataEngine."""

    include_tabletop_props = True

    def __init__(self, app, assets, config=None):
        import numpy as np
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.prims import SingleXFormPrim
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.sensors.camera import Camera

        self.config = config if config is not None else load_config()
        self.app, self.assets = app, Path(assets).expanduser().resolve()
        import sys
        from importlib.metadata import version

        self.runtime = {
            "isaacsim_version": version("isaacsim"),
            "python_executable": sys.executable,
        }
        print("Simulation runtime:", self.runtime, flush=True)
        self.tick = 0
        self.world = World(
            stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30
        )
        self.stage = self.world.stage
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath("/World"))
        self.table_height = self.config.table_height_m
        self.objects = {}

        floor = self.material("Floor", (0.27, 0.29, 0.30), roughness=0.8)
        wall = self.material("Wall", (0.70, 0.72, 0.70), roughness=0.9)
        rubber = self.material("Rubber", (0.045, 0.065, 0.07), roughness=0.85)
        metal = self.material("Metal", (0.32, 0.34, 0.36), metallic=0.8, roughness=0.3)
        self.box("Floor", (0, 0, -0.06), (8, 8, 0.12), floor)
        self.box("BackWall", (-1.8, 0, 1.5), (0.12, 8, 3), wall)
        self.box("SideWall", (0, 2.2, 1.5), (4, 0.12, 3), wall)
        self.box("Skirting", (-1.72, 0, 0.06), (0.04, 8, 0.12), metal)
        self.place_asset(
            "Table",
            "environment/table0/instance.usd",
            (0, 0),
            0,
            dimensions=(1.6, 0.9, self.table_height),
            dynamic=False,
        )
        build_workmat(self, rubber)
        self.box(
            "RobotMount",
            (-0.50, 0, self.table_height + 0.015),
            (0.23, 0.24, 0.03),
            metal,
        )

        if self.include_tabletop_props:
            # Opposite the robot (-X): all props beyond the board's +X edge (0.350 m).
            # Align visible centres near X=0.66 m, with equal edge gaps along Y.
            # These scans have different native up axes; do not share rotations.
            source = "objects/rigid"
            self.place_asset(
                "Mug",
                f"{source}/omniobject3d-mug/google_scan-mug_0160/Aligned_obj.usd",
                (0.668, 0.2428),
                self.table_height + 0.005,
                height=0.115,
                rotation=(0, 0, -30),
            )
            candidates = sorted(
                (self.assets / source / "google_scan-book").glob("*/Aligned_obj.usd")
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No USD assets in {self.assets / source / 'google_scan-book'}"
                )
            self.place_asset(
                "Book",
                candidates[0],
                (0.66, -0.26),
                self.table_height + 0.005,
                dimensions=(0.22, 0.16, 0.025),
                rotation=(0, 0, 8),
            )
            from construct_scene.geometry import world_bounds
            from pxr import Usd

            mug = world_bounds(self.stage, "/World/Mug")
            book = world_bounds(self.stage, "/World/Book")
            middle_y = float((mug[0, 1] + book[1, 1]) / 2)
            # Approved preview A: upright mint clock, 13 cm tall, facing the camera.
            self.place_asset(
                "Clock",
                f"{source}/clock/00000/object.usdz",
                (0.65, middle_y),
                self.table_height + 0.0005,
                height=0.13,
                rotation=(0, 0, 35),
                dynamic=False,
            )
            # Render the textured visual, not the packaged collision mesh.
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath("/World/Clock")):
                if prim.IsA(UsdGeom.Mesh):
                    is_collision = "collision" in str(prim.GetPath()).lower()
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(
                        is_collision
                    )
                    if is_collision:
                        UsdGeom.Imageable(prim).MakeInvisible()

        robot_path = "/World/Robot"
        add_reference_to_stage(str(self.asset("robot/franka/robot.usd")), robot_path)
        SingleXFormPrim(robot_path).set_world_pose(
            position=np.array([-0.50, 0, self.table_height + 0.03])
        )
        self.robot = self.world.scene.add(Robot(prim_path=robot_path, name="franka"))
        self.base = SingleXFormPrim(robot_path + "/fr3/panda_link0")
        self.hand = SingleXFormPrim(robot_path + "/fr3/panda_hand")
        # Explicit contact material on the table and props, authored only in our stage.
        contact = UsdPhysics.MaterialAPI.Apply(
            self.stage.DefinePrim("/World/Looks/Contact", "Material")
        )
        contact.CreateStaticFrictionAttr(0.7)
        contact.CreateDynamicFrictionAttr(0.5)
        contact.CreateRestitutionAttr(0.02)
        from pxr import UsdShade

        for name in self.objects:
            UsdShade.MaterialBindingAPI.Apply(
                self.stage.GetPrimAtPath("/World/" + name)
            ).Bind(UsdShade.Material(contact.GetPrim()), materialPurpose="physics")

        dome = UsdLux.DomeLight.Define(self.stage, "/World/Lights/Environment")
        dome.CreateTextureFileAttr(
            str(self.asset("environment/envmap_lib/studio_small_03_1k.hdr"))
        )
        dome.CreateIntensityAttr(250)
        dome.CreateTextureFormatAttr("latlong")
        key = UsdLux.RectLight.Define(self.stage, "/World/Lights/Softbox")
        key.CreateWidthAttr(2.2)
        key.CreateHeightAttr(1.5)
        key.CreateIntensityAttr(500)
        key.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.83))
        UsdGeom.Xformable(key).AddTranslateOp().Set(Gf.Vec3d(0.3, -0.3, 2.7))

        # Move the external camera 35% closer along the same viewing ray.
        eye, target = (
            (1.365, -1.56, self.table_height + 0.8475),
            (0, 0, self.table_height + 0.10),
        )
        camera_path = "/World/Camera"
        self.camera = self.world.scene.add(
            Camera(camera_path, resolution=(1280, 960), frequency=30)
        )
        transform = (
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )
        q = transform.ExtractRotationQuat()
        self.camera.set_world_pose(
            np.array(eye), np.array([q.GetReal(), *q.GetImaginary()]), camera_axes="usd"
        )
        self.camera.prim.GetAttribute("focalLength").Set(
            28.0
        )  # USD tenths of a stage unit
        self.camera.set_clipping_range(0.02, 100)
        set_camera_view(
            eye=np.array(eye),
            target=np.array(target),
            camera_prim_path="/OmniverseKit_Persp",
        )
        self.add_task_objects()
        self.world.reset()
        self.robot.set_solver_position_iteration_count(64)
        self.robot.set_solver_velocity_iteration_count(4)
        self.arm_indices = np.array(
            [self.robot.get_dof_index(f"panda_joint{i}") for i in range(1, 8)]
        )
        self.finger_indices = np.array(
            [self.robot.get_dof_index(f"panda_finger_joint{i}") for i in (1, 2)]
        )
        q = self.robot.get_joint_positions()
        q[self.arm_indices], q[self.finger_indices] = (
            HOME,
            self.config.gripper_max_width_m / 2,
        )
        self.robot.set_joints_default_state(positions=q)
        self.robot.set_joint_positions(q)  # Teleport ONLY at initialization.
        self.robot.get_articulation_controller().set_gains(
            kps=np.array([6000.0] * 7 + [800.0, 800.0]),
            kds=np.array([180.0] * 7 + [40.0, 40.0]),
        )
        from isaacsim.core.utils.types import ArticulationAction

        self.robot.apply_action(ArticulationAction(joint_positions=q))
        self.step(180)
        print("Scene ready. DOFs:", self.robot.dof_names, flush=True)

    def add_task_objects(self):
        """Subclass hook: add sensors and task bodies before physics initialization."""

    def asset(self, relative):
        return resolve_asset(self.assets, relative)

    def material(self, name, color, roughness=0.5, metallic=0.0):
        from pxr import Gf, Sdf, UsdShade

        material = UsdShade.Material.Define(self.stage, "/World/Looks/" + name)
        shader = UsdShade.Shader.Define(
            self.stage, material.GetPath().AppendChild("Surface")
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        return material

    def box(self, name, position, size, material):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

        cube = UsdGeom.Cube.Define(self.stage, "/World/" + name)
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)

    def place_asset(
        self,
        name,
        relative,
        xy,
        bottom,
        height=None,
        dimensions=None,
        dynamic=True,
        rotation=(0, 0, 0),
        longest_dimension=None,
    ):
        """Fit bounds in metres, rest on a surface; preserve referenced meshes/materials."""
        import numpy as np
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        root = UsdGeom.Xform.Define(self.stage, "/World/" + name)
        orient = UsdGeom.Xform.Define(self.stage, str(root.GetPath()) + "/Orientation")
        orient.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        model = add_reference_to_stage(
            str(self.asset(relative)), str(orient.GetPath()) + "/Model"
        )
        bounds = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            .ComputeWorldBound(model)
            .ComputeAlignedRange()
        )
        low, high = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        size = high - low
        if not np.all(np.isfinite(size)) or np.any(size <= 0):
            raise ValueError(f"Invalid bounds for {relative}: {size}")
        scale = (
            np.array(dimensions) / size
            if dimensions
            else np.full(
                3,
                longest_dimension / max(size)
                if longest_dimension is not None
                else height / size[2],
            )
        )
        center = (low + high) / 2 * scale
        root.AddTranslateOp().Set(
            Gf.Vec3d(
                float(xy[0] - center[0]),
                float(xy[1] - center[1]),
                float(bottom - low[2] * scale[2] + (0.002 if dynamic else 0)),
            )
        )
        root.AddScaleOp().Set(Gf.Vec3d(*scale))
        # Remove imported rigid-body behavior on descendants so there is one body per prop.
        for prim in Usd.PrimRange(model):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                prim.RemoveAPI(UsdPhysics.MassAPI)
            if prim.IsA(UsdGeom.Mesh):
                normalize_mesh_normals(prim)
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                    "convexHull"
                )
        if dynamic:
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(
                True
            )
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.15)
        self.objects[name] = {
            "xy": list(xy),
            "height": float(size[2] * scale[2]),
            "bottom": bottom,
            "rotation_xyz_deg": list(rotation),
            "scale_xyz": scale.tolist(),
            "fitted_dimensions_m": (size * scale).tolist(),
            "dynamic": dynamic,
            "asset": str(relative),
        }

    def step(self, count=1):
        for _ in range(count):
            if not self.app.is_running():
                raise RuntimeError("Simulation window closed")
            self.tick += 1
            self.world.step(render=self.tick % 4 == 0)

    def step_seconds(self, seconds):
        self.step(max(1, math.ceil(seconds / self.world.get_physics_dt())))

    def save(self, output, screenshot=False):
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        self.stage.GetRootLayer().Export(str(output / "tabletop.usda"))
        (output / "scene.json").write_text(
            json.dumps(
                {
                    "runtime": self.runtime,
                    "assets": str(self.assets),
                    "objects": self.objects,
                    "physics_config": self.config.to_dict(),
                    "joint_names": self.robot.dof_names,
                },
                indent=2,
            )
        )
        if screenshot:
            from PIL import Image

            for _ in range(48):
                self.world.render()
            rgba = self.camera.get_rgba()
            if rgba is None or rgba.size == 0:
                raise RuntimeError("Camera produced no image")
            Image.fromarray(rgba.astype("uint8")).save(output / "preview.png")
        print(f"Saved scene to {output}", flush=True)


def main():
    from simulation import arguments, start_app, keep_running
    from settings import config_from_args

    parser = arguments(__doc__, demo=True)
    args = parser.parse_args()
    config = config_from_args(parser, args)
    app = start_app(args)
    try:
        scene = TabletopScene(app, args.assets, config=config)
        scene.save(args.output, args.screenshot)
        keep_running(scene, args)
    except Exception:
        import traceback

        traceback.print_exc()
        app.app.post_quit(1)
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
