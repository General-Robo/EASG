"""Pick/place composes a grasp object and an interchangeable placement target."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import numpy as np

from construct_scene.assets import (
    OBJECTS,
    PLATES,
    reference_asset,
    normalize_mesh_normals,
)
from construct_scene.base import BaseTaskScene
from construct_scene.geometry import (
    world_bounds,
    mesh_world_points,
    mesh_world_triangles,
    vertical_surface_height,
)
from construct_scene.layout import apply_layout, settled_layout


class GraspObject(ABC):
    """A task role, not an asset category. A plate could implement this too."""

    label: str
    prim_path: str
    metadata: dict

    @abstractmethod
    def build(self, scene):
        """Set prim_path, handle and metadata before physics initialization."""

    @abstractmethod
    def pose(self):
        pass

    @abstractmethod
    def velocity(self):
        pass

    @abstractmethod
    def points(self):
        pass


class RigidGraspObject(GraspObject):
    def __init__(self, spec):
        self.spec = spec
        self.label = spec.label

    def pose(self):
        return self.handle.get_world_pose()

    def velocity(self):
        return self.handle.get_linear_velocity()

    def points(self):
        return mesh_world_points(self.stage, self.prim_path)

    def build(self, scene):
        self.stage = scene.stage
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
        from isaacsim.core.prims import SingleRigidPrim

        self.prim_path = "/World/" + self.spec.prim_name
        root = UsdGeom.Xform.Define(self.stage, self.prim_path)
        orient = UsdGeom.Xform.Define(self.stage, self.prim_path + "/Orientation")
        recenter = orient.AddTranslateOp()
        recenter.Set(Gf.Vec3d(0, 0, 0))
        # Isaac's reference resolver converts this Y-up asset to Z-up (+90 X).
        # Undo that conversion, then align the scanned banana's long axis with X.
        orient.AddRotateXYZOp().Set(Gf.Vec3f(*self.spec.rotation_xyz_deg))
        asset = scene.asset(self.spec.asset)
        model = reference_asset(str(asset), self.prim_path + "/Orientation/Model")
        bound = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            .ComputeWorldBound(model)
            .ComputeAlignedRange()
        )
        low, high = np.asarray(bound.GetMin()), np.asarray(bound.GetMax())
        size = high - low
        if (
            not np.all(np.isfinite(size))
            or np.any(size <= 0)
            or not np.isfinite(self.spec.length_m)
            or self.spec.length_m <= 0
        ):
            raise ValueError("Object dimensions must be finite and positive")
        scale = self.spec.length_m / max(size)
        recenter.Set(Gf.Vec3d(*(-(low + high) / 2)))
        root.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))
        object_height = float(size[2] * scale)
        # Preserve the approved initial object placement.
        position = np.array(
            [-0.04, -0.12, scene.config.workmat_top_m + object_height / 2 + 0.002]
        )
        for prim in Usd.PrimRange(model):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.HasAPI(UsdPhysics.MassAPI):
                prim.RemoveAPI(UsdPhysics.MassAPI)
            if prim.IsA(UsdGeom.Mesh):
                normalize_mesh_normals(prim)
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                    "convexDecomposition"
                )
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision.CreateContactOffsetAttr(0.002)
                collision.CreateRestOffsetAttr(0.0)
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(self.spec.mass_kg)
        PhysxSchema.PhysxRigidBodyAPI.Apply(
            root.GetPrim()
        ).CreateSolverPositionIterationCountAttr(64)
        # Retain scanned appearance and the asset's physics material (friction 1/1).
        self.handle = scene.world.scene.add(
            SingleRigidPrim(
                self.prim_path,
                name="target_" + self.spec.label,
                position=position,
                mass=self.spec.mass_kg,
            )
        )
        self.metadata = {
            "asset": str(asset),
            "xy": position[:2].tolist(),
            "fitted_dimensions_m": (size * scale).tolist(),
            "scale_xyz": [scale] * 3,
            "rotation_xyz_deg": list(self.spec.rotation_xyz_deg),
            "dynamic": True,
            "mass_kg": self.spec.mass_kg,
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
            "collision": "convexDecomposition",
        }


@dataclass(frozen=True)
class PlacementResult:
    inside: bool
    supported: bool
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    inside_vertex_count: int | None = None
    total_vertex_count: int | None = None

    @property
    def accepted(self):
        return self.inside and self.supported


class PlacementTarget(ABC):
    name: str
    label: str
    region_description: str
    prim_path: str
    metadata: dict
    legacy_layout_filename = None

    @abstractmethod
    def build(self, scene):
        """Set prim_path, handle, metadata, center_xy and support_height."""

    @abstractmethod
    def refresh(self):
        """Refresh geometry after layout application or physics settling."""

    @abstractmethod
    def evaluate_placement(self, points):
        """Evaluate geometry only; grasp/release history belongs to the task."""

    def result_fields(self, placement):
        """Optional target-specific diagnostics; not task success conditions."""
        return {}

    def describe_instruction(self, object_label):
        return (
            f"Pick up the {object_label} from the workmat, lift it at least 8 cm above its starting support, "
            f"and place it at the center of the {self.label}. Release the {object_label} and move the gripper "
            f"away. Confirm that the entire {object_label} rests stably {self.region_description}, with clearance from every edge."
        )


class PlateTarget(PlacementTarget):
    region_description = "inside the plate's rim"
    legacy_layout_filename = "plate_layout.json"

    def __init__(self, spec, xy=(0.15, 0.12)):
        self.spec = spec
        self.name = spec.name
        self.label = "white plate" if spec.name == "plate_white" else "plate"
        self.center_xy = np.asarray(xy, dtype=float)
        self.prim_path = "/World/GoalPlate"

    def build(self, scene):
        self.stage = scene.stage
        from isaacsim.core.prims import SingleXFormPrim
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade

        root = UsdGeom.Xform.Define(self.stage, "/World/GoalPlate")
        translation = root.AddTranslateOp()
        translation.Set(Gf.Vec3d(0))
        scale_op = root.AddScaleOp()
        scale_op.Set(Gf.Vec3d(1))
        orient = UsdGeom.Xform.Define(self.stage, "/World/GoalPlate/Orientation")
        asset, rotation = self.spec.asset, self.spec.rotation_xyz_deg
        orient.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        model = reference_asset(
            str(scene.asset(asset)), "/World/GoalPlate/Orientation/Model"
        )
        bounds = world_bounds(self.stage, str(model.GetPath()))
        size = bounds[1] - bounds[0]
        scale = self.spec.diameter_m / max(size[:2])
        scale_op.Set(Gf.Vec3d(scale))
        position = np.array([*self.center_xy, scene.config.workmat_top_m + 0.0005])
        position[:2] -= bounds[:, :2].mean(axis=0) * scale
        position[2] -= bounds[0, 2] * scale
        translation.Set(Gf.Vec3d(*position))
        meshes = [p for p in Usd.PrimRange(model) if p.IsA(UsdGeom.Mesh)]
        authored = [
            p
            for p in meshes
            if p.HasAPI(UsdPhysics.CollisionAPI)
            and p.GetAttribute("physics:collisionEnabled").Get() is not False
        ]
        collision_paths = {str(p.GetPath()) for p in (authored or meshes)}
        for prim in Usd.PrimRange(model):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.HasAPI(UsdPhysics.MassAPI):
                prim.RemoveAPI(UsdPhysics.MassAPI)
            if prim.IsA(UsdGeom.Mesh):
                normalize_mesh_normals(prim)
                enabled = str(prim.GetPath()) in collision_paths
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(enabled)
                if enabled:
                    # Static triangle meshes retain the concave dish; no convex lid across its bowl.
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                        "none"
                    )
                    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    api.CreateContactOffsetAttr(0.001)
                    api.CreateRestOffsetAttr(0.0)
        if self.spec.friction is not None:
            material = UsdPhysics.MaterialAPI.Apply(
                self.stage.DefinePrim("/World/Looks/PlatePhysics", "Material")
            )
            material.CreateStaticFrictionAttr(self.spec.friction)
            material.CreateDynamicFrictionAttr(self.spec.friction)
            material.CreateRestitutionAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(
                UsdShade.Material(material.GetPrim()),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
        self.handle = SingleXFormPrim(self.prim_path)
        self.triangles = mesh_world_triangles(self.stage, collision_paths)
        self.support_height = vertical_surface_height(self.triangles, self.center_xy)
        self.bounds = world_bounds(self.stage, "/World/GoalPlate")
        self.metadata = {
            "plate": self.spec.name,
            "asset": str(scene.asset(asset)),
            "diameter_m": self.spec.diameter_m,
            "dimensions_m": (size * scale).tolist(),
            "goal_xy_m": self.center_xy.tolist(),
            "surface_center_z_m": self.support_height,
            "bounds_world_m": self.bounds.tolist(),
            "rotation_xyz_deg": rotation,
            "scale": float(scale),
            "static": True,
            "collision": "triangle_mesh",
            "collision_meshes": sorted(collision_paths),
            "placement_region": "object mesh XY vertex fraction inside plate ellipse",
            "min_inside_vertex_fraction": self.spec.min_inside_vertex_fraction,
            "rim_margin_m": self.spec.rim_margin_m,
            "support_height_tolerance_m": self.spec.support_tolerance_m,
            "position_randomization": scene.position_randomization is not None,
        }

    def refresh(self):
        bounds = world_bounds(self.stage, self.prim_path)
        np.testing.assert_allclose(
            bounds[1] - bounds[0], self.bounds[1] - self.bounds[0], atol=1e-7
        )
        self.triangles += bounds.mean(axis=0) - self.bounds.mean(axis=0)
        self.bounds = bounds
        self.center_xy = bounds[:, :2].mean(axis=0)
        self.support_height = vertical_surface_height(self.triangles, self.center_xy)
        self.metadata.update(
            goal_xy_m=self.center_xy.tolist(),
            bounds_world_m=bounds.tolist(),
            surface_center_z_m=self.support_height,
        )

    def evaluate_placement(self, points):
        points = np.asarray(points, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
                or not np.all(np.isfinite(points))):
            raise ValueError("Plate evaluation requires nonempty finite (N, 3) mesh vertices")
        if not 0 < self.spec.min_inside_vertex_fraction <= 1:
            raise ValueError("Plate vertex fraction must be in (0, 1]")
        low, high = points.min(axis=0), points.max(axis=0)
        radius = (self.bounds[1, :2] - self.bounds[0, :2]) / 2 - self.spec.rim_margin_m
        if np.any(radius <= 0):
            raise ValueError("Plate rim margin leaves no placement region")
        inside_count = int(np.count_nonzero(
            np.sum(((points[:, :2] - self.center_xy) / radius) ** 2, axis=1) <= 1
        ))
        inside = inside_count / len(points) >= self.spec.min_inside_vertex_fraction
        supported = bool(
            abs(low[2] - self.support_height) < self.spec.support_tolerance_m
        )
        return PlacementResult(inside, supported, low, high, inside_count, len(points))

    def result_fields(self, placement):
        return dict(
            # Retain the literal 100% diagnostic; success uses inside_target instead.
            fully_inside_plate=placement.inside_vertex_count == placement.total_vertex_count,
            plate_containment_passed=placement.inside,
            plate_inside_vertex_count=placement.inside_vertex_count,
            plate_total_vertex_count=placement.total_vertex_count,
            plate_inside_vertex_fraction=placement.inside_vertex_count / placement.total_vertex_count,
            plate_min_inside_vertex_fraction=self.spec.min_inside_vertex_fraction,
            plate_rim_margin_m=self.spec.rim_margin_m,
            on_plate_height=placement.supported,
            plate_surface_z_m=self.support_height,
            plate_static=True,
        )


class SurfaceTarget(PlacementTarget):
    name = "blue_pad"
    label = "blue square pad"
    region_description = "inside the pad boundary"

    def __init__(self, xy=(0.15, 0.12), dimensions=(0.15, 0.15, 0.004), margin_m=0.003):
        self.center_xy = np.asarray(xy, dtype=float)
        self.dimensions = dimensions
        self.margin_m = margin_m
        self.prim_path = "/World/GoalPad"

    def build(self, scene):
        from isaacsim.core.prims import SingleXFormPrim

        self.stage = scene.stage
        scene.box(
            "GoalPad",
            (*self.center_xy, scene.config.workmat_top_m + self.dimensions[2] / 2),
            self.dimensions,
            scene.material("GoalBlue", (0.015, 0.15, 0.75), roughness=0.85),
        )
        self.handle = SingleXFormPrim(self.prim_path)
        self.metadata = {"goal_asset": self.name, "static": True}
        self.refresh()

    def refresh(self):
        self.bounds = world_bounds(self.stage, self.prim_path)
        self.center_xy = self.bounds[:, :2].mean(axis=0)
        self.support_height = float(self.bounds[1, 2])
        self.metadata.update(
            goal_xy_m=self.center_xy.tolist(),
            bounds_world_m=self.bounds.tolist(),
            surface_center_z_m=self.support_height,
        )

    def evaluate_placement(self, points):
        low, high = points.min(axis=0), points.max(axis=0)
        inside = bool(
            np.all(low[:2] >= self.bounds[0, :2] + self.margin_m)
            and np.all(high[:2] <= self.bounds[1, :2] - self.margin_m)
        )
        return PlacementResult(
            inside, bool(abs(low[2] - self.support_height) < 0.012), low, high
        )

    def result_fields(self, placement):
        return dict(
            fully_inside_pad=placement.inside, on_pad_height=placement.supported
        )


TARGET_NAMES = (*PLATES, "blue_pad")


def make_target(name):
    return SurfaceTarget() if name == "blue_pad" else PlateTarget(PLATES[name])


class PickPlaceScene(BaseTaskScene):
    def __init__(
        self,
        app,
        assets,
        *,
        grasp_object=None,
        placement_target=None,
        config=None,
        position_randomization=None,
    ):
        self.grasp_object = (
            grasp_object
            if grasp_object is not None
            else RigidGraspObject(OBJECTS["banana"])
        )
        self.placement_target = (
            placement_target
            if placement_target is not None
            else make_target("plate_white")
        )
        self.position_randomization = position_randomization
        super().__init__(app, assets, config=config)

    def build_task(self):
        obj, goal = self.grasp_object, self.placement_target
        obj.build(self)
        goal.build(self)
        self.contact_targets = [obj.prim_path]
        self.objects[obj.prim_path.rsplit("/", 1)[-1]] = obj.metadata
        self.objects[goal.prim_path.rsplit("/", 1)[-1]] = goal.metadata
        if self.position_randomization is not None:
            paths = {
                "workspace": "/World/WorkMat",
                "object": obj.prim_path,
                "goal": goal.prim_path,
            }
            self.position_layout = apply_layout(
                self.stage,
                paths,
                {"object": obj.handle, "goal": goal.handle},
                self.position_randomization,
            )
            obj.metadata["xy"] = self.position_layout["roles"]["object"][
                "sampled_center_xy_m"
            ]
            goal.refresh()

    def initialize_task(self):
        self.initial_object = self.grasp_object.pose()[0].copy()
        self.max_lift = 0.0
        self.bilateral_contact_seen = False

    def after_prepare(self):
        obj, goal = self.grasp_object, self.placement_target
        if self.position_randomization is not None:
            layout = settled_layout(self.stage, self.position_layout)
            goal.refresh()
            layout.update(
                lift_reference_position_m=self.initial_object.tolist(),
                goal_xy_m=goal.center_xy.tolist(),
                goal_support_height_m=goal.support_height,
                initial_images="initial/",
            )
            (self.output / "position_layout.json").write_text(
                json.dumps(layout, indent=2) + "\n"
            )
        position, orientation = obj.pose()
        record = dict(
            obj.metadata,
            target_object=obj.label,
            settled_position_world_m=position.tolist(),
            settled_quaternion_wxyz=orientation.tolist(),
            goal_xy_m=goal.center_xy.tolist(),
            goal_size_m=(goal.bounds[1] - goal.bounds[0]).tolist(),
            goal_asset=goal.name,
            goal_support_height_m=goal.support_height,
            position_randomization=self.position_randomization is not None,
        )
        (self.output / "object_layout.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        (self.output / "target_layout.json").write_text(
            json.dumps(goal.metadata, indent=2) + "\n"
        )
        if goal.legacy_layout_filename is not None:
            (self.output / goal.legacy_layout_filename).write_text(
                json.dumps(goal.metadata, indent=2) + "\n"
            )
        if self.position_randomization is not None:
            from observation import capture

            capture(self, self.output / "initial")

    def task_sample(self):
        p, q = self.grasp_object.pose()
        forces = self.contacts()
        lift = float(p[2] - self.initial_object[2])
        self.max_lift = max(self.max_lift, lift)
        if min(forces) > 0.1 and lift > 0.035:
            self.bilateral_contact_seen = True
        # Retain legacy telemetry keys so existing result consumers still work.
        return {
            "time": self.tick / 120,
            "block": p.tolist(),
            "block_quaternion": q.tolist(),
            "tcp": self.tcp_matrix()[:3, 3].tolist(),
            "finger_contact_N": forces,
            "gripper_width": float(
                sum(self.robot.get_joint_positions()[self.finger_indices])
            ),
        }

    def evaluate(self):
        obj, goal = self.grasp_object, self.placement_target
        p = obj.pose()[0]
        speed = float(np.linalg.norm(obj.velocity()))
        width = float(sum(self.robot.get_joint_positions()[self.finger_indices]))
        points = obj.points()
        placement = goal.evaluate_placement(points)
        released = width > self.config.gripper_max_width_m * (13 / 16)
        hand_clear = bool(np.linalg.norm(self.tcp_matrix()[:3, 3] - p) > 0.07)
        success = bool(
            self.max_lift > 0.05
            and self.bilateral_contact_seen
            and placement.accepted
            and released
            and speed < 0.02
        )
        result = {
            "success": success,
            "target_object": obj.label,
            "goal_asset": goal.name,
            "max_lift_m": self.max_lift,
            "bilateral_contact_while_lifted": self.bilateral_contact_seen,
            "initial_block_position": self.initial_object.tolist(),
            "final_block_position": p.tolist(),
            "goal_xy": goal.center_xy.tolist(),
            "goal_error_m": float(np.linalg.norm(p[:2] - goal.center_xy)),
            "released": bool(released),
            "hand_clear": hand_clear,
            "final_speed_m_s": speed,
            "physics_only": True,
            "object_poses_sent_to_vlm": False,
            "inside_target": placement.inside,
            "on_target_height": placement.supported,
            "target_mesh_bounds_min_m": placement.bounds_min.tolist(),
            "target_mesh_bounds_max_m": placement.bounds_max.tolist(),
        }
        result.update(goal.result_fields(placement))
        return result
