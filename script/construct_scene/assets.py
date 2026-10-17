"""Asset specifications and shared loading primitives; no task evaluation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ObjectSpec:
    label: str
    asset: str
    prim_name: str
    length_m: float = 0.12
    rotation_xyz_deg: tuple = (-90, 0, 90)
    mass_kg: float = 0.1


OBJECTS = {
    "banana": ObjectSpec(
        "banana",
        "objects/rigid/omniobject3d-banana/omniobject3d-banana_001/Aligned_obj.usd",
        "TargetBanana",
    ),
    "carrot": ObjectSpec(
        "carrot",
        "objects/rigid/carrot/omniobject3d-carrot_001/Aligned_obj.usd",
        "TargetCarrot",
    ),
    "cucumber": ObjectSpec(
        "cucumber",
        "objects/rigid/cucumber/omniobject3d-cucumber_015/Aligned_obj.usd",
        "TargetCucumber",
    ),
}


@dataclass(frozen=True)
class PlateSpec:
    name: str
    asset: str
    rotation_xyz_deg: tuple = (0, 0, 0)
    diameter_m: float = 0.24
    rim_margin_m: float = 0.0
    support_tolerance_m: float = 0.012
    friction: float | None = None
    min_inside_vertex_fraction: float = 0.75


PLATES = {
    name: PlateSpec(name, f"objects/rigid/plate/{name}/Aligned_obj.usd")
    for name in ("plate_white", "google_scan-dish_0079", "google_scan-dish_0158")
}
PLATES["00000"] = PlateSpec(
    "00000", "objects/rigid/plate/00000/object.usdz", (90, 0, 0), friction=0.5
)


def resolve_asset(root, relative):
    path = Path(root) / relative
    if not path.is_file():
        raise FileNotFoundError(
            f"Asset not found: {path}. Set --assets or EASG_ASSETS."
        )
    return path


def reference_asset(asset_path, prim_path):
    """Keep Isaac's asset up-axis conversion, including packaged USDZ assets."""
    from isaacsim.core.utils.stage import add_reference_to_stage

    if not Path(asset_path).is_file():
        raise FileNotFoundError(asset_path)
    return add_reference_to_stage(str(asset_path), prim_path)


def normalize_mesh_normals(prim):
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    normals = mesh.GetNormalsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if normals is not None and indices is not None and len(normals) == len(indices):
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
