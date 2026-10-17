"""USD geometry queries shared by asset loading, layouts and evaluation."""

import numpy as np


def world_bounds(stage, path, *, include_guides=False):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError(f"Missing layout prim: {path}")
    purposes = (
        ["default", "render", "guide"] if include_guides else ["default", "render"]
    )
    bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
        .ComputeWorldBound(prim)
        .ComputeAlignedRange()
    )
    result = np.array([bounds.GetMin(), bounds.GetMax()], dtype=float)
    if not np.all(np.isfinite(result)) or np.any(result[1] < result[0]):
        raise ValueError(f"Invalid layout bounds: {path}")
    return result


def world_pose(stage, path):
    from pxr import Usd, UsdGeom

    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        stage.GetPrimAtPath(path)
    )
    rotation = matrix.RemoveScaleShear().ExtractRotationQuat()
    return {
        "position_m": list(matrix.ExtractTranslation()),
        "orientation_wxyz": [rotation.GetReal(), *rotation.GetImaginary()],
        "world_matrix_row_vector": np.asarray(matrix).tolist(),
        "bounds_world_m": world_bounds(stage, path).tolist(),
    }


def vertical_surface_height(triangles, xy):
    """Highest mesh intersection on a vertical line; used only for physical evaluation."""
    tri = np.asarray(triangles, dtype=float)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    denominator = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (
        a[:, 1] - c[:, 1]
    )
    valid = np.abs(denominator) > 1e-14
    a, b, c, d = a[valid], b[valid], c[valid], denominator[valid]
    u = (
        (b[:, 1] - c[:, 1]) * (xy[0] - c[:, 0])
        + (c[:, 0] - b[:, 0]) * (xy[1] - c[:, 1])
    ) / d
    v = (
        (c[:, 1] - a[:, 1]) * (xy[0] - c[:, 0])
        + (a[:, 0] - c[:, 0]) * (xy[1] - c[:, 1])
    ) / d
    w = 1 - u - v
    hit = (u >= -1e-7) & (v >= -1e-7) & (w >= -1e-7)
    if not np.any(hit):
        raise ValueError("No plate surface at target center")
    return float(np.max((u * a[:, 2] + v * b[:, 2] + w * c[:, 2])[hit]))


def mesh_world_points(stage, path):
    """All mesh vertices, transformed without substituting an enclosing box."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    points = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if prim.IsA(UsdGeom.Mesh):
            local = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
            points.append(local @ matrix[:3, :3] + matrix[3, :3])
    if not points:
        raise ValueError(f"No mesh vertices at {path}")
    return np.concatenate(points)


def mesh_world_triangles(stage, paths):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    triangles = []
    for path in sorted(paths):
        prim = stage.GetPrimAtPath(path)
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        points = points @ matrix[:3, :3] + matrix[3, :3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        faces, offset = [], 0
        for count in counts:
            face = indices[offset : offset + count]
            offset += count
            faces.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))
        if faces:
            triangles.append(points[np.asarray(faces)])
    if not triangles:
        raise ValueError("No collision triangles for placement target")
    return np.concatenate(triangles)
