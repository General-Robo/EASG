"""Seeded XY layout sampling and application; no task or output dependencies."""

from dataclasses import asdict, dataclass
import numpy as np
from construct_scene.geometry import world_bounds, world_pose


@dataclass(frozen=True)
class PositionRandomization:
    seed: int
    edge_margin_m: float = 0.05
    min_gap_m: float = 0.03
    max_attempts: int = 500

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("position seed must be an integer in [0, 2**64)")
        for name in ("edge_margin_m", "min_gap_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")

    def to_dict(self):
        return {"enabled": True, **asdict(self)}


def rectangle_gap(a, b):
    """Euclidean minimum edge distance of two closed XY AABBs (zero if touching)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    separation = np.maximum(0.0, np.maximum(a[0] - b[1], b[0] - a[1]))
    return float(np.linalg.norm(separation))


def sample_layout(workspace, object_bounds, goal_bounds, config):
    """Bounds are [[xmin,ymin],[xmax,ymax]], already scaled and world-oriented."""
    rectangles = [
        np.asarray(v, dtype=float) for v in (workspace, object_bounds, goal_bounds)
    ]
    for bounds in rectangles:
        if (
            bounds.shape != (2, 2)
            or not np.all(np.isfinite(bounds))
            or np.any(bounds[1] <= bounds[0])
        ):
            raise ValueError(
                "Each XY rectangle must have finite, positive extents (shape 2x2)"
            )
    workspace, *footprints = rectangles
    inner = workspace + np.array(
        [[config.edge_margin_m] * 2, [-config.edge_margin_m] * 2]
    )
    halves = [(b[1] - b[0]) / 2 for b in footprints]
    intervals = [(inner[0] + h, inner[1] - h) for h in halves]
    for role, (low, high) in zip(("object", "goal"), intervals):
        if np.any(low > high):
            raise ValueError(
                f"{role} footprint does not fit inside workmat with {config.edge_margin_m} m margin"
            )
    rng = np.random.default_rng(config.seed)
    for attempt in range(1, config.max_attempts + 1):
        centers = [rng.uniform(low, high) for low, high in intervals]
        placed = [np.array([c - h, c + h]) for c, h in zip(centers, halves)]
        gap = rectangle_gap(*placed)
        overlap_or_touch = bool(
            np.all(placed[0][0] <= placed[1][1])
            and np.all(placed[1][0] <= placed[0][1])
        )
        if not overlap_or_touch and gap >= config.min_gap_m:
            return {
                "attempts": attempt,
                "rectangle_gap_m": gap,
                "workspace_xy_m": workspace.tolist(),
                "roles": {
                    role: {
                        "original_bounds_xy_m": b.tolist(),
                        "sampled_bounds_xy_m": p.tolist(),
                        "sampled_center_xy_m": c.tolist(),
                        "translation_xy_m": (c - b.mean(axis=0)).tolist(),
                    }
                    for role, b, p, c in zip(
                        ("object", "goal"), footprints, placed, centers
                    )
                },
            }
    raise ValueError(
        f"No non-overlapping layout found after {config.max_attempts} samples "
        f"(seed={config.seed}, edge_margin={config.edge_margin_m}, gap={config.min_gap_m})"
    )


def apply_layout(stage, paths, handles, config):
    """Called after task assets are built, before world.reset / any physics steps."""
    from isaacsim.core.prims import SingleXFormPrim

    if set(paths) != {"workspace", "object", "goal"}:
        raise ValueError(
            "Register workspace, object and goal prim paths for position randomization"
        )
    if len(set(paths.values())) != 3:
        raise ValueError("Layout prims must be distinct")
    for role in ("object", "goal"):
        if any(
            paths[role].startswith(other + "/") or other.startswith(paths[role] + "/")
            for other in paths.values()
            if other != paths[role]
        ):
            raise ValueError("Layout prims must not be nested")
    # A render-only shell may have a guide-purpose parent collider. Sample the
    # full physical workmat rectangle, independently of its visual appearance.
    bounds = {
        role: world_bounds(stage, path, include_guides=True)
        if role == "workspace"
        else world_bounds(stage, path)
        for role, path in paths.items()
    }
    layout = sample_layout(
        bounds["workspace"][:, :2],
        bounds["object"][:, :2],
        bounds["goal"][:, :2],
        config,
    )
    for role in ("object", "goal"):
        entry = layout["roles"][role]
        entry["prim_path"] = paths[role]
        entry["original_pose"] = world_pose(stage, paths[role])
        prim = handles[role] if role in handles else SingleXFormPrim(paths[role])
        position, _ = prim.get_world_pose()
        position = np.array(position, dtype=float)
        position[:2] += entry["translation_xy_m"]
        prim.set_world_pose(position=position)
        # World.reset restores registered prim defaults; update the owning handle too.
        prim.set_default_state(position=position)
        entry["sampled_pose"] = world_pose(stage, paths[role])
        np.testing.assert_allclose(
            entry["sampled_pose"]["position_m"][2],
            entry["original_pose"]["position_m"][2],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            np.array(entry["sampled_pose"]["world_matrix_row_vector"])[:3, :3],
            np.array(entry["original_pose"]["world_matrix_row_vector"])[:3, :3],
            atol=1e-7,
        )
    return {"schema_version": 1, "config": config.to_dict(), "paths": paths, **layout}


def settled_layout(stage, layout):
    """Record naturally settled poses without resampling or moving bodies."""
    for role in ("object", "goal"):
        layout["roles"][role]["settled_pose"] = world_pose(stage, layout["paths"][role])
    return layout
