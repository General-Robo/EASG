"""Capture RGB-D and measure/project visible surfaces along the TCP Z axis."""
import json
import numpy as np
from PIL import Image
import frame_drawing

def capture(scene, folder):
    folder.mkdir(parents=True, exist_ok=True)
    # Update rendered observations while preserving the physics state during API calls.
    for _ in range(8):
        scene.world.render()
    rgb = scene.wrist.get_rgba()[:, :, :3].astype('uint8')
    depth = scene.wrist.get_depth()
    if depth is None or not np.any(np.isfinite(depth) & (depth > 0)):
        raise RuntimeError('Wrist RGB-D camera did not produce valid depth')
    Image.fromarray(rgb).save(folder / 'wrist.png')
    np.save(folder / 'depth_m.npy', depth)
    Image.fromarray(scene.camera.get_rgba()[:, :, :3].astype('uint8')).save(folder / 'external.png')
    T = scene.tcp_matrix()
    yy, xx = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > .015) & (depth < 2.)
    pixels = np.column_stack([xx[valid], yy[valid]])
    points = scene.wrist.get_world_points_from_image_coords(pixels, depth[valid])
    hits = frame_drawing.measure(points, T)
    sides = [side for side in ('positive', 'negative') if hits[side] is not None]
    markers = [T[:3, 3], T[:3, 3] + .05*T[:3, 0], T[:3, 3] + .05*T[:3, 1]]
    markers.extend(hits[side]['surface_point_world'] for side in sides)
    uv = scene.wrist.get_image_coords_from_world_points(np.array(markers))
    canvas = Image.fromarray(rgb)
    frame_drawing.draw(canvas, uv, sides, hits)
    canvas.save(folder / 'wrist_tcp.png')
    # Save and send only the external view, annotated wrist, and raw wrist.
    images = [folder / 'external.png', folder / 'wrist_tcp.png', folder / 'wrist.png']
    width = float(sum(scene.robot.get_joint_positions()[scene.finger_indices]))
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    camera_position, camera_quat = scene.wrist.get_world_pose(camera_axes='ros')
    camera_rotation = quat_to_rot_matrix(camera_quat)
    tcp_camera = camera_rotation.T @ (T[:3, 3] - camera_position)
    tcp_axes_camera = camera_rotation.T @ T[:3, :3]
    info = {'observation_id': folder.name, 'vlm_image_order': [path.name for path in images], 'tcp_world': T.tolist(), 'wrist_K': scene.wrist.get_intrinsics_matrix().tolist(),
            'wrist_resolution': [int(rgb.shape[1]), int(rgb.shape[0])],
            'tcp_in_wrist_camera_m': tcp_camera.tolist(), 'tcp_axes_in_wrist_camera': tcp_axes_camera.tolist(),
            'tcp_z_distances_m': {side: None if hit is None else hit['distance_m'] for side, hit in hits.items()},
            'tcp_z_surface_points_world': {side: None if hit is None else hit['surface_point_world'] for side, hit in hits.items()},
            'tcp_z_projected_pixels': {side: uv[3+sides.index(side)].tolist() if side in sides else None for side in hits},
            'tcp_z_candidate_counts': {side: 0 if hit is None else hit['candidate_count'] for side, hit in hits.items()},
            'tcp_z_parameters': frame_drawing.PARAMETERS,
            'gripper_width_m': width,
            'tcp_height_above_workmat_m': float(T[2, 3] - scene.config.workmat_top_m),
            'tcp_projected_pixels': uv.tolist(),
            'motion_frame': 'paper_tcp_v1',
            'axis_overlay': frame_drawing.OVERLAY,
            'source': 'RGB-D and robot proprioception only'}
    (folder / 'observation.json').write_text(json.dumps(info, indent=2))
    return info, images
