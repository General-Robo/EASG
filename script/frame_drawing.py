"""TCP frame visualization: surface distances, labels and XYZ image overlays.

Measure the nearest supported RGB-D sample independently on each Z half-axis.
Draw the resulting endpoints and X/Y guides using projected pixel coordinates.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

AXIS_RADIUS_M = .006
MAX_AXIAL_M = .8
MIN_SAMPLES = 3


def axis_coordinates(points, tcp_pose):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (N, 3)')
    T = np.asarray(tcp_pose)
    delta = points - T[:3, 3]
    axial = delta @ T[:3, 2]
    radial = np.linalg.norm(delta - axial[:, None] * T[:3, 2], axis=1)
    return points, axial, radial


def sample_hit(points, axial, selected, count):
    return {'distance_m': float(axial[selected]),
            'surface_point_world': points[selected].tolist(),
            'candidate_count': int(count)}


def pixel_text(pixel):
    return 'Unavailable' if pixel is None else f'({pixel[0]:.1f}, {pixel[1]:.1f})'


PARAMETERS = {'axis_radius_m': AXIS_RADIUS_M, 'max_abs_axial_m': MAX_AXIAL_M,
              'min_samples_per_side': MIN_SAMPLES, 'selection': 'nearest_to_tcp_per_side',
              'zero_assignment': 'positive_slot', 'dead_zone_m': 0.0}
OVERLAY = 'X/Y guides unchanged; independent blue +Z/-Z segments end at each nearest visible depth sample, signed distances from TCP; unavailable sides omitted.'


def measure(points, tcp_pose):
    points, axial, radial = axis_coordinates(points, tcp_pose)
    valid = np.isfinite(axial) & (radial < AXIS_RADIUS_M) & (np.abs(axial) < MAX_AXIAL_M)
    hits = {}
    for side, mask in (('positive', axial >= 0), ('negative', axial < 0)):
        candidates = np.flatnonzero(valid & mask)
        hit = None
        if len(candidates) >= MIN_SAMPLES:
            selected = candidates[np.argmin(np.abs(axial[candidates]))]
            hit = sample_hit(points, axial, selected, len(candidates))
        hits[side] = hit
    return hits


def draw(canvas, projected, sides, hits):
    labels = []
    for side in sides:
        value = hits[side]['distance_m']
        name = '+Z' if side == 'positive' else '-Z'
        labels.append(f'{name} {value * 100:+.2f} cm' if value != 0 else 'Z=0')
    draw_tcp_axes(canvas, projected, z_labels=labels)


def format_depth_info(info):
    distances = info['tcp_z_distances_m']
    lines = []
    for side, label in (('positive', '+Z'), ('negative', '-Z')):
        value = distances[side]
        lines.append(f'Nearest visible surface along {label}: ' +
                     ('Unavailable.' if value is None else f'{value * 100:+.2f} cm.'))
    lines.append('Distances are signed from the TCP origin, independently on each half-axis. '
                 'Unavailable means unknown, not empty space; the corresponding endpoint is omitted.')
    return '\n'.join(lines)


def format_projection_info(info):
    origin = info['tcp_projected_pixels'][0]
    pixels = info['tcp_z_projected_pixels']
    return (f'TCP cross pixel=({origin[0]:.1f}, {origin[1]:.1f}); '
            f'BLUE +Z endpoint pixel={pixel_text(pixels["positive"])}; '
            f'BLUE -Z endpoint pixel={pixel_text(pixels["negative"])}.')


def draw_tcp_axes(canvas, projected, *, z_labels):
    """Draw signed X/Y guides and the supplied finite Z surface segments."""
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    origin = np.asarray(projected[0], dtype=float)
    if not np.all(np.isfinite(origin)):
        return
    width, height = canvas.size
    directions = []
    labels = []
    for index, endpoint in enumerate(projected[1:3]):
        vector = np.asarray(endpoint, dtype=float) - origin
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm < 1e-4:
            # An axis seen exactly end-on has no screen-space direction.
            continue
        direction = vector / norm
        low, high = -np.inf, np.inf
        for axis, limit in enumerate((width, height)):
            if abs(direction[axis]) < 1e-9:
                if not 1 <= origin[axis] <= limit - 2:
                    low, high = 1, 0
                    break
            else:
                ends = sorted(((1 - origin[axis]) / direction[axis],
                               (limit - 2 - origin[axis]) / direction[axis]))
                low, high = max(low, ends[0]), min(high, ends[1])
        if low >= high:
            continue
        color = ('#ff454f', '#24df72', '#359dff')[index]
        overlap = any(abs(np.dot(direction, prior)) > .985 for prior in directions)
        directions.append(direction)
        start, end = origin + low * direction, origin + high * direction
        before_y = canvas.copy() if index == 1 else None
        if overlap:
            # Preserve coincident projections with narrower dashes rather than
            # rotating or offsetting either physical axis.
            for distance in np.arange(0, high - low, 18):
                a = start + direction * distance
                b = start + direction * min(distance + 10, high - low)
                draw.line([tuple(a), tuple(b)], fill=color, width=3)
        else:
            draw.line([tuple(start), tuple(end)], fill=color, width=4)
        # Only fade the Y stroke where it coincides with the finite Z segment.
        # Preserve all line widths, arrows, labels, and the true projected points.
        if before_y is not None:
            for endpoint in projected[3:]:
                if not np.all(np.isfinite(endpoint)):
                    continue
                z_vector = np.asarray(endpoint, dtype=float) - origin
                z_length = np.linalg.norm(z_vector)
                if z_length > 1 and abs(np.dot(direction, z_vector / z_length)) > .985:
                    z_direction = z_vector / z_length
                    yy, xx = np.indices((height, width))
                    dx, dy = xx - origin[0], yy - origin[1]
                    along = dx*z_direction[0] + dy*z_direction[1]
                    across = np.abs(dx*z_direction[1] - dy*z_direction[0])
                    mask = ((along >= 0) & (along <= z_length) & (across <= 4))
                    faded = Image.blend(before_y, canvas, .20)
                    canvas.paste(faded, (0, 0), Image.fromarray(mask.astype('uint8') * 255))
        for sign, distance in ((-1, low), (1, high)):
            if sign * distance <= 0:
                continue
            tip = origin + distance * direction
            outward = sign * direction
            normal = np.array([-outward[1], outward[0]])
            arrow_length, arrow_half_width = (10, 4) if overlap else (16, 7)
            draw.polygon([tuple(tip), tuple(tip - arrow_length*outward + arrow_half_width*normal),
                          tuple(tip - arrow_length*outward - arrow_half_width*normal)], fill=color)
            label = ('+' if sign > 0 else '-') + 'XYZ'[index]
            box = draw.textbbox((0, 0), label, font=font)
            bw, bh = box[2]-box[0]+12, box[3]-box[1]+10
            anchor = tip - 26*outward
            # Put labels of coincident axes on opposite sides of the common line.
            side = 1 if overlap else -1
            if abs(direction[1]) >= abs(direction[0]):
                left = anchor[0]+12 if side > 0 else anchor[0]-bw-12
                top = anchor[1]-bh/2
            else:
                left = anchor[0]-bw/2
                top = anchor[1]+12 if side > 0 else anchor[1]-bh-12
            left = float(np.clip(left, 5, max(5, width-bw-5)))
            top = float(np.clip(top, 5, max(5, height-bh-5)))
            labels.append((left, top, bw, bh, box, label, color))
    # Paper convention: Z terminates at the selected visible depth sample.
    # Never extend Z to an image edge or invent a fixed-length depth estimate.
    for endpoint, label in zip(projected[3:], z_labels):
        if not np.all(np.isfinite(endpoint)):
            continue
        tip = np.asarray(endpoint, dtype=float)
        vector = tip - origin
        length = np.linalg.norm(vector)
        color = '#359dff'
        if length > 1:
            direction = vector / length
            normal = np.array([-direction[1], direction[0]])
            draw.line([tuple(origin), tuple(tip)], fill=color, width=4)
            size = min(14., length * .4)
            draw.polygon([tuple(tip), tuple(tip-size*direction+size*.4*normal),
                          tuple(tip-size*direction-size*.4*normal)], fill=color)
        if 0 <= tip[0] < width and 0 <= tip[1] < height:
            draw.ellipse((tip[0]-3, tip[1]-3, tip[0]+3, tip[1]+3), fill=color)
            box = draw.textbbox((0, 0), label, font=font)
            bw, bh = box[2]-box[0]+12, box[3]-box[1]+10
            left = float(np.clip(tip[0]+12, 5, max(5, width-bw-5)))
            label_y = tip[1]+6 if label.startswith('-Z') else tip[1]-bh-6
            top = float(np.clip(label_y, 5, max(5, height-bh-5)))
            labels.append((left, top, bw, bh, box, label, color))
    for left, top, bw, bh, box, label, color in labels:
        draw.rounded_rectangle((left, top, left+bw, top+bh), radius=5, fill='#202328')
        draw.text((left+6-box[0], top+5-box[1]), label, font=font, fill=color)
    if 0 <= origin[0] < width and 0 <= origin[1] < height:
        x, y = origin
        draw.ellipse((x-5, y-5, x+5, y+5), fill='white', outline='black', width=1)
        draw.ellipse((x-2, y-2, x+2, y+2), fill='black')
