TCP Z measurement and overlay
=============================

The TCP Z axis is measured independently on its positive and negative halves.
There is no measurement-mode option. This does not change the TCP coordinate
frame, camera pose, action conversion, or scene physics.

Implementation
--------------
frame_drawing.py measures visible RGB-D samples, formats signed distances and
draws the projected XYZ axes. Its drawing function does not access depth or scene geometry.
observation.py captures images/depth and projects the selected surface points.

Measurement
-----------
- Search within a radius of 6 mm around the full TCP Z axis, with absolute axial
  distance below 80 cm. Each side requires at least 3 candidate depth samples.
- Select the minimum absolute axial distance independently on each side.
- Exact zero is represented once in the positive slot, labeled Z=0.
- Unavailable sides have null distance and no endpoint; unknown is not empty.
- Robot/gripper surfaces are not filtered. A negative distance does not prove
  a successful grasp. No hidden surfaces or object identities are consulted.
- These are sampled approximations, not exact mesh/ray intersections. Distances
  are axial projections, not Euclidean distances to the camera or sample.
- Neither side tracks the same target surface across steps.

Observations record tcp_z_distances_m, tcp_z_surface_points_world,
tcp_z_projected_pixels, tcp_z_candidate_counts and tcp_z_parameters.
Historical experiments retain their own snapshots and observation schemas;
use those frozen runtimes when reproducing their original behavior.

Tests (no simulation or model API required):
  ./python.sh -m unittest discover -s script/test
