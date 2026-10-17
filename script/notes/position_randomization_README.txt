Position randomization (XY only)

Default: OFF. Fixed object/goal XY, camera and TCP configuration are retained.
run.py pick_place defaults to the static 24 cm white plate.
Select --object banana (default), cucumber or carrot.
Select --target plate_white, google_scan-dish_0079, google_scan-dish_0158, 00000 or blue_pad.
Enable: --position-randomization
Disable explicitly: --no-position-randomization
Reproduce a sampled layout: --position-randomization --position-seed 42
If enabled without a seed, a 64-bit seed is generated and saved.
The seed controls layout sampling, not simulator/GPU numerical determinism.

Example, preview only (no model request), from the EASG directory:
./python.sh script/run.py --headless --task pick_place --object banana --observe-only --position-randomization --position-seed 42 --prompt-template script/prompt/prompt.txt --output output/banana_random_seed42_preview
For other foods, use --object cucumber or --object carrot. For actual experiments, add these two layout
flags to the established experiment command; model, reasoning, temperature and
other experiment settings are independent of this feature.

Rules:
- Read world XY AABBs from the actual scaled/oriented object, goal and workmat.
- Board currently spans X [-0.25,0.35], Y [-0.295,0.295] m (60 x 59 cm).
  Bounds are read from scene geometry, not hardcoded in the sampler.
- Both complete footprints remain within the workmat with a 0.05 m edge margin.
  This is footprint-edge to board-edge distance, not centre to edge.
- Their rectangles must not overlap/touch; shortest Euclidean rectangle-to-
  rectangle distance must be at least 0.03 m (including diagonal separation).
- Sample both XY centers uniformly within their allowed center intervals and
  reject invalid pairs, up to 500 attempts. Impossible layouts raise an error.
- Change only root world X/Y: leave Z, rotation, scale and other objects unchanged.
- Update registered body's default state as well as its pose before world.reset.
- Use existing settling and arm initialization, then record actual poses and lift
  reference. No post-settling rejection/resampling is performed.
- The task calls placement_target.refresh() after layout changes and evaluates actual geometry.
  Plates additionally refresh their translated mesh and use the inner floor,
  not the highest rim, as support height. Plate containment requires at least
  75% of object mesh vertices (XY projection) inside the plate ellipse, including
  its boundary, with zero rim inset. Other grasp/release/stability checks remain.
  Evaluations record the inside/total counts, fraction and threshold;
  fully_inside_plate still means 100%, while inside_target is the 75% criterion.
  Blue-pad success
  uses a 3 mm containment margin. These differ from initial 3 cm separation.

Saved artifacts (when enabled):
- run_config.json: position_randomization.enabled, seed, margins, attempt limit.
- position_layout.json: seed/config, paths, workspace, original and sampled
  rectangles, XY translations, original/sample/settled world poses, lift reference.
  World matrices use USD row-vector convention; positions are metres; quaternion
  is normalized WXYZ with scale removed. Root pose may differ from AABB center.
- initial/: external.png, wrist.png, wrist_tcp.png, observation.json, depth, etc.
- plate_layout.json (plate goals): actual centre, bounds, inner-floor support
  height, asset, scale and static collision settings, synchronized after sampling.
- step_001/: normal first task observation.
Layout/ground-truth records are not inserted into the VLM prompt.
Same geometry/orientation/workspace and seed reproduce the sampled layout across
prompts. Physics settling may have small numerical differences; actual poses
are saved separately. Replay requires the original randomization flags and seed.

Replacing objects or target regions:
- Sampling code is asset/name/size independent; do not hardcode banana dimensions.
- construct_scene/layout.py owns PositionRandomization and the pure sampler.
- apply_layout(stage, paths, handles, config) receives three separate, non-nested
  workspace/object/goal paths and the owning simulation handles. It updates both
  current poses and reset defaults, and returns a record without mutating task state.
- PickPlaceScene composes GraspObject and PlacementTarget implementations from
  construct_scene/pick_place.py. Each supplies prim_path and handle after build().
- The task refreshes target geometry, records settled_layout(), and writes JSON;
  layout.py does not know about success, plate floor heights, or output paths.
- close_laptop has no pick/place goal and rejects this option explicitly.
- Goal/object shapes are treated as XY rectangles even if meshes are curved.
  A target's full AABB is used for initial layout clearance, not automatically
  as its success region. PlacementTarget.evaluate_placement() owns that decision.
