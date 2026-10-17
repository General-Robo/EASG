Default workmat appearance: F | Local Ash 0111.

Approved preview: output/asset_previews/wood_boards_6_20261015/F_local_ash
Source: RoboDojo/Assets/Material/material_0111
The three texture files are unmodified copies of the approved source maps.
Runtime resolves them entirely within EASG/Asset; RoboDojo and /tmp are not needed.

Ash_BaseColor.png: sRGB colour map, 2048 x 2048.
Ash_Normal.png: linear tangent-space normal map, 2048 x 2048.
Ash_ORM.png: packed map, 2048 x 2048; green channel drives roughness.
Rendering is non-metallic, matching the F preview shader.

UVs: per-face box projection, 0.65 m texture period, grain along board X.
Side faces reuse source grain; no separate end-grain reconstruction.
No displacement is applied; normal mapping changes shading only.
Existing dimensions, rounded visual geometry, collision geometry, friction and
compliant-contact parameters remain unchanged. This is a visual wood finish,
not a change to rigid wood contact physics.
