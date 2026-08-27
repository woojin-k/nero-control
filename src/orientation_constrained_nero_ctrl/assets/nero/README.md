# AgileX Nero URDF (vendored)

Source: official AgileX repo https://github.com/agilexrobotics/agx_arm_urdf
(`nero/` subtree), commit `f6642ce0d7872c686f29c99e9e10cd23d1d49313`
(2026-06-09, "feat(urdf): 更新 Piper/Nero 法兰与末端执行器模型").
License: see `LICENSE` (copied from the repo root).

Files:

- `urdf/nero_description.urdf` — upstream original; meshes referenced via
  ROS `package://agx_arm_description/...` paths.
- `urdf/nero_description_standalone.urdf` — same URDF with mesh paths
  rewritten to relative `../meshes/...` so it loads standalone
  (PyBullet, Pinocchio, yourdfpy, Isaac, MuJoCo converters) without a ROS
  package. This is the one to use for our IK/simulation work.
- `urdf/nero_with_*.xacro` — upstream variants with gripper / Revo2 flange
  end effectors (still `package://` paths).
- `meshes/*.stl` — collision meshes; `meshes/dae/*.dae` — visual meshes.

Arm: 7-DoF (joint1..joint7, all revolute about local Z), fixed
`world -> base_link`. Verified: XML parses, all mesh references resolve,
joint limits present.
