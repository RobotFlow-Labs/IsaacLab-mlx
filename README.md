![MLX](assets/mlx_logo_dark.png)

---

# IsaacLab-MLX

[![MLX](https://img.shields.io/badge/backend-MLX-black.svg)](https://github.com/ml-explore/mlx)
[![Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-black.svg)](https://www.apple.com/mac/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

**IsaacLab-MLX** is a public fork of [Isaac Lab](https://github.com/isaac-sim/IsaacLab) focused on making the Isaac Lab workflow runnable on Apple Silicon macOS by replacing the CUDA-first training/runtime slice with `MLX + mac-sim`.

The fork is not trying to recreate full Omniverse, RTX, or PhysX GPU parity on macOS. The practical goal is narrower and more useful: preserve the high-level Isaac Lab task and training ergonomics wherever possible, introduce clean backend seams for compute and simulation, and build a Mac-native path that can grow task by task.

## Why This Fork Exists

Upstream Isaac Lab is built around NVIDIA Isaac Sim and a CUDA-oriented runtime. That is the right choice for Linux and NVIDIA GPU deployments, but it leaves Apple Silicon users without a realistic local development path.

This fork exists to close that gap:

- keep Isaac Lab recognizable to existing users
- run the learning stack on Apple Silicon with [MLX](https://github.com/ml-explore/mlx)
- replace import-time CUDA and Omniverse crashes with explicit backend capability checks
- build a Mac-native simulator adapter for the first useful environments instead of waiting for full engine parity

## Current Status

What works today:

- runtime/backend selection seam for `torch-cuda|mlx` and `isaacsim|mac-sim`
- lazy import boundaries in `envs`, `sim`, `assets`, `sensors`, `managers`, `controllers`, `devices`, `scene`, and `markers` so the macOS path fails explicitly instead of exploding on missing `omni.*`
- reproducible source bootstrap script for upstream repositories
- a runnable `MLX + mac-sim` cartpole vertical slice
- cartpole showcase variants covering Box, Discrete, MultiDiscrete, Tuple, and Dict spaces
- a runnable `MLX + mac-sim` cart-double-pendulum MARL slice with dict actions/observations/rewards
- a runnable `MLX + mac-sim` quadcopter slice with root-state dynamics
- a runnable `MLX + mac-sim` ANYmal-C flat locomotion slice with contact-aware rewards, resets, and training smoke
- a runnable `MLX + mac-sim` ANYmal-C rough locomotion slice with procedural wave terrain, analytic terrain raycasts, and deterministic replay coverage
- a runnable `MLX + mac-sim` H1 flat locomotion slice with contact-aware rewards, resets, and training smoke
- a runnable `MLX + mac-sim` H1 rough locomotion slice with procedural wave terrain, analytic height scans, deterministic replay coverage, and corrected rough observation sizing for the shared H1 policy path
- trainable rough locomotion slices for ANYmal-C and H1 with shared PPO/checkpoint contracts, replay support, and CI smoke coverage
- a Metal-backed locomotion root-step helper for the ANYmal-C and H1 slices, with benchmark and semantic-drift diagnostics that report `hotpath: "mlx-metal-root-step"` when that narrow kernel is active
- trainable `MLX + mac-sim` Franka reach, cube-lift, teddy-bear lift, instance-randomized two-cube stack, two-cube stack, three-cube stack, bin-anchored three-cube stack, cabinet-drawer, and open-drawer slices with deterministic analytic kinematics, lightweight grasp/open/stack logic, and family-specific benchmark diagnostics that now distinguish `hotpath: "mlx-metal-ee"` for the shared Franka end-effector path, `hotpath: "mlx-metal-franka-stack"` for the two-cube stack seam, and `hotpath: "mlx-metal-franka-stack-rgb"` for the three-cube/bin-stack seam when those narrow Metal helpers are available
- trainable `MLX + mac-sim` OpenArm reach, bimanual reach, cube-lift, and open-drawer slices with explicit reduced-contract metadata, deterministic analytic surrogate kinematics, and replay/checkpoint support through the same public MLX wrapper and installed CLI surface
- trainable `MLX + mac-sim` UR10 reach and UR10e deploy-reach slices with deterministic analytic pose tracking, reduced-contract metadata (`semantic_contract="reduced-analytic-pose"`), benchmark coverage, and replay/checkpoint support through the same public MLX wrapper and installed CLI surface
- trainable `MLX + mac-sim` UR10e gear-assembly slices for the Robotiq 2F-140 and 2F-85 grippers with explicit reduced-contract metadata (`semantic_contract="reduced-analytic-assembly"`), insertion-aware policy observations, benchmark coverage, and replay/checkpoint support through the same public MLX wrapper and installed CLI surface
- upstream-compatible Franka reach/lift/stack/open-drawer controller variants now resolve through the lazy task registry, public `isaaclab_rl.mlx` wrapper, and installed MLX CLI onto the existing reduced mac-native slices instead of failing on macOS import/runtime seams
- upstream-compatible `Isaac-Deploy-Reach-UR10e-v0`, `Isaac-Deploy-Reach-UR10e-Play-v0`, and `Isaac-Deploy-Reach-UR10e-ROS-Inference-v0` identifiers now resolve through the same lazy registry, public wrapper, and installed CLI onto the canonical `ur10e-deploy-reach` slice while keeping the reduced-contract metadata explicit
- upstream-compatible `Isaac-Deploy-GearAssembly-UR10e-2F140-v0`, `Isaac-Deploy-GearAssembly-UR10e-2F140-Play-v0`, `Isaac-Deploy-GearAssembly-UR10e-2F85-v0`, and `Isaac-Deploy-GearAssembly-UR10e-2F85-Play-v0` identifiers now resolve through the same lazy registry, public wrapper, and installed CLI onto the canonical `ur10e-gear-assembly-2f140` and `ur10e-gear-assembly-2f85` slices while keeping the reduced-contract metadata explicit
- upstream-compatible `Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0`, `Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0`, `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0`, and `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0` now resolve through the same lazy registry, public wrapper, and installed CLI onto the canonical `franka-stack` or `franka-stack-rgb` slices with explicit reduced-contract metadata instead of pretending blueprint / skillgen / visuomotor / Cosmos subsystem parity; the upstream `Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0` alias and the reduced pick-place surrogates `Isaac-PickPlace-GR1T2-Abs-v0`, `Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0`, `Isaac-PickPlace-G1-InspireFTP-Abs-v0`, `Isaac-NutPour-GR1T2-Pink-IK-Abs-v0`, and `Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0` remain explicit reduced-contract exceptions instead of pretending scene parity
- a first mac-native analytic terrain raycast / height-scan sensor substrate for locomotion tasks
- eval-only synthetic cartpole RGB/depth camera slices with deterministic analytic `100x100` observations, upstream-aligned reset ranges, and sensor benchmark coverage
- a backend-local macOS external stereo camera discovery/capture path for UVC devices such as ZED 2i, including a live Terminal-hosted `zed-sdk-mlx` validation path for macOS TCC-safe raw capture
- a basic MLX stereo/depth smoke path on raw side-by-side YUYV dumps
- MLX training, checkpoint save/load, and replay scripts
- a public `isaaclab_rl.mlx` wrapper surface for the current MLX/mac task set
- a runtime diagnostics surface that reports the supported public task manifest, backend seams, honest mac capability metadata, and the concrete articulated `mac-sim` contract without overstating full Isaac Sim parity
- a shared MLX PPO helper substrate for checkpoint metadata, GAE, advantage normalization, and resume-hidden-dim recovery
- a shared checkpoint sidecar schema with metadata versioning, task IDs, policy distribution tags, explicit env-vs-policy action-space fields, and rough-task replay config inference
- a planner compatibility seam for `mac-planners` with deterministic joint-space interpolation, timed waypoint payloads, richer obstacle metadata, and world-state updates
- a plain ROS 2 compatibility bridge for JSONL message transport, planner-world envelopes, joint-trajectory envelopes, and optional `ros2 topic pub/echo` command construction without CUDA assumptions
- portability guards for optional `torch`/`warp` utility imports on macOS
- smoke tests for the backend seam and mac-native task slices
- a maintained kernel inventory for the next Warp/CUDA families still blocking broader parity

What this does not claim yet:

- full Isaac Sim compatibility on macOS
- RTX sensors, Omniverse camera modules, Warp kernels, cuRobo, or the full task zoo
- Isaac ROS acceleration or CUDA transport parity

## MLX Port Architecture

The fork is organized around two explicit seams:

### Compute backend

The compute backend isolates tensor/runtime concerns:

- device selection
- checkpoint save/load
- RNG and backend metadata
- room for MLX-native custom kernel paths

Current public options:

- `torch-cuda`: upstream Isaac Sim path
- `mlx`: Apple Silicon path

Implementation entrypoint:

- [`source/isaaclab/isaaclab/backends/runtime.py`](source/isaaclab/isaaclab/backends/runtime.py)

### Simulation backend

The simulation backend isolates the minimum simulator contract Isaac Lab environments need:

- `reset(soft=...)`
- `step(render=..., update_fabric=...)`
- joint state reads
- joint effort writes
- root/joint state writes
- backend capability reporting

Current public options:

- `isaacsim`: upstream runtime adapter
- `mac-sim`: Mac-native adapter path

The current `mac-sim` implementation is still intentionally narrower than upstream Isaac Sim, but it is no longer only task-local slices. It now includes a shared generic batched articulation/scene substrate for reset/step plus joint/root-state IO, and it currently powers 26 current mac-native rollout tasks: cartpole, cart-double-pendulum, quadcopter, ANYmal-C flat, ANYmal-C rough, H1 flat, H1 rough, Franka reach, OpenArm reach, OpenArm bimanual reach, UR10 reach, UR10e deploy-reach, UR10e gear-assembly 2F-140, UR10e gear-assembly 2F-85, UR10 long-suction stack, UR10 short-suction stack, Franka lift, OpenArm lift, Franka teddy-bear lift, Franka stack instance-randomize, Franka stack, Franka stack RGB, Franka bin-stack, Franka cabinet, Franka open-drawer, and OpenArm open-drawer, plus synthetic cartpole RGB/depth camera variants for 28 public MLX/mac task IDs overall. Twenty-four of those public tasks are trainable end-to-end on the MLX/mac path. The task capability matrix below is the authoritative public support surface.

### Kernel backend

The kernel backend isolates Warp and future Metal custom-kernel paths:

- `warp`: upstream Isaac Sim + Warp path
- `metal`: Apple Silicon path for MLX + Metal-backed kernel replacements
- `cpu`: correctness fallback for bring-up and unsupported kernels

Today the public MLX tasks mostly use pure MLX ops plus compiled MLX helpers. The explicit kernel selection seam now has narrow Metal-backed helpers on both the Franka end-effector path and the shared locomotion root-step path, so future kernels can keep landing without rewriting task-facing APIs again.

For the current locomotion slices, the shared root-step integrator now runs through a Metal-backed MLX kernel and benchmark diagnostics surface that reports `hotpath: "mlx-metal-root-step"` when that narrow helper is active, while the remaining locomotion contact/support helpers stay on compiled MLX. For the Franka manipulation slices, the shared analytic end-effector helper still reports `hotpath: "mlx-metal-ee"` for reach/lift/cabinet/open-drawer tasks, while the two-cube stack seam reports `hotpath: "mlx-metal-franka-stack"` and the three-cube/bin-stack seam reports `hotpath: "mlx-metal-franka-stack-rgb"` when those dedicated Metal helpers are active.

## MLX Quick Start

This path is for Apple Silicon macOS.

### Install Matrix

Use the smallest extra set that matches the workflow you actually need.

| Workflow | Command |
| --- | --- |
| MLX/mac core runtime | `uv pip install --python .venv/bin/python -e source/isaaclab[macos-mlx,dev]` |
| MLX/mac core runtime + task registry | `uv pip install --python .venv/bin/python -e source/isaaclab[macos-mlx,dev] -e source/isaaclab_tasks` |
| MLX/mac core runtime + optional RL wrappers | `uv pip install --python .venv/bin/python -e source/isaaclab[macos-mlx,dev] -e source/isaaclab_rl[dev]` |
| Upstream CUDA / Isaac Sim runtime | `uv pip install --python .venv/bin/python -e source/isaaclab[cuda-isaacsim,dev]` |
| Optional vision helpers | `uv pip install --python .venv/bin/python -e source/isaaclab[vision]` |
| Optional terrain viewer helpers | `uv pip install --python .venv/bin/python -e source/isaaclab[terrain-viewer]` |
| Optional teleop devices | `uv pip install --python .venv/bin/python -e source/isaaclab[teleop]` |
| Optional ONNX export helpers | `uv pip install --python .venv/bin/python -e source/isaaclab[onnx-export]` |
| Optional livestream/web helpers | `uv pip install --python .venv/bin/python -e source/isaaclab[livestream]` |
| Optional Pink IK helpers | `uv pip install --python .venv/bin/python -e source/isaaclab[pink-ik]` |
| Optional OpenXR retargeting helpers | `uv pip install --python .venv/bin/python -e source/isaaclab[openxr-retargeting]` |
| Optional IsaacLab task extras for CUDA AutoMate | `uv pip install --python .venv/bin/python -e source/isaaclab_tasks[cuda-automate]` |
| Optional RL logging/video extras | `uv pip install --python .venv/bin/python -e source/isaaclab_rl[rl-logging,video]` |

If you want one public MLX/mac bootstrap command instead of a manual install sequence:

```bash
uv run scripts/bootstrap_uv_mlx.py
```

That creates `.venv`, installs the core MLX/mac package plus the lazy task registry and public MLX wrapper, and installs the release-facing console entry points:

- `.venv/bin/isaaclab-mlx train`
- `.venv/bin/isaaclab-mlx evaluate`
- `.venv/bin/isaaclab-mlx-train`
- `.venv/bin/isaaclab-mlx-evaluate`
- `.venv/bin/isaaclab-mlx-runtime-diagnostics`

### Public Support Matrix

This is the current public support contract for runtime combinations, not just what can be installed.

| Platform / Runtime | Status | Notes |
| --- | --- | --- |
| Apple Silicon + `mlx` + `metal` + `mac-sim` | Supported | Shared generic batched articulation/root-state substrate plus the current mac-native task slice: cartpole, cart-double-pendulum, quadcopter, ANYmal-C flat, ANYmal-C rough, H1 flat, H1 rough, Franka reach, OpenArm reach, OpenArm bimanual reach, UR10 reach, UR10e deploy-reach, UR10e gear-assembly 2F-140, UR10e gear-assembly 2F-85, Franka lift, OpenArm lift, Franka teddy-bear lift, Franka stack instance-randomize, Franka stack, Franka stack RGB, Franka bin-stack, Franka cabinet, Franka open-drawer, OpenArm open-drawer, and synthetic cartpole RGB/depth camera tasks |
| Apple Silicon + `mlx` + `cpu` + `mac-sim` | Supported for correctness/debug | Useful for bring-up only, not benchmark claims |
| Linux/NVIDIA + `torch-cuda` + `warp` + `isaacsim` | Supported reference path | Upstream-compatible CUDA / Isaac Sim runtime |
| Apple Silicon + `isaacsim` runtime | Unsupported | This fork does not ship Isaac Sim / Omniverse parity on macOS |
| Apple Silicon + backend-local external stereo capture | Supported prototype | AVFoundation discovery plus raw stereo dump and MLX depth smoke; live ZED/ZED 2i capture is supported through a camera-authorized Terminal host with `zed-sdk-mlx`, not as Isaac Sim camera parity |
| Apple Silicon + `mac-planners` planner seam | Supported prototype | Deterministic joint-space interpolation with timed waypoints plus richer box/sphere/capsule/mesh obstacle metadata; not cuRobo parity |
| Apple Silicon + plain ROS 2 message/process bridge | Supported prototype | JSONL bridge, planner-world / joint-trajectory envelope builders, and optional `ros2` CLI command builder without CUDA/NITROS |
| macOS RTX / Omniverse UI / Kit extensions | Unsupported | Explicit capability-gated failures only |
| macOS planners / cuRobo / Isaac ROS CUDA transport | Reference-only / deferred | Not part of the current public MLX support surface |

### Import-Safe Surface On macOS

These imports are now part of the tested MLX/mac bootstrap surface and are expected to work without Isaac Sim,
Warp, or torch installed:

- `isaaclab`, `isaaclab.envs`, `isaaclab.sim`, `isaaclab.sim.schemas`, `isaaclab.sim.converters`, `isaaclab.sim.spawners.from_files`
- `isaaclab.utils.io`, `isaaclab.utils.noise`, `isaaclab.utils.types`, `isaaclab.utils.modifiers`, `isaaclab.utils.interpolation`
- `isaaclab.markers`, `isaaclab.markers.config`
- `isaaclab.devices.openxr`
- `isaaclab.sensors.camera`, `isaaclab.sensors.ray_caster`, `isaaclab.sensors.ray_caster.patterns`
- `isaaclab_tasks`, `isaaclab_rl.mlx`, `isaaclab_rl.sb3`, `isaaclab_rl.skrl`

On the mac path, configuration helpers such as `ViewerCfg`, `VisualizationMarkersCfg`, `XrCfg`,
`remove_camera_configs`, `CameraCfg`, `RayCasterCfg`, and the lazy task registry stay available. Runtime-only
Isaac Sim objects continue to fail through explicit backend checks when they are actually requested.
For sensors specifically, the import-safe `isaaclab.sensors.camera` and `isaaclab.sensors.ray_caster` surfaces are
currently config-oriented on macOS; the supported runtime sensor path today is the backend-local analytic terrain height scan in
[`source/isaaclab/isaaclab/backends/mac_sim/sensors.py`](source/isaaclab/isaaclab/backends/mac_sim/sensors.py).
The generic `mac-sensors` backend now intentionally reports:

- `raycast = true`
- `analytic_camera_tasks = true`
- `external_stereo_capture = true`
- `cameras = false`, `depth = false`, `rgb = false`

That is deliberate. Synthetic cartpole camera tasks and backend-local ZED/UVC capture are supported, but they are not being presented as generic Isaac Sim camera parity.

### 1. Create the environment with `uv`

```bash
cd IsaacLab
uv run scripts/bootstrap_uv_mlx.py
```

If you want the upstream Isaac Sim runtime on Linux/NVIDIA instead, install the CUDA-side extra:

```bash
uv pip install --python .venv/bin/python -e source/isaaclab[cuda-isaacsim,dev]
```

### 2. Train the MLX cartpole baseline

```bash
.venv/bin/isaaclab-mlx-train \
  --task cartpole \
  --num-envs 256 \
  --updates 200 \
  --rollout-steps 64 \
  --epochs-per-update 4 \
  --checkpoint logs/mlx/cartpole_policy.npz
```

### 3. Replay a trained checkpoint

```bash
.venv/bin/isaaclab-mlx-evaluate \
  --task cartpole \
  --checkpoint logs/mlx/cartpole_policy.npz \
  --episodes 3
```

If you want the combined installed CLI instead of the split entry points:

```bash
.venv/bin/isaaclab-mlx train --task cartpole --updates 10
.venv/bin/isaaclab-mlx evaluate --task h1-rough --episodes 1
.venv/bin/isaaclab-mlx-runtime-diagnostics logs/runtime/runtime-diagnostics.json
```

Optional resume flow:

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/train_cartpole.py \
  --resume-from logs/mlx/cartpole_policy.npz \
  --checkpoint logs/mlx/cartpole_policy_resumed.npz \
  --updates 50
```

The torch-centric RL wrapper extension is now optional. If you need upstream SB3, rl-games, or RSL-RL wrappers, install them separately from the MLX path:

```bash
uv pip install --python .venv/bin/python -e source/isaaclab_rl[sb3]
uv pip install --python .venv/bin/python -e source/isaaclab_rl[rsl-rl]
```

For the MLX/mac-sim task slices documented in this fork, `source/isaaclab_rl` is not required.

### 4. Run cart-double-pendulum MARL smoke

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/evaluate_task.py \
  --task cart-double-pendulum \
  --num-envs 64 --episodes 3 --max-steps 10000 --random-actions
```

### 5. Run quadcopter smoke

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/evaluate_task.py \
  --task quadcopter \
  --num-envs 64 --episodes 3 --episode-length-s 0.5 --max-steps 10000 --thrust-action 0.2 --no-random-actions
```

### 6. Train and replay the ANYmal-C flat slice

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/train_task.py \
  --task anymal-c-flat \
  --num-envs 256 \
  --updates 10 \
  --rollout-steps 24 \
  --epochs-per-update 2 \
  --checkpoint logs/mlx/anymal_c_flat_policy.npz
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/evaluate_task.py \
  --task anymal-c-flat \
  --checkpoint logs/mlx/anymal_c_flat_policy.npz \
  --episodes 3
```

### 7. Train and replay the H1 flat slice

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/train_task.py \
  --task h1-flat \
  --num-envs 256 \
  --updates 10 \
  --rollout-steps 24 \
  --epochs-per-update 2 \
  --checkpoint logs/mlx/h1_flat_policy.npz
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/evaluate_task.py \
  --task h1-flat \
  --checkpoint logs/mlx/h1_flat_policy.npz \
  --episodes 3
```

### 8. Use the shared MLX task CLIs

The fork now exposes one shared train entrypoint and one shared evaluation/replay entrypoint for the current MLX/mac-sim slices:

- [`scripts/reinforcement_learning/mlx/train_task.py`](scripts/reinforcement_learning/mlx/train_task.py)
- [`scripts/reinforcement_learning/mlx/evaluate_task.py`](scripts/reinforcement_learning/mlx/evaluate_task.py)

The task-specific scripts remain as thin wrappers so existing commands still work, but the shared CLIs are the stable surface going forward.

The same task family is also available as a public Python API through [`source/isaaclab_rl/isaaclab_rl/mlx.py`](source/isaaclab_rl/isaaclab_rl/mlx.py):

```python
from isaaclab_rl.mlx import evaluate_mlx_task, train_mlx_task

train_payload = train_mlx_task("anymal-c-flat", num_envs=128, updates=10)
eval_payload = evaluate_mlx_task("h1-flat", num_envs=32, episodes=3)
```

Additional task slices exposed through the same API:

```python
rough_payload = evaluate_mlx_task("anymal-c-rough", num_envs=32, episodes=2)
h1_rough_payload = evaluate_mlx_task("h1-rough", num_envs=32, episodes=2)
rgb_payload = evaluate_mlx_task("cartpole-rgb-camera", num_envs=32, episodes=2)
depth_payload = evaluate_mlx_task("cartpole-depth-camera", num_envs=32, episodes=2)
reach_train_payload = train_mlx_task("franka-reach", num_envs=64, updates=5)
reach_payload = evaluate_mlx_task("franka-reach", checkpoint=reach_train_payload["checkpoint_path"], episodes=2)
ur10e_train_payload = train_mlx_task("ur10e-deploy-reach", num_envs=64, updates=5)
ur10e_payload = evaluate_mlx_task("ur10e-deploy-reach", checkpoint=ur10e_train_payload["checkpoint_path"], episodes=2)
gear_2f140_train_payload = train_mlx_task("ur10e-gear-assembly-2f140", num_envs=64, updates=5)
gear_2f140_payload = evaluate_mlx_task("Isaac-Deploy-GearAssembly-UR10e-2F140-Play-v0", checkpoint=gear_2f140_train_payload["checkpoint_path"], episodes=2)
gear_2f85_train_payload = train_mlx_task("ur10e-gear-assembly-2f85", num_envs=64, updates=5)
gear_2f85_payload = evaluate_mlx_task("Isaac-Deploy-GearAssembly-UR10e-2F85-Play-v0", checkpoint=gear_2f85_train_payload["checkpoint_path"], episodes=2)
lift_train_payload = train_mlx_task("franka-lift", num_envs=64, updates=5)
lift_payload = evaluate_mlx_task("franka-lift", checkpoint=lift_train_payload["checkpoint_path"], episodes=2)
stack_train_payload = train_mlx_task("franka-stack", num_envs=64, updates=5)
stack_payload = evaluate_mlx_task("franka-stack", checkpoint=stack_train_payload["checkpoint_path"], episodes=2)
stack_rgb_train_payload = train_mlx_task("franka-stack-rgb", num_envs=64, updates=5)
stack_rgb_payload = evaluate_mlx_task("franka-stack-rgb", checkpoint=stack_rgb_train_payload["checkpoint_path"], episodes=2)
bin_stack_train_payload = train_mlx_task("franka-bin-stack", num_envs=64, updates=5)
bin_stack_payload = evaluate_mlx_task("franka-bin-stack", checkpoint=bin_stack_train_payload["checkpoint_path"], episodes=2)
cabinet_train_payload = train_mlx_task("franka-cabinet", num_envs=64, updates=5)
cabinet_payload = evaluate_mlx_task("franka-cabinet", checkpoint=cabinet_train_payload["checkpoint_path"], episodes=2)
```

### 9. Train and replay the current trainable manipulation slices

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/train_franka_reach.py \
  --num-envs 128 \
  --updates 10 \
  --rollout-steps 24 \
  --epochs-per-update 2 \
  --checkpoint logs/mlx/franka_reach_policy.npz
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/play_franka_reach.py \
  --checkpoint logs/mlx/franka_reach_policy.npz \
  --episodes 3
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/train_ur10e_deploy_reach.py \
  --num-envs 128 \
  --updates 10 \
  --rollout-steps 24 \
  --epochs-per-update 2 \
  --checkpoint logs/mlx/ur10e_deploy_reach_policy.npz
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/play_ur10e_deploy_reach.py \
  --checkpoint logs/mlx/ur10e_deploy_reach_policy.npz \
  --episodes 3
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/train_franka_lift.py \
  --num-envs 128 \
  --updates 10 \
  --rollout-steps 24 \
  --epochs-per-update 2 \
  --checkpoint logs/mlx/franka_lift_policy.npz
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/play_franka_lift.py \
  --checkpoint logs/mlx/franka_lift_policy.npz \
  --episodes 3
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/train_franka_teddy_bear_lift.py \
  --num-envs 128 \
  --updates 10 \
  --rollout-steps 24 \
  --epochs-per-update 2 \
  --checkpoint logs/mlx/franka_teddy_bear_lift_policy.npz
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/play_franka_teddy_bear_lift.py \
  --checkpoint logs/mlx/franka_teddy_bear_lift_policy.npz \
  --episodes 3
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/train_franka_stack_instance_randomize.py \
  --num-envs 128 \
  --updates 10 \
  --rollout-steps 24 \
  --epochs-per-update 2 \
  --checkpoint logs/mlx/franka_stack_instance_randomize_policy.npz
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/play_franka_stack_instance_randomize.py \
  --checkpoint logs/mlx/franka_stack_instance_randomize_policy.npz \
  --episodes 3
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/train_franka_stack.py \
  --num-envs 128 \
  --updates 10 \
  --rollout-steps 24 \
  --epochs-per-update 2 \
  --checkpoint logs/mlx/franka_stack_policy.npz
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/play_franka_stack.py \
  --checkpoint logs/mlx/franka_stack_policy.npz \
  --episodes 3
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/train_franka_stack_rgb.py \
  --num-envs 128 \
  --updates 10 \
  --rollout-steps 24 \
  --epochs-per-update 2 \
  --checkpoint logs/mlx/franka_stack_rgb_policy.npz
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/play_franka_stack_rgb.py \
  --checkpoint logs/mlx/franka_stack_rgb_policy.npz \
  --episodes 3
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/train_franka_bin_stack.py \
  --num-envs 128 \
  --updates 10 \
  --rollout-steps 24 \
  --epochs-per-update 2 \
  --checkpoint logs/mlx/franka_bin_stack_policy.npz
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/play_franka_bin_stack.py \
  --checkpoint logs/mlx/franka_bin_stack_policy.npz \
  --episodes 3
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/train_franka_cabinet.py \
  --num-envs 128 \
  --updates 10 \
  --rollout-steps 24 \
  --epochs-per-update 2 \
  --checkpoint logs/mlx/franka_cabinet_policy.npz
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/play_franka_cabinet.py \
  --checkpoint logs/mlx/franka_cabinet_policy.npz \
  --episodes 3
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/train_franka_open_drawer.py \
  --num-envs 128 \
  --updates 10 \
  --rollout-steps 24 \
  --epochs-per-update 2 \
  --checkpoint logs/mlx/franka_open_drawer_policy.npz
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/play_franka_open_drawer.py \
  --checkpoint logs/mlx/franka_open_drawer_policy.npz \
  --episodes 3
```

### 10. Probe the backend-local mac camera path

The fork now also ships backend-local macOS camera tools for external stereo devices. These do not depend on Isaac Sim or the Omniverse camera stack.

- [`scripts/tools/probe_mac_camera.py`](scripts/tools/probe_mac_camera.py)
- [`scripts/tools/mac_stereo_depth_smoke.py`](scripts/tools/mac_stereo_depth_smoke.py)

Live ZED/ZED 2i capture on macOS is host-app sensitive because Camera permission is attributed by TCC to the host app, not just the child Python process. The supported live path is a camera-authorized Terminal host plus [`zed-sdk-mlx`](https://github.com/RobotFlow-Labs/zed-sdk-mlx). The current `zed-sdk-mlx` capture helper is fixed to raw `2560x720 @ 30 FPS`.

Device discovery:

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/tools/probe_mac_camera.py \
  --json-out logs/hardware/mac_camera_probe.json
```

Live ZED 2i capture through `zed-sdk-mlx`:

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/tools/probe_mac_camera.py \
  --capture-device-index 0 \
  --capture-width 2560 \
  --capture-height 720 \
  --capture-backend zed-sdk-mlx-terminal \
  --zed-sdk-mlx-repo /path/to/zed-sdk-mlx \
  --capture-output logs/hardware/zed-live.yuv \
  --json-out logs/hardware/zed-live.json
```

Stereo/depth smoke on a raw side-by-side YUYV dump:

```bash
uv run --python .venv/bin/python scripts/tools/mac_stereo_depth_smoke.py \
  logs/hardware/synthetic_stereo.raw \
  logs/hardware/synthetic_depth \
  --max-disparity 64 \
  --summary-out logs/hardware/synthetic_depth/summary.json
```

The current live-camera validation path is intentionally separated from the main CI ring because camera ownership, TCC attribution, and device ownership depend on the host app. The software path is still fully test-backed through synthetic stereo dumps, and the live ZED path now has a hardware-validated Terminal-hosted workflow.

### 11. Run the focused backend test suite

```bash
PYTHONPATH=.:source/isaaclab:source/isaaclab_rl .venv/bin/pytest \
  scripts/tools/test/test_bootstrap_isaac_sources.py \
  source/isaaclab/test/backends/test_runtime.py \
  source/isaaclab/test/backends/test_task_registry.py \
  source/isaaclab/test/backends/test_kernel_inventory.py \
  source/isaaclab/test/backends/test_kernel_compat.py \
  source/isaaclab/test/backends/test_mac_camera_capture.py \
  source/isaaclab/test/backends/test_mac_cartpole_camera.py \
  source/isaaclab/test/backends/test_mac_hotpath.py \
  source/isaaclab/test/backends/test_mac_sensor_raycast.py \
  source/isaaclab/test/backends/test_mac_stereo_depth.py \
  source/isaaclab/test/backends/test_mlx_task_cli.py \
  source/isaaclab/test/backends/test_planner_compat.py \
  source/isaaclab/test/backends/test_ros2_bridge.py \
  source/isaaclab_rl/test/test_import_safety.py \
  source/isaaclab_rl/test/test_mlx_wrapper.py \
  source/isaaclab/test/backends/test_portability_utils.py \
  source/isaaclab/test/backends/test_mac_benchmark_suite.py \
  source/isaaclab/test/backends/test_mac_semantic_drift.py \
  source/isaaclab/test/backends/test_mac_state_primitives.py \
  source/isaaclab/test/backends/test_mac_phase_b_support.py \
  source/isaaclab/test/backends/test_mac_cartpole.py \
  source/isaaclab/test/backends/test_mac_cartpole_showcase.py \
  source/isaaclab/test/backends/test_mac_cart_double_pendulum.py \
  source/isaaclab/test/backends/test_mac_anymal_c.py \
  source/isaaclab/test/backends/test_mac_anymal_c_rough.py \
  source/isaaclab/test/backends/test_mac_franka_reach.py \
  source/isaaclab/test/backends/test_mac_ur10e_deploy_reach.py \
  source/isaaclab/test/backends/test_mac_ur10e_gear_assembly.py \
  source/isaaclab/test/backends/test_mac_franka_lift.py \
  source/isaaclab_rl/test/test_mlx_cli.py \
  source/isaaclab/test/backends/test_mac_h1.py \
  source/isaaclab/test/backends/test_mac_quadcopter.py -q
```

## Runtime Selection

The backend seam is exposed through the app/runtime layer:

- `--compute-backend torch-cuda|mlx`
- `--kernel-backend warp|metal|cpu`
- `--sim-backend isaacsim|mac-sim`

These flags are published through:

- [`source/isaaclab/isaaclab/app/app_launcher.py`](source/isaaclab/isaaclab/app/app_launcher.py)

Examples:

```bash
# Upstream path
python some_script.py --compute-backend torch-cuda --kernel-backend warp --sim-backend isaacsim

# macOS port path
python some_script.py --compute-backend mlx --kernel-backend metal --sim-backend mac-sim
```

Important constraint: `AppLauncher` now accepts `--compute-backend mlx --sim-backend mac-sim` in bootstrap mode, but it does not launch Isaac Sim/Omniverse there. Use the dedicated MLX scripts for task execution.

## M5 Benchmarking

The fork now includes a dedicated MLX benchmark entrypoint for the current mac-native task set:

- [`scripts/benchmarks/mlx/benchmark_mac_tasks.py`](scripts/benchmarks/mlx/benchmark_mac_tasks.py)

Example:

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/benchmarks/mlx/benchmark_mac_tasks.py \
  --task-group current-mac-native \
  --num-envs 256 \
  --steps 512 \
  --json-out logs/benchmarks/mlx/m5-baseline.json
```

The benchmark emits:

- per-task `env_steps_per_s` for the current MLX/mac-sim env slices
- stable named task groups derived from the typed manifest in [`source/isaaclab/isaaclab/backends/supported_tasks.py`](source/isaaclab/isaaclab/backends/supported_tasks.py)
- a stable public `current-mac-native` task group for cartpole, cart-double-pendulum, quadcopter, ANYmal-C flat, ANYmal-C rough, H1 flat, H1 rough, Franka reach, OpenArm reach, OpenArm bimanual reach, UR10 reach, UR10e deploy-reach, UR10e gear-assembly 2F-140, UR10e gear-assembly 2F-85, UR10 long-suction stack, UR10 short-suction stack, Franka lift, OpenArm lift, Franka teddy-bear lift, Franka stack instance-randomize, Franka stack, Franka stack RGB, Franka bin-stack, Franka cabinet, Franka open-drawer, and OpenArm open-drawer
- a stable `sensor-mac-native` benchmark projection that extends the public camera slices with benchmark-only height-scan variants for `anymal-c-flat` and `h1-flat`
- a stable `full` benchmark projection that combines the public task surface with the benchmark-only sensor/training rows for one normalized dashboard/trend artifact
- runtime metadata including compute, kernel, sensor, and planner backend selection
- runtime metadata that now separates `public_benchmark_groups` from `benchmark_task_groups`, so benchmark-only rows do not masquerade as public tasks
- per-task and suite-level `cpu_fallback` reporting so benchmark JSON shows when the run silently dropped to the CPU kernel backend
- rollout `output_signature` fields for control, locomotion, and sensor slices so semantic drift can be tracked without relying on throughput numbers alone
- companion `*-dashboard.json` and `*-trend.json` artifacts for normalized MLX/mac regression reporting and M-series comparisons

CI now preserves benchmark JSON, dashboard/trend JSON, a dedicated import-safety artifact proving the MLX/mac path can run without `isaacsim`, `omni`, `carb`, or `pxr` installed, and a nightly semantic drift report against the committed MLX/mac baseline in [`scripts/benchmarks/mlx/baselines/semantic-baseline.json`](scripts/benchmarks/mlx/baselines/semantic-baseline.json).

### Benchmark Expectations

- `current-mac-native` is the stable public regression suite for the current MLX/mac task set.
- `sensor-mac-native` is a benchmark projection, not a second public task catalog. It includes the two public cartpole camera slices plus the benchmark-only `anymal-c-flat-height-scan` and `h1-flat-height-scan` variants.
- `full` is the normalized dashboard/trend projection used by CI when one artifact needs to cover rollout plus training health together.
- CI benchmark smokes are regression signals, not public performance claims.
- A benchmark run is only considered healthy when `cpu_fallback.detected == false`.
- Use `logs/benchmarks/mlx/m5-baseline.json` for local M5 comparisons and `logs/benchmarks/mlx/m5-baseline-trend.json` when you want a compact comparison payload across M-series machines. Do not compare against CI smoke numbers.
- CI stores immutable `*-trend.json` artifacts per run; those are the retained history source for M-series regression review, not the raw smoke JSON.
- Nightly semantic drift checks compare the deterministic rollout contracts in the committed baseline against a fresh `full` benchmark run. Throughput is intentionally excluded from that drift gate.
- When a benchmarked task contract changes intentionally, refresh the baseline with:
  `PYTHONPATH=.:source/isaaclab .venv/bin/python scripts/benchmarks/mlx/benchmark_mac_tasks.py --task-group full --num-envs 8 --steps 8 --train-updates 1 --rollout-steps 4 --epochs-per-update 1 --json-out logs/benchmarks/mlx/full-smoke.json`
  then:
  `PYTHONPATH=.:source/isaaclab .venv/bin/python scripts/benchmarks/mlx/check_semantic_drift.py --results logs/benchmarks/mlx/full-smoke.json --baseline scripts/benchmarks/mlx/baselines/semantic-baseline.json --snapshot-out logs/benchmarks/mlx/full-smoke-semantic-snapshot.json --write-baseline`
  and rerun the compare command to confirm the refreshed baseline passes.
- CI now includes a release-surface smoke that installs the MLX runtime without `dev` extras or `PYTHONPATH`, then exercises the installed `isaaclab-mlx-runtime-diagnostics`, `isaaclab-mlx`, `isaaclab-mlx-evaluate`, and `isaaclab-mlx-train` entry points from a clean `uv` environment.
- Backend-local external stereo validation still lives in synthetic stereo smoke tests and optional host-specific hardware probes; the benchmark suite only covers the synthetic cartpole camera task slices and the analytic terrain/raycast slices.
- The runtime diagnostics artifact now proves two distinct things: the public typed task manifest and the concrete articulated `mac-sim` substrate. It does not claim full scene-graph, RTX, or Isaac Sim engine parity.

## Planner And ROS Compatibility

The current mac-native planner bridge is deliberately modest. `mac-planners` now exposes a deterministic joint-space interpolation seam that can accept richer world-state obstacles and produce timed serializable waypoint plans without relying on cuRobo or Isaac Sim:

- [`source/isaaclab/isaaclab/backends/planner_compat.py`](source/isaaclab/isaaclab/backends/planner_compat.py)
- [`scripts/tools/mac_planner_smoke.py`](scripts/tools/mac_planner_smoke.py)

The current ROS 2 bridge is also intentionally plain. It focuses on message/process interoperability first, not NITROS or CUDA transport:

- [`source/isaaclab/isaaclab/backends/ros2_compat.py`](source/isaaclab/isaaclab/backends/ros2_compat.py)
- [`scripts/tools/ros2_bridge_smoke.py`](scripts/tools/ros2_bridge_smoke.py)

Example planner smoke:

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/tools/mac_planner_smoke.py \
  logs/planner/mac-planner-smoke.json
```

Example ROS 2 bridge smoke:

```bash
uv run --python .venv/bin/python scripts/tools/ros2_bridge_smoke.py \
  logs/hardware/ros2-bridge-smoke.jsonl \
  --summary-out logs/hardware/ros2-bridge-smoke-summary.json
```

This is the current compatibility contract:

- planner compatibility on macOS means serializable box/sphere/capsule/mesh world updates, attachment metadata, and deterministic timed joint-space plans
- ROS compatibility on macOS means plain message/process interoperability first, including ROS-friendly world-state and joint-trajectory envelopes plus typed round-trip reconstruction without importing ROS Python bindings
- batched ROS planner envelopes are reconstructed by `batch_index`, not by input order, so JSONL message reordering cannot silently corrupt planner/world batch recovery
- CUDA stream transport, NITROS, and GXF remain future follow-on work

## Kernel Inventory

The next Warp/CUDA kernel families blocking broader parity are tracked in:

- [`source/isaaclab/isaaclab/backends/kernel_inventory.py`](source/isaaclab/isaaclab/backends/kernel_inventory.py)

That inventory is test-backed and currently covers:

- mesh raycast kernels used by ray-caster sensors and terrain sampling
- wrench composer kernels relevant to manipulation, now with helper-level MLX parity coverage
- Fabric transform kernels for future engine-parity work
- tiled-camera reshape kernels for future camera parity, now with helper-level MLX parity coverage

## Implemented MLX Vertical Slice

Current task capability matrix:

| Task | Train | Replay / Eval | Benchmarked | Checkpoint / Resume | Notes |
| --- | --- | --- | --- | --- | --- |
| `Isaac-Cartpole-Direct-v0` | Yes | Yes | `current-mac-native` | Yes | Discrete cartpole PPO slice |
| `Isaac-Cartpole-RGB-Camera-Direct-v0` | No | Yes | `sensor-mac-native` | No | Synthetic analytic RGB camera observations on cartpole control/reward semantics |
| `Isaac-Cartpole-Depth-Camera-Direct-v0` | No | Yes | `sensor-mac-native` | No | Synthetic analytic depth camera observations on cartpole control/reward semantics |
| `Isaac-Cart-Double-Pendulum-Direct-v0` | No | Yes | `current-mac-native` | No | MARL dict observations/actions |
| `Isaac-Quadcopter-Direct-v0` | No | Yes | `current-mac-native` | No | Root-state thrust/moment control |
| `Isaac-Velocity-Flat-Anymal-C-Direct-v0` | Yes | Yes | `current-mac-native` | Yes | Flat-terrain locomotion with an optional benchmark-only height-scan variant |
| `Isaac-Velocity-Rough-Anymal-C-Direct-v0` | Yes | Yes | `current-mac-native` | Yes | Procedural wave terrain plus analytic terrain raycasts with MLX PPO train/replay support |
| `Isaac-Velocity-Flat-H1-v0` | Yes | Yes | `current-mac-native` | Yes | Flat-terrain locomotion with an optional benchmark-only height-scan variant |
| `Isaac-Velocity-Rough-H1-v0` | Yes | Yes | `current-mac-native` | Yes | Procedural wave terrain, analytic terrain raycasts, rough-task checkpoint inference, and MLX PPO train/replay support |
| `Isaac-Reach-Franka-v0` | Yes | Yes | `current-mac-native` | Yes | Analytic joint-space reach slice with MLX PPO train/replay support; compatible upstream aliases include `Isaac-Reach-Franka-IK-Abs-v0`, `Isaac-Reach-Franka-IK-Rel-v0`, `Isaac-Reach-Franka-OSC-v0`, and `Isaac-Reach-Franka-OSC-Play-v0` |
| `Isaac-Reach-OpenArm-v0` | Yes | Yes | `current-mac-native` | Yes | Reduced analytic OpenArm reach slice with explicit `semantic_contract="reduced-openarm-surrogate"` metadata and shared MLX PPO train/replay support; compatible upstream alias also includes `Isaac-Reach-OpenArm-Play-v0` |
| `Isaac-Reach-OpenArm-Bi-v0` | Yes | Yes | `current-mac-native` | Yes | Reduced analytic dual-arm OpenArm reach slice with explicit `semantic_contract="reduced-openarm-bimanual-surrogate"` metadata, dual-arm terminal metrics, and shared MLX PPO train/replay support; compatible upstream alias also includes `Isaac-Reach-OpenArm-Bi-Play-v0` |
| `Isaac-Reach-UR10-v0` | Yes | Yes | `current-mac-native` | Yes | Reduced analytic UR10 reach slice with explicit `semantic_contract="reduced-analytic-pose"` metadata, pose-tracking terminal metrics, and shared MLX PPO train/replay support; compatible upstream alias also includes `Isaac-Reach-UR10-Play-v0` |
| `Isaac-Deploy-Reach-UR10e-v0` | Yes | Yes | `current-mac-native` | Yes | Reduced analytic UR10e deploy-reach slice with explicit `semantic_contract="reduced-analytic-pose"` metadata, pose-command terminal metrics, and shared MLX PPO train/replay support; compatible upstream aliases also include `Isaac-Deploy-Reach-UR10e-Play-v0` and the reduced-contract `Isaac-Deploy-Reach-UR10e-ROS-Inference-v0` |
| `Isaac-Deploy-GearAssembly-UR10e-2F140-v0` | Yes | Yes | `current-mac-native` | Yes | Reduced analytic UR10e gear-assembly slice for the Robotiq 2F-140 gripper with explicit `semantic_contract="reduced-analytic-assembly"` metadata, insertion-aware policy observations, and shared MLX PPO train/replay support; compatible upstream alias also includes `Isaac-Deploy-GearAssembly-UR10e-2F140-Play-v0` |
| `Isaac-Deploy-GearAssembly-UR10e-2F85-v0` | Yes | Yes | `current-mac-native` | Yes | Reduced analytic UR10e gear-assembly slice for the Robotiq 2F-85 gripper with explicit `semantic_contract="reduced-analytic-assembly"` metadata, insertion-aware policy observations, and shared MLX PPO train/replay support; compatible upstream alias also includes `Isaac-Deploy-GearAssembly-UR10e-2F85-Play-v0` |
| `Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0` | Yes | Yes | `current-mac-native` | Yes | Reduced analytic UR10 long-suction three-cube stack slice with explicit `semantic_contract="reduced-analytic-suction-stack"` metadata, suction-state surrogate logic, and shared MLX PPO train/replay support; the canonical mac-native task is `ur10-long-suction-stack` |
| `Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0` | Yes | Yes | `current-mac-native` | Yes | Reduced analytic UR10 short-suction three-cube stack slice with explicit `semantic_contract="reduced-analytic-suction-stack"` metadata, suction-state surrogate logic, and shared MLX PPO train/replay support; the canonical mac-native task is `ur10-short-suction-stack` |
| `Isaac-Lift-Cube-Franka-v0` | Yes | Yes | `current-mac-native` | Yes | Analytic lift slice with lightweight grasp logic and MLX PPO train/replay support; compatible upstream aliases include `Isaac-Lift-Cube-Franka-IK-Abs-v0` and `Isaac-Lift-Cube-Franka-IK-Rel-v0` |
| `Isaac-Lift-Cube-OpenArm-v0` | Yes | Yes | `current-mac-native` | Yes | Reduced analytic OpenArm lift slice with explicit `semantic_contract="reduced-openarm-surrogate"` metadata, lightweight grasp logic, and shared MLX PPO train/replay support; compatible upstream alias also includes `Isaac-Lift-Cube-OpenArm-Play-v0` |
| `Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0` | Yes | Yes | `current-mac-native` | Yes | Reduced plush-object lift slice mapped onto the analytic lift substrate with shared MLX PPO train/replay support and benchmark diagnostics keyed to the teddy-bear object contract |
| `Isaac-Stack-Cube-Instance-Randomize-Franka-v0` | Yes | Yes | `current-mac-native` | Yes | Reduced instance-randomized two-cube stack slice with explicit variant-id observations, deterministic distinct support/movable object sampling, and shared MLX PPO train/replay support; compatible upstream alias also includes `Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0` |
| `Isaac-Stack-Cube-Franka-v0` | Yes | Yes | `current-mac-native` | Yes | Analytic two-cube stack slice with lightweight grasp/release logic and MLX PPO train/replay support; compatible upstream aliases include `Isaac-Stack-Cube-Franka-IK-Abs-v0`, `Isaac-Stack-Cube-Franka-IK-Rel-v0`, `Isaac-Stack-Cube-RedGreen-Franka-IK-Rel-v0`, `Isaac-Stack-Cube-BlueGreen-Franka-IK-Rel-v0`, and the reduced-contract `Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0` / `Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0` aliases |
| `Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0` | Yes | Yes | `current-mac-native` | Yes | Analytic three-cube sequential stack slice with staged terminal metrics and MLX PPO train/replay support; compatible upstream aliases also include `Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0` and the reduced-contract `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0` / `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0` aliases |
| `Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0` | Yes | Yes | `current-mac-native` | Yes | Reduced bin-anchored three-cube stack slice with explicit `semantic_contract="reduced-no-mimic"` metadata, benchmark anchor metrics, and shared MLX PPO train/replay support |
| `Isaac-Franka-Cabinet-Direct-v0` | Yes | Yes | `current-mac-native` | Yes | Reduced drawer-handle workflow with deterministic open-distance semantics and MLX PPO train/replay support |
| `Isaac-Open-Drawer-Franka-v0` | Yes | Yes | `current-mac-native` | Yes | Manager-style open-drawer slice mapped onto the reduced analytic drawer substrate with MLX PPO train/replay support; compatible upstream aliases include `Isaac-Open-Drawer-Franka-IK-Abs-v0` and `Isaac-Open-Drawer-Franka-IK-Rel-v0` |
| `Isaac-Open-Drawer-OpenArm-v0` | Yes | Yes | `current-mac-native` | Yes | Reduced analytic OpenArm drawer slice with explicit `semantic_contract="reduced-openarm-surrogate"` metadata, deterministic handle/open-distance terminal metrics, and shared MLX PPO train/replay support; compatible upstream alias also includes `Isaac-Open-Drawer-OpenArm-Play-v0` |

Implementation entrypoints:

- cartpole environment and trainer in [`source/isaaclab/isaaclab/backends/mac_sim/cartpole.py`](source/isaaclab/isaaclab/backends/mac_sim/cartpole.py)
- synthetic cartpole RGB/depth camera environment and analytic camera helper in [`source/isaaclab/isaaclab/backends/mac_sim/cartpole_camera.py`](source/isaaclab/isaaclab/backends/mac_sim/cartpole_camera.py)
- cartpole showcase space variants in [`source/isaaclab/isaaclab/backends/mac_sim/showcase.py`](source/isaaclab/isaaclab/backends/mac_sim/showcase.py)
- cart-double-pendulum MARL environment in [`source/isaaclab/isaaclab/backends/mac_sim/cart_double_pendulum.py`](source/isaaclab/isaaclab/backends/mac_sim/cart_double_pendulum.py)
- quadcopter environment in [`source/isaaclab/isaaclab/backends/mac_sim/quadcopter.py`](source/isaaclab/isaaclab/backends/mac_sim/quadcopter.py)
- ANYmal-C flat and rough locomotion environments in [`source/isaaclab/isaaclab/backends/mac_sim/anymal_c.py`](source/isaaclab/isaaclab/backends/mac_sim/anymal_c.py)
- H1 flat locomotion environment and trainer in [`source/isaaclab/isaaclab/backends/mac_sim/h1.py`](source/isaaclab/isaaclab/backends/mac_sim/h1.py)
- Franka reach/OpenArm reach/OpenArm bimanual reach/UR10 reach/UR10e deploy-reach/UR10e gear-assembly/lift/OpenArm lift/teddy-bear-lift/stack-instance-randomize/stack/stack-RGB/bin-stack/cabinet/open-drawer/OpenArm open-drawer manipulation environments in [`source/isaaclab/isaaclab/backends/mac_sim/manipulation.py`](source/isaaclab/isaaclab/backends/mac_sim/manipulation.py)
- analytic terrain raycast / height-scan substrate in [`source/isaaclab/isaaclab/backends/mac_sim/sensors.py`](source/isaaclab/isaaclab/backends/mac_sim/sensors.py)
- backend-local macOS external camera discovery/capture helpers in [`source/isaaclab/isaaclab/backends/mac_sim/cameras.py`](source/isaaclab/isaaclab/backends/mac_sim/cameras.py)
- backend-local MLX stereo/depth helpers in [`source/isaaclab/isaaclab/backends/mac_sim/stereo_depth.py`](source/isaaclab/isaaclab/backends/mac_sim/stereo_depth.py)
- public MLX wrapper surface in [`source/isaaclab_rl/isaaclab_rl/mlx.py`](source/isaaclab_rl/isaaclab_rl/mlx.py)
- installed MLX train/eval CLI surface in [`source/isaaclab_rl/isaaclab_rl/mlx_cli.py`](source/isaaclab_rl/isaaclab_rl/mlx_cli.py)
- installed runtime diagnostics CLI surface in [`source/isaaclab/isaaclab/backends/runtime_cli.py`](source/isaaclab/isaaclab/backends/runtime_cli.py)
- shared trainer entrypoint in [`scripts/reinforcement_learning/mlx/train_task.py`](scripts/reinforcement_learning/mlx/train_task.py)
- shared replay/eval entrypoint in [`scripts/reinforcement_learning/mlx/evaluate_task.py`](scripts/reinforcement_learning/mlx/evaluate_task.py)
- backend-local camera probe in [`scripts/tools/probe_mac_camera.py`](scripts/tools/probe_mac_camera.py)
- backend-local stereo/depth smoke in [`scripts/tools/mac_stereo_depth_smoke.py`](scripts/tools/mac_stereo_depth_smoke.py)
- backend/runtime diagnostics CLI in [`scripts/tools/mac_runtime_diagnostics.py`](scripts/tools/mac_runtime_diagnostics.py)

The cartpole path preserves the important upstream task semantics:

- observation order remains `pole_pos, pole_vel, cart_pos, cart_vel`
- reward structure matches upstream Isaac Lab cartpole
- termination conditions remain cart out-of-bounds or pole angle exceeding `pi/2`
- reset sampling keeps the pole-angle randomization behavior
- observations returned after done/reset are post-reset observations, matching the upstream direct RL flow
- synthetic cartpole camera tasks preserve the cartpole control/reward/reset semantics while swapping vector observations for deterministic image tensors with upstream-aligned `100x100` shapes
- cart-double-pendulum preserves per-agent dict observations/rewards/dones for `cart` and `pendulum`
- quadcopter preserves a root-state-centric policy observation layout with vectorized thrust/moment control
- ANYmal-C preserves a command-tracking locomotion observation layout with flat-terrain contacts, base-contact termination, and MLX PPO smoke coverage
- ANYmal-C rough preserves the same command-tracking layout while swapping in procedural wave terrain and analytic terrain raycasts for deterministic rough-terrain evaluation
- H1 flat preserves a 19-DOF command-driven locomotion observation layout with contact-aware reward terms, base-contact termination, and MLX PPO smoke coverage
- H1 rough preserves the same H1 policy layout while fixing the configured observation width to include the nine default height-scan channels used by the rough task
- Franka reach preserves a deterministic joint-space control/reward loop with explicit target-distance semantics and now exposes checkpointed MLX PPO training/replay
- UR10e deploy-reach preserves a deterministic pose-command reach workflow with analytic end-effector tracking, explicit target-distance and orientation-error terminal metrics, and reduced-contract metadata so the mac-native slice stays honest about the upstream deployment-oriented semantics it does not fully preserve
- UR10e gear assembly preserves the upstream-sized 19D policy observation contract by encoding shaft insertion progress back into the observed shaft pose, then exposes explicit reduced-contract metadata so the 2F-140 and 2F-85 slices stay honest about the factory-contact and ROS deployment semantics they do not fully preserve
- OpenArm reach preserves the single-arm reach workflow while exposing explicit reduced-contract metadata and a 7-DoF analytic surrogate instead of pretending exact OpenArm morphology parity
- OpenArm bimanual reach preserves a dual-arm reach workflow with explicit left/right target-distance terminal metrics and reduced-contract metadata instead of pretending exact OpenArm body-stack parity
- UR10 reach preserves a deterministic pose-tracking reach workflow with analytic end-effector tracking, explicit target-distance and orientation-error terminal metrics, and reduced-contract metadata rather than implying broader UR controller-stack parity
- Franka lift preserves deterministic cube placement, lightweight grasp state, explicit lift-success semantics, and multi-step replay coverage
- OpenArm lift preserves the shared lift PPO/checkpoint/reset contract while exposing reduced-contract metadata for the OpenArm surrogate geometry instead of forking a new lift simulator family
- Franka teddy-bear lift preserves the shared lift PPO/checkpoint/reset contract while swapping in a reduced plush-object placement and success-height envelope, so the mac-native slice stays honest to the upstream task ID without pretending full soft-body or visual parity
- Franka stack instance-randomize preserves the shared stack PPO/checkpoint/reset contract while appending explicit variant-id observations and deterministic distinct support/movable object sampling, so the upstream randomized task ID stays honest without silently collapsing into the plain stack slice
- Franka stack preserves deterministic support-cube placement, lightweight release-driven stack success semantics, and train/replay parity with the other manipulation slices
- Franka stack RGB preserves a staged three-cube manipulation contract with transition-only middle-stage bonuses, top-stage terminal metrics derived from pre-reset observations, and train/replay parity with the other manipulation slices
- Franka cabinet preserves a reduced drawer-handle workflow with deterministic open-distance semantics, lightweight handle-grasp logic, and train/replay parity with the other manipulation slices
- Franka open-drawer preserves the reduced analytic drawer substrate while exposing the manager-style public task ID, deterministic handle-distance/open-distance terminal metrics, and the same train/replay/checkpoint contract as the other Franka slices
- OpenArm open-drawer preserves the reduced analytic drawer workflow with OpenArm-surrogate reduced-contract metadata, deterministic handle-distance/open-distance terminal metrics, and the same train/replay/checkpoint contract as the other manipulation slices
- upstream-compatible reach/lift/stack/open-drawer controller variants plus the `Isaac-Deploy-Reach-UR10e-v0` / `Isaac-Deploy-Reach-UR10e-Play-v0` / `Isaac-Deploy-Reach-UR10e-ROS-Inference-v0` deploy aliases now normalize to canonical MLX task keys in the public wrapper and installed CLI, so the mac-native surface stays recognizable without pretending separate controller-physics or deployment-runtime parity
- upstream task IDs that name heavier policy, model, or process layers now either normalize to explicit reduced-contract aliases or stay isaacsim-only. The supported reduced aliases here are `Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0`, `Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0`, `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0`, `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0`, `Isaac-Deploy-Reach-UR10e-ROS-Inference-v0`, `Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0`, `Isaac-PickPlace-GR1T2-Abs-v0`, `Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0`, `Isaac-PickPlace-G1-InspireFTP-Abs-v0`, `Isaac-NutPour-GR1T2-Pink-IK-Abs-v0`, and `Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0`; all keep explicit reduced-contract metadata instead of pretending blueprint / skillgen / visuomotor / Cosmos / ROS-inference / pick-place scene parity
- the first mac-native sensor slice is an analytic terrain raycast / height-scan path for locomotion benchmarks
- the `sensor-mac-native` benchmark rows now cover the synthetic cartpole RGB/depth camera tasks plus `height_scan_enabled=True` variants of the ANYmal-C and H1 flat locomotion tasks
- benchmark and semantic drift reports now surface `hotpath: "mlx-metal-root-step"` for the locomotion root-step seam when active, `hotpath: "mlx-metal-ee"` for the Franka reach/lift/cabinet/open-drawer seam, `hotpath: "mlx-metal-franka-stack"` for the two-cube stack seam, and `hotpath: "mlx-metal-franka-stack-rgb"` for the three-cube/bin-stack seam when those dedicated helpers are active, while keeping the remaining helpers explicit when they stay on compiled MLX
- the first backend-local mac camera slice is a UVC/AVFoundation discovery path plus raw stereo YUYV depth smoke, with live hardware validation kept host-specific

## Bootstrapping Upstream Sources

This fork also includes a repository bootstrap script so the Isaac Lab / Isaac Sim / Isaac ROS sources can be cloned reproducibly into a workspace.

Script:

- [`scripts/bootstrap_isaac_sources.py`](scripts/bootstrap_isaac_sources.py)

Example:

```bash
uv run scripts/bootstrap_isaac_sources.py \
  --dest ../isaac-sources \
  --with-isaacsim \
  --with-isaac-ros
```

The script:

- clones `IsaacLab` by default
- optionally clones `IsaacSim`
- optionally clones `isaac_ros_common`
- runs `git lfs install` / `git lfs pull` only for `IsaacSim`
- writes a manifest with remotes, requested refs, and resolved SHAs

## What Has Changed Relative To Upstream

The most important fork changes are:

- backend/runtime abstractions in [`source/isaaclab/isaaclab/backends`](source/isaaclab/isaaclab/backends)
- MLX cartpole reference implementation in [`source/isaaclab/isaaclab/backends/mac_sim`](source/isaaclab/isaaclab/backends/mac_sim)
- lazy import handling in `envs`, `sim`, and `utils` so Mac users get explicit unsupported-backend errors
- CUDA device binding moved behind an adapter instead of direct unconditional `torch.cuda.set_device(...)`
- public README and bootstrap path oriented around a publishable MLX fork

## Roadmap

Near-term priorities:

1. Expand the new generic `mac-sim` articulation/scene substrate to cover more shared contacts, sensors, and asset patterns.
2. Add additional locomotion/manipulation tasks with compatible observation/action structure.
3. Push more task code through backend capability checks instead of import-time backend assumptions.
4. Define a stable checkpoint/config story across multiple MLX tasks.
5. Expand the public support matrix and benchmark expectations as the mac-native slice grows.

Deferred work:

- Omniverse UI and Kit extension parity
- RTX camera and sensor parity
- Warp and CUDA custom kernel replacements beyond the first task set
- cuRobo and Isaac ROS CUDA transport features

## Relationship To Upstream Isaac Lab

This repository is a fork of the upstream Isaac Lab project and still depends heavily on its architecture, APIs, tasks, and research workflow design.

Upstream remains the reference source for:

- task semantics
- asset conventions
- environment configuration patterns
- Isaac Sim integrations

Original upstream repository:

- [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)

Related reference repositories:

- [isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim)
- [NVIDIA-ISAAC-ROS/isaac_ros_common](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common)
- [ml-explore/mlx](https://github.com/ml-explore/mlx)

## Documentation

The upstream Isaac Lab documentation remains the best reference for the broader project model:

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Isaac Sim Version Dependency

Upstream Isaac Lab is built on top of Isaac Sim and requires compatible Isaac Sim versions. This fork still treats Isaac Sim as the reference runtime for the upstream path.

| Isaac Lab Version             | Isaac Sim Version         |
| ----------------------------- | ------------------------- |
| `main` branch                 | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.3.X`                      | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.2.X`                      | Isaac Sim 4.5 / 5.0       |
| `v2.1.X`                      | Isaac Sim 4.5             |
| `v2.0.X`                      | Isaac Sim 4.5             |

## Contributing

Contributions are welcome, especially in these areas:

- additional MLX-backed task ports
- generalized `mac-sim` contacts, sensors, assets, and scene services on top of the shared batched articulation layer
- capability gating around CUDA-only integrations
- Apple Silicon verification and CI
- documentation for MLX and macOS users

For general upstream contribution guidance, see:

- [Upstream contributing guide](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html)

## Troubleshooting

Common current pitfalls:

- If `mlx` is missing, install it into the active `uv` environment.
- If imports fail on `omni.*` or `isaacsim.*`, you are probably invoking an upstream Isaac Sim path rather than the `mac-sim` path.
- If you import `isaaclab.sensors.camera` or `isaaclab.sensors.ray_caster` on `mac-sim`, config helpers are available but runtime exports like `Camera`, `RayCaster`, and `spawn_camera` should fail explicitly with `sim-backend=isaacsim`.
- If you use `AppLauncher` with `--sim-backend mac-sim`, it runs in bootstrap mode and returns a placeholder app object. Use the dedicated MLX scripts for environment stepping/training.
- If a task depends on cameras, Warp, cuRobo, or Omniverse-only features, it should currently be treated as unsupported on the macOS path.

## Support

- Use GitHub Discussions for design discussions, roadmap ideas, and supported-task requests.
- Use GitHub Issues for concrete bugs, missing capability checks, broken MLX scripts, or backend portability problems.

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

Isaac Lab still requires Isaac Sim for the upstream runtime path, and Isaac Sim includes components under proprietary licensing terms. Please see the [Isaac Sim license](docs/licenses/dependencies/isaacsim-license.txt) for details.

The `isaaclab_mimic` extension requires cuRobo, which has proprietary licensing terms listed in [`docs/licenses/dependencies/cuRobo-license.txt`](docs/licenses/dependencies/cuRobo-license.txt).

## Citation

If you use Isaac Lab or this MLX fork in your research, please cite the technical report:

```bibtex
@article{mittal2025isaaclab,
  title={Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning},
  author={Mayank Mittal and Pascal Roth and James Tigue and Antoine Richard and Octi Zhang and Peter Du and Antonio Serrano-Muñoz and Xinjie Yao and René Zurbrügg and Nikita Rudin and Lukasz Wawrzyniak and Milad Rakhsha and Alain Denzler and Eric Heiden and Ales Borovicka and Ossama Ahmed and Iretiayo Akinola and Abrar Anwar and Mark T. Carlson and Ji Yuan Feng and Animesh Garg and Renato Gasoto and Lionel Gulich and Yijie Guo and M. Gussert and Alex Hansen and Mihir Kulkarni and Chenran Li and Wei Liu and Viktor Makoviychuk and Grzegorz Malczyk and Hammad Mazhar and Masoud Moghani and Adithyavairavan Murali and Michael Noseworthy and Alexander Poddubny and Nathan Ratliff and Welf Rehberg and Clemens Schwarke and Ritvik Singh and James Latham Smith and Bingjie Tang and Ruchik Thaker and Matthew Trepte and Karl Van Wyk and Fangzhou Yu and Alex Millane and Vikram Ramasamy and Remo Steiner and Sangeeta Subramanian and Clemens Volk and CY Chen and Neel Jawale and Ashwin Varghese Kuruttukulam and Michael A. Lin and Ajay Mandlekar and Karsten Patzwaldt and John Welsh and Huihua Zhao and Fatima Anes and Jean-Francois Lafleche and Nicolas Moënne-Loccoz and Soowan Park and Rob Stepinski and Dirk Van Gelder and Chris Amevor and Jan Carius and Jumyung Chang and Anka He Chen and Pablo de Heras Ciechomski and Gilles Daviet and Mohammad Mohajerani and Julia von Muralt and Viktor Reutskyy and Michael Sauter and Simon Schirm and Eric L. Shi and Pierre Terdiman and Kenny Vilella and Tobias Widmer and Gordon Yeoman and Tiffany Chen and Sergey Grizan and Cathy Li and Lotus Li and Connor Smith and Rafael Wiltz and Kostas Alexis and Yan Chang and David Chu and Linxi "Jim" Fan and Farbod Farshidian and Ankur Handa and Spencer Huang and Marco Hutter and Yashraj Narang and Soha Pouya and Shiwei Sheng and Yuke Zhu and Miles Macklin and Adam Moravanszky and Philipp Reist and Yunrong Guo and David Hoeller and Gavriel State},
  journal={arXiv preprint arXiv:2511.04831},
  year={2025},
  url={https://arxiv.org/abs/2511.04831}
}
```

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. This fork builds on that work and on the upstream Isaac Lab architecture.
