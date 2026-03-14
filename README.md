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
- trainable `MLX + mac-sim` Franka reach, cube-lift, cube-stack, and cabinet-drawer slices with deterministic analytic kinematics, lightweight grasp/open/stack logic, and benchmark diagnostics that report `hotpath: "mlx-compiled"`
- a first mac-native analytic terrain raycast / height-scan sensor substrate for locomotion tasks
- eval-only synthetic cartpole RGB/depth camera slices with deterministic analytic `100x100` observations, upstream-aligned reset ranges, and sensor benchmark coverage
- a backend-local macOS external stereo camera discovery/capture path for UVC devices such as ZED 2i, including a live Terminal-hosted `zed-sdk-mlx` validation path for macOS TCC-safe raw capture
- a basic MLX stereo/depth smoke path on raw side-by-side YUYV dumps
- MLX training, checkpoint save/load, and replay scripts
- a public `isaaclab_rl.mlx` wrapper surface for the current MLX/mac task set
- a shared MLX PPO helper substrate for checkpoint metadata, GAE, advantage normalization, and resume-hidden-dim recovery
- a shared checkpoint sidecar schema with metadata versioning, task IDs, policy distribution tags, and explicit env-vs-policy action-space fields
- a planner compatibility seam for `mac-planners` with deterministic joint-space interpolation and world-state updates
- a plain ROS 2 compatibility bridge for JSONL message transport and optional `ros2 topic pub/echo` command construction without CUDA assumptions
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

The current `mac-sim` implementation is intentionally narrow. It currently covers 11 current mac-native rollout tasks: cartpole, cart-double-pendulum, quadcopter, ANYmal-C flat, ANYmal-C rough, H1 flat, H1 rough, Franka reach, Franka lift, Franka stack, Franka cabinet, plus synthetic cartpole RGB/depth camera variants. The task capability matrix below is the authoritative public support surface.

### Kernel backend

The kernel backend isolates Warp and future Metal custom-kernel paths:

- `warp`: upstream Isaac Sim + Warp path
- `metal`: Apple Silicon path for MLX + Metal-backed kernel replacements
- `cpu`: correctness fallback for bring-up and unsupported kernels

Today the public MLX tasks mostly use pure MLX ops. The explicit kernel selection seam exists now so future Metal kernels can land without rewriting task-facing APIs again.

For the current locomotion and Franka manipulation slices, the hottest shared batched paths now run through compiled MLX helpers in `isaaclab.backends.mac_sim.hotpath`, and benchmark diagnostics surface that as `hotpath: "mlx-compiled"`.

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

### Public Support Matrix

This is the current public support contract for runtime combinations, not just what can be installed.

| Platform / Runtime | Status | Notes |
| --- | --- | --- |
| Apple Silicon + `mlx` + `metal` + `mac-sim` | Supported | Current mac-native slice: cartpole, cart-double-pendulum, quadcopter, ANYmal-C flat, ANYmal-C rough, H1 flat, H1 rough, Franka reach, Franka lift, Franka stack, Franka cabinet, and synthetic cartpole RGB/depth camera tasks |
| Apple Silicon + `mlx` + `cpu` + `mac-sim` | Supported for correctness/debug | Useful for bring-up only, not benchmark claims |
| Linux/NVIDIA + `torch-cuda` + `warp` + `isaacsim` | Supported reference path | Upstream-compatible CUDA / Isaac Sim runtime |
| Apple Silicon + `isaacsim` runtime | Unsupported | This fork does not ship Isaac Sim / Omniverse parity on macOS |
| Apple Silicon + backend-local external stereo capture | Supported prototype | AVFoundation discovery plus raw stereo dump and MLX depth smoke; live ZED/ZED 2i capture is supported through a camera-authorized Terminal host with `zed-sdk-mlx`, not as Isaac Sim camera parity |
| Apple Silicon + `mac-planners` planner seam | Supported prototype | Deterministic joint-space interpolation compatibility layer; not cuRobo parity |
| Apple Silicon + plain ROS 2 message/process bridge | Supported prototype | JSONL bridge and optional `ros2` CLI command builder without CUDA/NITROS |
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

### 1. Create the environment with `uv`

```bash
cd IsaacLab
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e source/isaaclab[macos-mlx,dev]
```

If you want the upstream Isaac Sim runtime on Linux/NVIDIA instead, install the CUDA-side extra:

```bash
uv pip install --python .venv/bin/python -e source/isaaclab[cuda-isaacsim,dev]
```

### 2. Train the MLX cartpole baseline

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/train_cartpole.py \
  --num-envs 256 \
  --updates 200 \
  --rollout-steps 64 \
  --epochs-per-update 4 \
  --checkpoint logs/mlx/cartpole_policy.npz
```

### 3. Replay a trained checkpoint

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/reinforcement_learning/mlx/play_cartpole.py \
  --checkpoint logs/mlx/cartpole_policy.npz \
  --episodes 3
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
lift_train_payload = train_mlx_task("franka-lift", num_envs=64, updates=5)
lift_payload = evaluate_mlx_task("franka-lift", checkpoint=lift_train_payload["checkpoint_path"], episodes=2)
stack_train_payload = train_mlx_task("franka-stack", num_envs=64, updates=5)
stack_payload = evaluate_mlx_task("franka-stack", checkpoint=stack_train_payload["checkpoint_path"], episodes=2)
cabinet_train_payload = train_mlx_task("franka-cabinet", num_envs=64, updates=5)
cabinet_payload = evaluate_mlx_task("franka-cabinet", checkpoint=cabinet_train_payload["checkpoint_path"], episodes=2)
```

### 9. Train and replay the Franka reach, lift, stack, and cabinet slices

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
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/tools/mac_stereo_depth_smoke.py \
  logs/hardware/synthetic_stereo.raw \
  logs/hardware/synthetic_depth \
  --max-disparity 64
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
  source/isaaclab/test/backends/test_mac_franka_lift.py \
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
- a stable `current-mac-native` task group for cartpole, cart-double-pendulum, quadcopter, ANYmal-C flat, ANYmal-C rough, H1 flat, H1 rough, Franka reach, Franka lift, Franka stack, and Franka cabinet
- a stable `sensor-mac-native` task group for `cartpole-rgb-camera`, `cartpole-depth-camera`, `anymal-c-flat-height-scan`, and `h1-flat-height-scan`
- a stable `full` task group that adds the current sensor slices plus a lightweight cartpole training benchmark for shared dashboard coverage
- runtime metadata including compute, kernel, sensor, and planner backend selection
- per-task and suite-level `cpu_fallback` reporting so benchmark JSON shows when the run silently dropped to the CPU kernel backend
- rollout `output_signature` fields for control, locomotion, and sensor slices so semantic drift can be tracked without relying on throughput numbers alone
- companion `*-dashboard.json` and `*-trend.json` artifacts for normalized MLX/mac regression reporting and M-series comparisons

CI now preserves benchmark JSON, dashboard/trend JSON, a dedicated import-safety artifact proving the MLX/mac path can run without `isaacsim`, `omni`, `carb`, or `pxr` installed, and a nightly semantic drift report against the committed MLX/mac baseline in [`scripts/benchmarks/mlx/baselines/semantic-baseline.json`](scripts/benchmarks/mlx/baselines/semantic-baseline.json).

### Benchmark Expectations

- `current-mac-native` is the stable public regression suite for the current MLX/mac task set.
- `sensor-mac-native` is the stable regression suite for the first analytic sensor slices: `cartpole-rgb-camera`, `cartpole-depth-camera`, `anymal-c-flat-height-scan`, and `h1-flat-height-scan`.
- `full` is the normalized dashboard/trend suite used by CI when one artifact needs to cover rollout plus training health together.
- CI benchmark smokes are regression signals, not public performance claims.
- A benchmark run is only considered healthy when `cpu_fallback.detected == false`.
- Use `logs/benchmarks/mlx/m5-baseline.json` for local M5 comparisons and `logs/benchmarks/mlx/m5-baseline-trend.json` when you want a compact comparison payload across M-series machines. Do not compare against CI smoke numbers.
- CI stores immutable `*-trend.json` artifacts per run; those are the retained history source for M-series regression review, not the raw smoke JSON.
- Nightly semantic drift checks compare the deterministic rollout contracts in the committed baseline against a fresh `full` benchmark run. Throughput is intentionally excluded from that drift gate.
- Backend-local external stereo validation still lives in synthetic stereo smoke tests and optional host-specific hardware probes; the benchmark suite only covers the synthetic cartpole camera task slices and the analytic terrain/raycast slices.

## Planner And ROS Compatibility

The current mac-native planner bridge is deliberately modest. `mac-planners` now exposes a deterministic joint-space interpolation seam that can accept world-state obstacles and produce serializable waypoint plans without relying on cuRobo or Isaac Sim:

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
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/tools/ros2_bridge_smoke.py \
  logs/hardware/ros2-bridge-smoke.jsonl \
  --summary-out logs/hardware/ros2-bridge-smoke-summary.json
```

This is the current compatibility contract:

- planner compatibility on macOS means serializable world updates plus deterministic joint-space plans
- ROS compatibility on macOS means plain message/process interoperability first
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
| `Isaac-Velocity-Flat-Anymal-C-Direct-v0` | Yes | Yes | `current-mac-native`, `sensor-mac-native` | Yes | Flat-terrain locomotion, optional height scan |
| `Isaac-Velocity-Rough-Anymal-C-Direct-v0` | No | Yes | `current-mac-native` | No | Procedural wave terrain plus analytic terrain raycasts |
| `Isaac-Velocity-Flat-H1-v0` | Yes | Yes | `current-mac-native`, `sensor-mac-native` | Yes | Flat-terrain locomotion, optional height scan |
| `Isaac-Velocity-Rough-H1-v0` | No | Yes | `current-mac-native` | No | Procedural wave terrain, analytic terrain raycasts, and H1 rough-terrain semantics |
| `Isaac-Reach-Franka-v0` | Yes | Yes | `current-mac-native` | Yes | Analytic joint-space reach slice with MLX PPO train/replay support |
| `Isaac-Lift-Cube-Franka-v0` | Yes | Yes | `current-mac-native` | Yes | Analytic lift slice with lightweight grasp logic and MLX PPO train/replay support |
| `Isaac-Stack-Cube-Franka-v0` | Yes | Yes | `current-mac-native` | Yes | Analytic two-cube stack slice with lightweight grasp/release logic and MLX PPO train/replay support |

Implementation entrypoints:

- cartpole environment and trainer in [`source/isaaclab/isaaclab/backends/mac_sim/cartpole.py`](source/isaaclab/isaaclab/backends/mac_sim/cartpole.py)
- synthetic cartpole RGB/depth camera environment and analytic camera helper in [`source/isaaclab/isaaclab/backends/mac_sim/cartpole_camera.py`](source/isaaclab/isaaclab/backends/mac_sim/cartpole_camera.py)
- cartpole showcase space variants in [`source/isaaclab/isaaclab/backends/mac_sim/showcase.py`](source/isaaclab/isaaclab/backends/mac_sim/showcase.py)
- cart-double-pendulum MARL environment in [`source/isaaclab/isaaclab/backends/mac_sim/cart_double_pendulum.py`](source/isaaclab/isaaclab/backends/mac_sim/cart_double_pendulum.py)
- quadcopter environment in [`source/isaaclab/isaaclab/backends/mac_sim/quadcopter.py`](source/isaaclab/isaaclab/backends/mac_sim/quadcopter.py)
- ANYmal-C flat and rough locomotion environments in [`source/isaaclab/isaaclab/backends/mac_sim/anymal_c.py`](source/isaaclab/isaaclab/backends/mac_sim/anymal_c.py)
- H1 flat locomotion environment and trainer in [`source/isaaclab/isaaclab/backends/mac_sim/h1.py`](source/isaaclab/isaaclab/backends/mac_sim/h1.py)
- Franka reach/lift/stack/cabinet manipulation environments in [`source/isaaclab/isaaclab/backends/mac_sim/manipulation.py`](source/isaaclab/isaaclab/backends/mac_sim/manipulation.py)
- analytic terrain raycast / height-scan substrate in [`source/isaaclab/isaaclab/backends/mac_sim/sensors.py`](source/isaaclab/isaaclab/backends/mac_sim/sensors.py)
- backend-local macOS external camera discovery/capture helpers in [`source/isaaclab/isaaclab/backends/mac_sim/cameras.py`](source/isaaclab/isaaclab/backends/mac_sim/cameras.py)
- backend-local MLX stereo/depth helpers in [`source/isaaclab/isaaclab/backends/mac_sim/stereo_depth.py`](source/isaaclab/isaaclab/backends/mac_sim/stereo_depth.py)
- public MLX wrapper surface in [`source/isaaclab_rl/isaaclab_rl/mlx.py`](source/isaaclab_rl/isaaclab_rl/mlx.py)
- shared trainer entrypoint in [`scripts/reinforcement_learning/mlx/train_task.py`](scripts/reinforcement_learning/mlx/train_task.py)
- shared replay/eval entrypoint in [`scripts/reinforcement_learning/mlx/evaluate_task.py`](scripts/reinforcement_learning/mlx/evaluate_task.py)
- backend-local camera probe in [`scripts/tools/probe_mac_camera.py`](scripts/tools/probe_mac_camera.py)
- backend-local stereo/depth smoke in [`scripts/tools/mac_stereo_depth_smoke.py`](scripts/tools/mac_stereo_depth_smoke.py)

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
- Franka lift preserves deterministic cube placement, lightweight grasp state, explicit lift-success semantics, and multi-step replay coverage
- Franka stack preserves deterministic support-cube placement, lightweight release-driven stack success semantics, and train/replay parity with the other manipulation slices
- Franka cabinet preserves a reduced drawer-handle workflow with deterministic open-distance semantics, lightweight handle-grasp logic, and train/replay parity with the other manipulation slices
- the first mac-native sensor slice is an analytic terrain raycast / height-scan path for locomotion benchmarks
- the `sensor-mac-native` benchmark rows now cover the synthetic cartpole RGB/depth camera tasks plus `height_scan_enabled=True` variants of the ANYmal-C and H1 flat locomotion tasks
- benchmark and semantic drift reports now surface `hotpath: "mlx-compiled"` for the Franka reach/lift/stack/cabinet slices alongside the locomotion tasks
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

1. Expand `mac-sim` from the current task slices into a more general articulation/scene layer.
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
- generalized `mac-sim` asset and articulation support
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
