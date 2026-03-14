# Port To MLX Todo

This file tracks the full CUDA-to-MLX port program for IsaacLab on Apple Silicon.

## Completed Foundations

- [x] Runtime selection seam for `torch-cuda|mlx`
- [x] Simulation selection seam for `isaacsim|mac-sim`
- [x] Kernel selection seam for `warp|metal|cpu`
- [x] Sensor and planner capability adapters
- [x] AppLauncher mac-sim bootstrap mode
- [x] MLX/mac-sim reference slices for cartpole, cart-double-pendulum, quadcopter
- [x] Focused Apple Silicon backend tests
- [x] Public benchmark harness for current MLX task slices
- [x] Public `uv` install path for `source/isaaclab[macos-mlx,dev]`

## Track 1: Packaging And Public Install

- [x] Remove torch-only assumptions from `source/isaaclab_rl/setup.py`
- [x] Move RL framework dependencies into extras instead of base install
- [x] Add install smoke tests for `source/isaaclab_rl` on the mac path
- [ ] Document which extras are required for MLX-only versus upstream CUDA paths
- [ ] Add a support matrix for base package, RL wrappers, tasks, sensors, planners

## Track 2: Import Safety And Capability Gating

- [x] Make `isaaclab_rl.sb3` import-safe on mac without forcing `isaaclab.envs` or torch imports at module import time
- [x] Make `isaaclab_rl.skrl` import-safe on mac without forcing Isaac Sim env imports at module import time
- [x] Split `RmpFlowControllerCfg` out of the heavy controller module so config imports stay mac-safe
- [ ] Audit `isaaclab.controllers` for remaining eager Isaac Sim imports
- [ ] Audit `isaaclab.sim` subpackages for public imports that should be lazy-gated
- [ ] Audit `isaaclab.envs` helpers and config loaders for mac-safe import paths
- [ ] Add explicit capability checks for unsupported sensors, planners, and UI-only components

## Track 3: mac-sim Core Generalization

- [ ] Introduce a reusable articulated-state container shared by mac-sim tasks
- [ ] Generalize joint-space reset/step helpers out of the task-specific envs
- [ ] Add root-state write/read helpers shared by quadcopter and future tasks
- [ ] Add basic terrain representation and collision hooks
- [ ] Add contact modeling needed for the first locomotion task
- [ ] Add a shared env origin/grid manager for large batched scenes

## Track 4: Next Task Ports

- [ ] Port one quadruped locomotion task
- [ ] Port one humanoid locomotion task
- [ ] Port one Franka-style reach task
- [ ] Port one Franka-style lift/manipulation task
- [ ] Add a raycast-driven navigation or perception task
- [ ] Add a benchmark case for each newly ported task

## Track 5: Kernel Replacement

- [ ] Inventory the exact Warp/custom CUDA kernels used by the first target tasks
- [ ] Implement MLX-op replacements where possible
- [ ] Add Metal-backed replacements for hot loops that need throughput
- [ ] Keep CPU fallback only for bring-up and parity debugging
- [ ] Add per-kernel reference tests against upstream outputs

## Track 6: Sensors

- [ ] Add a `mac-sensors` raycast implementation
- [ ] Define the task-level sensor contract for depth outputs
- [ ] Add a basic task-usable camera/depth pipeline
- [ ] Add capability-gated explicit unsupported errors for RTX-only camera features
- [ ] Add sensor parity tests for shapes, frames, and reset behavior

## Track 7: RL And Training

- [ ] Define an MLX-native RL wrapper surface separate from torch-centric wrappers
- [ ] Port a reusable PPO trainer abstraction outside the cartpole-specific slice
- [ ] Add checkpoint compatibility rules across MLX tasks
- [ ] Add evaluation and replay helpers shared by all MLX tasks
- [ ] Add train/infer coverage for at least one locomotion task

## Track 8: CI And Release

- [x] Extend `.github/workflows/mlx-macos.yml` with package install smoke via extras
- [ ] Add benchmark smoke artifact upload for M-series runs
- [ ] Add import-safety jobs that run with no Isaac Sim installed
- [ ] Add nightly drift checks against selected upstream task semantics
- [ ] Publish the support matrix and benchmark expectations in release docs

## Track 9: Follow-On Compatibility

- [ ] Define a planner compatibility layer to replace cuRobo gradually
- [ ] Start ROS 2 interoperability without CUDA/NITROS assumptions
- [ ] Add CPU/Metal transport notes for future Isaac ROS compatibility

## Working Rule

Every concrete success should update both:

- `/Users/ilessio/.codex/skills/port-to-mlx/references/progress_log.md`
- this repo todo when it changes status materially
