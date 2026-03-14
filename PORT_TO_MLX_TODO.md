# Port To MLX Todo

This file tracks the full CUDA-to-MLX port program for IsaacLab on Apple Silicon.

## Program Shape

- Horizon: multi-day to multi-week porting program
- Cadence: each pass should land one concrete compatibility or runtime win on `main`
- Validation rule: every runtime/import/packaging change must add or extend tests, install smoke, or benchmark smoke
- Memory rule: every meaningful win must be appended to `/Users/ilessio/.codex/skills/port-to-mlx/references/progress_log.md`

## Phase Breakdown

### Phase A. Import And Install Safety

Estimated effort: 8-16 focused hours

- Goal: make the public MLX/mac path install and import cleanly without Isaac Sim
- Exit criteria:
  - editable install works for `source/isaaclab[macos-mlx,dev]`
  - base `source/isaaclab_rl[dev]` can coexist without forcing framework extras
  - public config modules import without `carb`, `omni`, `isaacsim`, or `warp`

### Phase B. Shared Runtime Abstractions

Estimated effort: 10-20 focused hours

- Goal: move shared runtime assumptions behind explicit adapters
- Exit criteria:
  - compute, kernel, sim, planner, and sensor seams are the only supported backend touchpoints
  - unsupported features fail through capability checks
  - benchmark and diagnostics surfaces report the selected backend contract

### Phase C. mac-sim Core Generalization

Estimated effort: 20-40 focused hours

- Goal: replace task-local simulator logic with reusable batched simulator primitives
- Exit criteria:
  - shared articulation/root-state layer exists
  - resets, state writes, origins, and common reward/termination helpers are centralized
  - first locomotion task can reuse the shared simulator substrate

### Phase D. Task Port Expansion

Estimated effort: 30-60 focused hours

- Goal: expand beyond the current cartpole/cart-double-pendulum/quadcopter slice
- Exit criteria:
  - at least one quadruped locomotion task
  - at least one humanoid task
  - at least one manipulation reach/lift task
  - each new task has replay/train smoke and benchmark coverage

### Phase E. Kernel And Sensor Replacement

Estimated effort: 30-80 focused hours

- Goal: replace the first important Warp/CUDA kernels and add task-usable sensors
- Exit criteria:
  - raycast path exists on mac
  - hot loops have MLX-op or Metal implementations
  - benchmarked paths do not silently fall back to CPU

### Phase F. Public Release Hardening

Estimated effort: 8-20 focused hours

- Goal: make the fork publishable and maintainable
- Exit criteria:
  - CI proves install/import/runtime smoke on Apple Silicon
  - benchmark artifacts are archived
  - support matrix and known limitations are documented

## Active Queue

These are the highest-priority items to keep moving without waiting for deeper architecture work:

1. Keep stripping config-only imports away from heavy runtime modules.
2. Keep shrinking the set of modules that require `carb`, `omni`, `isaacsim`, or `warp` at import time.
3. Generalize the current mac-sim task slices into shared batched simulator primitives.
4. Start the first locomotion port as soon as the shared articulation substrate is ready.
5. Add benchmark artifact collection and trend comparison for M-series runs.

## Definitions Of Done

- Import-safe:
  The module imports on `mlx + mac-sim` without Isaac Sim installed.
- Packaging-safe:
  The documented `uv` install path works in a fresh environment.
- Runtime-safe:
  The code path either runs correctly on `mlx + mac-sim` or raises an explicit unsupported backend/capability error.
- Ported task:
  The task has reset/step smoke, replay smoke, and benchmark coverage.
- Performance-ready:
  The benchmarked path is free of accidental CPU fallback.

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
- [ ] Move any remaining config-only constants away from heavy runtime helper modules

## Track 2: Import Safety And Capability Gating

- [x] Make `isaaclab_rl.sb3` import-safe on mac without forcing `isaaclab.envs` or torch imports at module import time
- [x] Make `isaaclab_rl.skrl` import-safe on mac without forcing Isaac Sim env imports at module import time
- [x] Split `RmpFlowControllerCfg` out of the heavy controller module so config imports stay mac-safe
- [x] Split Nucleus path constants out of the heavy `isaaclab.utils.assets` module
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
