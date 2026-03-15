# Port To MLX Task Board

This file is the execution backlog for the full CUDA-to-MLX port of IsaacLab on Apple Silicon.

It is intentionally written as a task source, not as a loose idea list. The goal is to keep shipping work on `main`
without pausing for replanning after every small success.

## Working Rules

- Always take the next unblocked task from the `Execution Order` section.
- Every merged task must update:
  - this file
  - `/Users/ilessio/.codex/skills/port-to-mlx/references/progress_log.md`
- Every compatibility task needs at least one of:
  - import-safety test
  - install smoke
  - benchmark smoke
  - replay/train smoke
- “Done” means:
  - code landed on `main`
  - validation ran
  - follow-on tasks updated

## Status Legend

- `DONE`: landed and validated
- `ACTIVE`: current execution focus
- `READY`: unblocked and next in line
- `BLOCKED`: depends on earlier tasks
- `LATER`: intentionally deferred until the earlier substrate exists

## Execution Order

1. `MLX-IMPORT-001` through `MLX-IMPORT-008`
2. `MLX-PKG-001` through `MLX-PKG-004`
3. `MLX-SIM-001` through `MLX-SIM-010`
4. `MLX-TASK-001` through `MLX-TASK-012`
5. `MLX-KERNEL-001` through `MLX-KERNEL-008`
6. `MLX-SENSOR-001` through `MLX-SENSOR-006`
7. `MLX-RL-001` through `MLX-RL-006`
8. `MLX-CI-001` through `MLX-CI-006`
9. `MLX-ROS-001` through `MLX-ROS-003`

## Current Wins

- `DONE` Runtime backend seam for `torch-cuda|mlx`
- `DONE` Sim backend seam for `isaacsim|mac-sim`
- `DONE` Kernel backend seam for `warp|metal|cpu`
- `DONE` AppLauncher mac-sim bootstrap mode
- `DONE` MLX/mac-sim slices for cartpole, cart-double-pendulum, quadcopter
- `DONE` Focused backend tests and benchmark harness
- `DONE` Base package extras for `macos-mlx`, `cuda-isaacsim`, and `dev`
- `DONE` RL package base install no longer forces framework extras
- `DONE` `isaaclab_rl.sb3` and `isaaclab_rl.skrl` import-safe on MLX/mac path
- `DONE` `RmpFlowControllerCfg` split from the heavy controller implementation
- `DONE` Lightweight Nucleus constant module split from the heavy assets helper
- `DONE` `isaaclab.sim` config surfaces for schemas, converters, and file spawners import on `mlx + mac-sim`
- `DONE` `isaaclab.envs.mdp.actions` config surfaces import on `mlx + mac-sim` without runtime action modules
- `DONE` macOS install/import smoke now writes retained JSON artifacts and benchmark results in CI
- `DONE` `isaaclab_tasks` lazy mac task registry for cartpole, cart-double-pendulum, and quadcopter
- `DONE` clean MLX/mac base install no longer pulls `torch`, `onnx`, `hidapi`, `transformers`, `starlette`, `tensorboard`, `numba`, or `moviepy`
- `DONE` shared config/helper imports for `envs.common`, `utils.noise`, `utils.io`, `utils.types`, and `envs.utils.spaces` are now torch-free on the mac bootstrap path
- `DONE` a first static manifest now keeps AutoMate, Factory, FORGE, Franka Cabinet, and manager-based pick-place task IDs discoverable on mac while gating them behind explicit `sim-backend=isaacsim` errors
- `DONE` shared mac-sim batched state primitives now back cartpole, cart-double-pendulum, and quadcopter
- `DONE` reusable environment origin/grid helper now backs quadcopter root-state resets and goal sampling
- `DONE` Phase B substrate now includes deterministic reset samplers, flat-terrain primitives, contact approximation buffers, contact-oriented locomotion utilities, and shared rollout/replay helpers
- `DONE` Phase C now includes a first quadruped locomotion slice for `Isaac-Velocity-Flat-Anymal-C-Direct-v0` with replay smoke, benchmark smoke, and MLX PPO train/play scripts
- `DONE` First humanoid locomotion slice landed for `Isaac-Velocity-Flat-H1-v0` with task registry coverage, play/train smokes, and benchmark smoke
- `DONE` Second humanoid locomotion slice landed for `Isaac-Velocity-Rough-H1-v0` with wave terrain, analytic height scans, shared H1 substrate reuse, corrected rough observation-space sizing, benchmark coverage, and deterministic replay tests
- `DONE` First mac-native manipulation slices landed for `Isaac-Reach-Franka-v0` and `Isaac-Lift-Cube-Franka-v0` with lazy registry wiring, public MLX wrapper support, benchmark coverage, and focused backend tests
- `DONE` First trainable manipulation slice landed for `Isaac-Reach-Franka-v0` with shared PPO helpers, checkpoint/replay support, public MLX wrapper training, and CI smoke coverage
- `DONE` Second trainable manipulation slice landed for `Isaac-Lift-Cube-Franka-v0` with shared PPO helpers, checkpoint/replay support, public MLX wrapper training, and CI smoke coverage
- `DONE` Third trainable manipulation slice landed for `Isaac-Stack-Cube-Franka-v0` with compiled stack hotpath helpers, shared PPO/checkpoint contracts, public MLX wrapper training, benchmark coverage, semantic baseline refresh, and CI smoke coverage
- `DONE` Fourth trainable stack variant landed for `Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0` with a reduced three-cube sequential stack backend, staged terminal benchmark metrics, shared PPO/checkpoint contracts, public MLX wrapper training, direct thin CLI wrappers, semantic baseline refresh, and CI smoke coverage
- `DONE` Fourth trainable manipulation slice landed for `Isaac-Franka-Cabinet-Direct-v0` with a reduced drawer workflow, compiled cabinet hotpath helper, shared PPO/checkpoint contracts, public MLX wrapper training, benchmark coverage, semantic baseline refresh, and CI smoke coverage
- `DONE` Sixth trainable Franka manipulation slice landed for `Isaac-Open-Drawer-Franka-v0` with a reduced analytic drawer substrate, public MLX wrapper/CLI exposure, benchmark coverage, refreshed semantic baseline, and focused backend tests
- `DONE` First raycast-driven mac-native task landed for `Isaac-Velocity-Rough-Anymal-C-Direct-v0` with procedural wave terrain, analytic terrain raycasts, benchmark coverage, and deterministic replay tests
- `DONE` Rough locomotion slices for ANYmal-C and H1 now expose full MLX PPO train/replay surfaces with rough-task checkpoint metadata, wrapper coverage, and CI smoke coverage
- `DONE` Synthetic cartpole RGB/depth camera slices landed as eval-only mac-native tasks with deterministic analytic `100x100` observations, public MLX wrapper exposure, sensor benchmark coverage, and CI smoke coverage
- `DONE` Locomotion hotpaths now combine a true Metal-backed root-step kernel with compiled MLX contact/support helpers, and benchmark/semantic reports surface `hotpath: "mlx-metal-root-step"` for the ANYmal-C and H1 slices while Franka keeps its own family-specific hotpath label
- `DONE` Benchmark coverage for the current mac-native task set now lives behind a stable `current-mac-native` benchmark group enforced by tests and CI
- `DONE` Checkpoint/resume and replay contracts are now explicitly covered across the current mac-native task slices
- `DONE` Maintained kernel inventory now maps the next real Warp/CUDA families to source files, target task classes, and replacement strategies
- `DONE` Shared MLX task CLIs now cover train and replay/eval flows across the current mac-native slices
- `DONE` Installed console entry points now expose MLX train/eval/runtime diagnostics without `PYTHONPATH`, and CI release smoke exercises them from a clean `uv` environment
- `DONE` CI now has a dedicated import-safety lane plus benchmark artifact validation for the MLX/mac path
- `DONE` Planner and ROS compatibility prototypes now exist as explicit backend seams with software smokes, backend tests, and capability-gated docs
- `DONE` Planner and ROS compatibility now carry richer planner world-state obstacles plus ROS-friendly world-state / timed joint-trajectory envelopes without requiring ROS Python bindings
- `DONE` CI now proves a release-style MLX install path without `dev` extras or `PYTHONPATH`, then parses rough locomotion/manipulation configs and exercises the public wrapper
- `DONE` Generic `mac-sensors` capability metadata is now honest about the public runtime surface: analytic raycasts plus synthetic camera task slices and backend-local external stereo capture, not generic Isaac Sim camera parity
- `DONE` Supported public MLX/mac tasks now come from a shared typed manifest with a runtime diagnostics CLI so wrapper task lists, runtime diagnostics, and the public task surface stay aligned without hand-maintained duplication
- `DONE` Benchmark group ownership now lives behind the same typed manifest, with runtime diagnostics separating public benchmark groups from benchmark-only projections so sensor/training benchmark rows do not masquerade as public tasks
- `DONE` `mac-sim` now includes a shared generic batched articulation/scene substrate for reset/step and joint/root-state IO, while task-specific contacts, sensors, and reward logic still layer on top
- `DONE` Env/runtime diagnostics now prove the articulated `mac-sim` contract on a real locomotion backend instead of only reporting the high-level backend seam
- `DONE` ROS/planner software smokes now exercise the real `mac-planners` backend and verify typed round-trip reconstruction of planner world-state and joint trajectories
- `DONE` Planner/ROS batch helpers now restore batches by `batch_index` and report actual batch envelope counts instead of inferring from message order or `max(index) + 1`
- `DONE` Stereo/depth smoke now validates raw capture artifacts before processing and writes a machine-checkable JSON summary artifact
- `DONE` `uv run scripts/bootstrap_uv_mlx.py` now bootstraps the public MLX/mac editable environment in one command
- `DONE` Upstream-compatible Franka reach/stack/open-drawer controller variants now resolve to the canonical mac-native manipulation slices through the lazy task registry, public MLX wrapper, and installed CLI without aliasing heavier visuomotor or blueprint task families
- `DONE` Upstream-compatible Franka lift IK variants now resolve to the canonical mac-native lift slice, while teddy-bear lift and richer Franka stack visuomotor/cosmos/blueprint/skillgen/bin-mimic families remain discoverable through explicit `sim-backend=isaacsim` gating on mac

## Phase A: Import And Packaging Safety

### MLX-IMPORT-001

- Status: `DONE`
- Title: Split controller config dataclasses away from heavy Isaac Sim runtime modules
- Validation:
  - focused backend tests
  - fresh import smoke

### MLX-IMPORT-002

- Status: `DONE`
- Title: Split Nucleus constants away from `isaaclab.utils.assets`
- Validation:
  - focused backend tests
  - fresh import smoke

### MLX-IMPORT-003

- Status: `DONE`
- Title: Audit `isaaclab.sim` public config/helper imports and isolate config-safe surfaces
- Scope:
  - `isaaclab.sim.__init__`
  - `isaaclab.sim.schemas`
  - `isaaclab.sim.spawners.*_cfg`
  - `isaaclab.sim.utils` modules that should be config-safe
- Acceptance:
  - config-only imports do not pull `carb`, `omni`, `isaacsim`, or `pxr` unless explicitly upstream-only
  - tests prove the new import-safe path
- Validation:
  - focused backend suite
  - torch-free local install/import smoke
  - local benchmark smoke JSON output

### MLX-IMPORT-004

- Status: `DONE`
- Depends on: `MLX-IMPORT-003`
- Title: Audit `isaaclab.envs.mdp.actions` config surfaces for config-only imports that still pull runtime-heavy modules
- Acceptance:
  - action config modules import on `mlx + mac-sim`
  - unsupported runtime paths fail when instantiated, not when imported
- Validation:
  - focused backend suite
  - torch-free local install/import smoke

### MLX-IMPORT-005

- Status: `DONE`
- Depends on: `MLX-IMPORT-003`
- Title: Audit `isaaclab.markers` config surfaces for any remaining heavy imports
- Acceptance:
  - config modules use lightweight constants/helpers where possible
- Validation:
  - focused backend suite
  - torch-free import smoke for `isaaclab.markers` and `isaaclab.markers.config`

### MLX-IMPORT-006

- Status: `DONE`
- Depends on: `MLX-IMPORT-003`
- Title: Audit `isaaclab.devices.openxr` retargeter config/helpers for constant-only imports
- Acceptance:
  - config constants come from lightweight modules
  - heavy runtime helpers stay lazy
- Validation:
  - focused backend suite
  - torch-free import smoke for `isaaclab.devices.openxr` and lazy retargeter packages

### MLX-IMPORT-007

- Status: `DONE`
- Depends on: `MLX-IMPORT-003`
- Title: Add explicit import-safety test matrix for public bootstrap modules
- Acceptance:
  - test covers `isaaclab`, `isaaclab.sim`, `isaaclab.controllers`, `isaaclab_rl`
  - runs without Isaac Sim installed
- Validation:
  - `source/isaaclab/test/backends/test_runtime.py`
  - `source/isaaclab_rl/test/test_import_safety.py`
  - torch-free local install/import smoke

### MLX-IMPORT-008

- Status: `DONE`
- Depends on: `MLX-IMPORT-007`
- Title: Publish a definitive “import-safe on mac” module surface
- Acceptance:
  - documented in README
  - kept in sync with tests
- Validation:
  - README import-safe surface section
  - `source/isaaclab/test/backends/test_runtime.py`

### MLX-IMPORT-009

- Status: `DONE`
- Depends on: `MLX-IMPORT-007`
- Title: Replace `isaaclab_tasks` eager recursive package registration with a lazy task registry
- Scope:
  - `source/isaaclab_tasks/isaaclab_tasks/__init__.py`
  - `source/isaaclab_tasks/isaaclab_tasks/utils/importer.py`
- Acceptance:
  - `import isaaclab_tasks` does not recursively import every task package on the MLX/mac path
  - task/config entry points resolve lazily from strings or a registry manifest

### MLX-IMPORT-010

- Status: `DONE`
- Depends on: `MLX-IMPORT-009`
- Title: Split direct-task CUDA/IsaacSim clusters from task package import surfaces
- Scope:
  - AutoMate
  - Factory
  - FORGE
  - first manager-based manipulation config clusters that import `carb` or `isaacsim` at module load
- Acceptance:
  - task packages expose config/registration surfaces without importing `warp`, `carb`, `pxr`, or `isaacsim`
  - runtime-only backends load on demand behind capability checks
- Progress:
  - static manifest/gating completed for AutoMate, Factory, FORGE, Franka Cabinet, and manager-based pick-place
  - static manifest/gating extended to shadow-hand vision, shadow-hand-over, DexSuite, deploy, navigation, and locomanipulation tracking clusters

### MLX-PKG-001

- Status: `DONE`
- Title: Core package extras split for MLX vs CUDA users

### MLX-PKG-002

- Status: `DONE`
- Title: RL package extras split so framework deps are optional

### MLX-PKG-003

- Status: `DONE`
- Title: Add clear support matrix for package extras and runtime combinations
- Acceptance:
  - README section listing supported install combos
  - one install smoke per documented combo

### MLX-PKG-004

- Status: `DONE`
- Depends on: `MLX-PKG-003`
- Title: Add fresh-env install smoke for documented public paths in CI
- Acceptance:
  - MLX base install smoke
  - MLX + RL base install smoke
  - upstream CUDA path remains unaffected
- Validation:
  - fresh-env `uv` install matrix for core, core+tasks, and core+rl

## Phase B: Shared mac-sim Substrate

### MLX-SIM-001

- Status: `DONE`
- Title: Extract shared articulated joint-state container from cartpole/cart-double-pendulum
- Acceptance:
  - no duplicated joint state reset/write logic across those envs
- Validation:
  - `source/isaaclab/test/backends/test_mac_state_primitives.py`
  - cartpole/cart-double-pendulum mac backend smoke tests

### MLX-SIM-002

- Status: `DONE`
- Depends on: `MLX-SIM-001`
- Title: Extract shared root-state container from quadcopter path
- Acceptance:
  - root pose/velocity read/write helpers centralized
- Validation:
  - `source/isaaclab/test/backends/test_mac_state_primitives.py`
  - quadcopter mac backend smoke tests

### MLX-SIM-003

- Status: `DONE`
- Depends on: `MLX-SIM-001`, `MLX-SIM-002`
- Title: Create reusable mac-sim batched state primitives module
- Acceptance:
  - cartpole/cart-double-pendulum/quadcopter use the shared substrate
- Validation:
  - focused backend suite
  - benchmark smoke `logs/benchmarks/mlx/sim-primitives-smoke.json`

### MLX-SIM-004

- Status: `DONE`
- Depends on: `MLX-SIM-003`
- Title: Add reusable environment origin/grid manager
- Validation:
  - quadcopter backend smoke tests
  - benchmark smoke `logs/benchmarks/mlx/sim-primitives-smoke.json`

### MLX-SIM-005

- Status: `DONE`
- Depends on: `MLX-SIM-003`
- Title: Add terrain representation for the first locomotion task
- Validation:
  - `source/isaaclab/test/backends/test_mac_phase_b_support.py`

### MLX-SIM-006

- Status: `DONE`
- Depends on: `MLX-SIM-005`
- Title: Add contact approximation model sufficient for first locomotion bring-up
- Validation:
  - `source/isaaclab/test/backends/test_mac_phase_b_support.py`

### MLX-SIM-007

- Status: `DONE`
- Depends on: `MLX-SIM-006`
- Title: Add contact-oriented reward/termination utilities
- Validation:
  - `source/isaaclab/test/backends/test_mac_phase_b_support.py`

### MLX-SIM-008

- Status: `DONE`
- Depends on: `MLX-SIM-003`
- Title: Centralize reset sampling helpers and determinism controls
- Validation:
  - `source/isaaclab/test/backends/test_mac_phase_b_support.py`
  - focused backend suite

### MLX-SIM-009

- Status: `DONE`
- Depends on: `MLX-SIM-003`
- Title: Add capability reporting from concrete mac-sim adapters into benchmarks and diagnostics
- Validation:
  - `scripts/benchmarks/mlx/benchmark_mac_tasks.py`
  - benchmark smoke JSON output

### MLX-SIM-010

- Status: `DONE`
- Depends on: `MLX-SIM-003`
- Title: Add shared replay/rollout helpers for mac-native tasks
- Validation:
  - `source/isaaclab/test/backends/test_mac_phase_b_support.py`

## Phase C: Task Ports

### MLX-TASK-001

- Status: `DONE`
- Title: Port cartpole

### MLX-TASK-002

- Status: `DONE`
- Title: Port cart-double-pendulum

### MLX-TASK-003

- Status: `DONE`
- Title: Port quadcopter

### MLX-TASK-004

- Status: `DONE`
- Depends on: `MLX-SIM-005`, `MLX-SIM-006`, `MLX-SIM-007`
- Title: Port first quadruped locomotion task
- Suggested target:
  - `Isaac-Velocity-Flat-Anymal-C-Direct-v0`
- Validation:
  - `source/isaaclab/test/backends/test_mac_anymal_c.py`
  - `source/isaaclab/test/backends/test_task_registry.py`
  - `scripts/reinforcement_learning/mlx/play_anymal_c.py`

### MLX-TASK-005

- Status: `DONE`
- Depends on: `MLX-TASK-004`
- Title: Add quadruped replay smoke and benchmark
- Validation:
  - `scripts/reinforcement_learning/mlx/play_anymal_c.py`
  - `scripts/benchmarks/mlx/benchmark_mac_tasks.py`
  - `logs/benchmarks/mlx/anymal-phase-c-smoke.json`

### MLX-TASK-006

- Status: `DONE`
- Depends on: `MLX-TASK-004`
- Title: Add quadruped training smoke on MLX
- Validation:
  - `scripts/reinforcement_learning/mlx/train_anymal_c.py`
  - `source/isaaclab/test/backends/test_mac_anymal_c.py`
  - `.github/workflows/mlx-macos.yml`

### MLX-TASK-007

- Status: `DONE`
- Depends on: `MLX-TASK-004`
- Title: Port first humanoid locomotion task
- Suggested target:
  - `Isaac-Velocity-Flat-H1-v0`
- Validation:
  - `source/isaaclab/test/backends/test_mac_h1.py`
  - `source/isaaclab/test/backends/test_task_registry.py`
  - `scripts/reinforcement_learning/mlx/train_h1.py`
  - `scripts/reinforcement_learning/mlx/play_h1.py`
  - `scripts/benchmarks/mlx/benchmark_mac_tasks.py`

### MLX-TASK-008

- Status: `DONE`
- Depends on: `MLX-SIM-003`
- Title: Port first manipulation reach task
- Suggested target:
  - `Isaac-Reach-Franka-v0`
- Validation:
  - `source/isaaclab/test/backends/test_mac_franka_reach.py`
  - `source/isaaclab/test/backends/test_task_registry.py`
  - `source/isaaclab_rl/test/test_mlx_wrapper.py`
  - `scripts/benchmarks/mlx/benchmark_mac_tasks.py`

### MLX-TASK-009

- Status: `DONE`
- Depends on: `MLX-TASK-008`
- Title: Port first manipulation lift task
- Suggested target:
  - `Isaac-Lift-Cube-Franka-v0`
- Validation:
  - `source/isaaclab/test/backends/test_mac_franka_lift.py`
  - `source/isaaclab/test/backends/test_task_registry.py`
  - `source/isaaclab_rl/test/test_mlx_wrapper.py`
  - `scripts/benchmarks/mlx/benchmark_mac_tasks.py`

### MLX-TASK-010

- Status: `DONE`
- Depends on: `MLX-SENSOR-001`
- Title: Port first raycast-driven task
- Suggested target:
  - `Isaac-Velocity-Rough-Anymal-C-Direct-v0`
- Validation:
  - `source/isaaclab/test/backends/test_mac_anymal_c_rough.py`
  - `source/isaaclab/test/backends/test_task_registry.py`
  - `source/isaaclab_rl/test/test_mlx_wrapper.py`
  - `scripts/benchmarks/mlx/benchmark_mac_tasks.py`

### MLX-TASK-011

- Status: `DONE`
- Title: Keep all current task slices benchmarked on every major substrate change
- Validation:
  - `source/isaaclab/test/backends/test_mac_benchmark_suite.py`
  - `scripts/benchmarks/mlx/benchmark_mac_tasks.py`
  - `.github/workflows/mlx-macos.yml`

### MLX-TASK-012

- Status: `DONE`
- Title: Keep checkpoint/replay contracts stable across all mac-native tasks
- Validation:
  - `source/isaaclab/test/backends/test_mac_cartpole.py`
  - `source/isaaclab/test/backends/test_mac_cart_double_pendulum.py`
  - `source/isaaclab/test/backends/test_mac_quadcopter.py`
  - `source/isaaclab/test/backends/test_mac_anymal_c.py`
  - `source/isaaclab/test/backends/test_mac_h1.py`

## Phase D: Kernel Replacement

### MLX-KERNEL-001

- Status: `DONE`
- Title: Inventory first real Warp/custom CUDA kernels needed by planned locomotion target
- Validation:
  - `source/isaaclab/isaaclab/backends/kernel_inventory.py`
  - `source/isaaclab/test/backends/test_kernel_inventory.py`
  - `scripts/benchmarks/mlx/benchmark_mac_tasks.py`

### MLX-KERNEL-002

- Status: `DONE`
- Depends on: `MLX-KERNEL-001`
- Title: Implement MLX-op replacements for non-hot helper kernels
- Validation:
  - `source/isaaclab/isaaclab/backends/kernel_compat.py`
  - `source/isaaclab/test/backends/test_kernel_compat.py`
  - `source/isaaclab/isaaclab/backends/kernel_inventory.py`

### MLX-KERNEL-003

- Status: `DONE`
- Depends on: `MLX-KERNEL-001`
- Title: Implement Metal-backed replacements for locomotion hot loops
- Progress:
  - `locomotion_root_step_hotpath(...)` now uses a Metal-backed MLX kernel in `source/isaaclab/isaaclab/backends/mac_sim/hotpath.py`
  - contact and support aggregation helpers remain on `mx.compile` until a separate benchmarked Metal tranche is justified
- Validation:
  - `source/isaaclab/isaaclab/backends/mac_sim/hotpath.py`
  - `source/isaaclab/isaaclab/backends/mac_sim/contacts.py`
  - `source/isaaclab/isaaclab/backends/mac_sim/anymal_c.py`
  - `source/isaaclab/isaaclab/backends/mac_sim/h1.py`
  - `source/isaaclab/test/backends/test_mac_hotpath.py`
  - `source/isaaclab/test/backends/test_mac_anymal_c.py`
  - `source/isaaclab/test/backends/test_mac_h1.py`
  - `source/isaaclab/test/backends/test_mac_semantic_drift.py`
  - `scripts/benchmarks/mlx/benchmark_mac_tasks.py`

### MLX-KERNEL-004

- Status: `DONE`
- Depends on: `MLX-KERNEL-001`
- Title: Add per-kernel parity tests against upstream outputs where feasible
- Validation:
  - `source/isaaclab/test/backends/test_kernel_compat.py`
  - `source/isaaclab/test/backends/test_mac_benchmark_suite.py`

### MLX-KERNEL-005

- Status: `DONE`
- Title: Add benchmark reporting that detects accidental CPU fallback
- Validation:
  - `scripts/benchmarks/mlx/benchmark_mac_tasks.py`
  - `source/isaaclab/test/backends/test_kernel_compat.py`
  - `source/isaaclab/test/backends/test_mac_benchmark_suite.py`

### MLX-KERNEL-006

- Status: `DONE`
- Title: Inventory raycast kernels for future sensor port
- Validation:
  - `source/isaaclab/isaaclab/backends/kernel_inventory.py`
  - `source/isaaclab/test/backends/test_kernel_inventory.py`

### MLX-KERNEL-007

- Status: `DONE`
- Title: Inventory camera/tiled-camera reshape kernels
- Validation:
  - `source/isaaclab/isaaclab/backends/kernel_inventory.py`
  - `source/isaaclab/test/backends/test_kernel_inventory.py`

### MLX-KERNEL-008

- Status: `DONE`
- Title: Create a shared kernel-compat layer instead of scattered replacements
- Progress:
  - shared compatibility surface lives in `source/isaaclab/isaaclab/backends/kernel_compat.py`
  - follow-on backend-aware wiring into remaining upstream Warp call sites stays separate from this substrate task
- Validation:
  - `source/isaaclab/isaaclab/backends/kernel_compat.py`
  - `source/isaaclab/test/backends/test_kernel_compat.py`

## Phase E: Sensors

### MLX-SENSOR-001

- Status: `DONE`
- Title: Implement `mac-sensors` raycast substrate
- Validation:
  - `source/isaaclab/isaaclab/backends/mac_sim/sensors.py`
  - `source/isaaclab/test/backends/test_mac_sensor_raycast.py`
  - focused backend suite

### MLX-SENSOR-002

- Status: `DONE`
- Depends on: `MLX-SENSOR-001`
- Title: Add raycast parity tests and benchmark
- Validation:
  - `source/isaaclab/test/backends/test_mac_sensor_raycast.py`
  - `source/isaaclab/test/backends/test_mac_benchmark_suite.py`
  - `logs/benchmarks/mlx/sensor-smoke.json`

### MLX-SENSOR-003

- Status: `DONE`
- Title: Define minimal depth output contract for task-usable cameras
- Validation:
  - `source/isaaclab/isaaclab/backends/mac_sim/stereo_depth.py`
  - `source/isaaclab/test/backends/test_mac_stereo_depth.py`
  - synthetic stereo smoke via `scripts/tools/mac_stereo_depth_smoke.py`

### MLX-SENSOR-004

- Status: `DONE`
- Depends on: `MLX-SENSOR-003`
- Title: Add basic camera/depth path for non-RTX tasks
- Validation:
  - `source/isaaclab/isaaclab/backends/mac_sim/cameras.py`
  - `scripts/tools/probe_mac_camera.py`
  - `scripts/tools/mac_stereo_depth_smoke.py`
  - `source/isaaclab/test/backends/test_mac_camera_capture.py`
  - focused backend suite

### MLX-SENSOR-005

- Status: `DONE`
- Title: Ensure unsupported camera/RTX features fail explicitly via capability checks

### MLX-SENSOR-006

- Status: `DONE`
- Title: Add benchmark coverage for sensor-heavy mac-native tasks
- Progress:
  - `sensor-mac-native` now covers `cartpole-rgb-camera`, `cartpole-depth-camera`, `anymal-c-flat-height-scan`, and `h1-flat-height-scan`
  - benchmark task-group ownership now comes from `supported_tasks.py`, and `training-mac-native` has a direct execution test instead of only indirect `full` coverage
- Validation:
  - `scripts/benchmarks/mlx/benchmark_mac_tasks.py`
  - `.github/workflows/mlx-macos.yml`
  - `logs/benchmarks/mlx/sensor-smoke.json`

## Phase F: RL And Training

### MLX-RL-001

- Status: `DONE`
- Title: Extract reusable PPO trainer substrate from cartpole-specific code

### MLX-RL-002

- Status: `DONE`
- Depends on: `MLX-RL-001`
- Title: Define shared MLX policy/checkpoint format for mac-native tasks

### MLX-RL-003

- Status: `DONE`
- Depends on: `MLX-TASK-004`, `MLX-RL-001`
- Title: Train first locomotion task on MLX
- Validation:
  - `source/isaaclab/test/backends/test_mlx_task_cli.py`
  - shared ANYmal-C train smoke
  - focused backend suite

### MLX-RL-004

- Status: `DONE`
- Title: Add shared replay/eval scripts for all MLX task slices
- Validation:
  - `source/isaaclab/test/backends/test_mlx_task_cli.py`
  - shared cartpole/cart-double-pendulum/quadcopter/H1 eval smokes
  - focused backend suite

### MLX-RL-005

- Status: `DONE`
- Title: Define MLX-native wrapper surface instead of relying on torch-centric RL wrappers
- Validation:
  - `source/isaaclab_rl/isaaclab_rl/mlx.py`
  - `source/isaaclab/test/backends/test_mlx_task_cli.py`
  - `source/isaaclab_rl/test/test_mlx_wrapper.py`
  - focused backend suite

### MLX-RL-006

- Status: `DONE`
- Title: Add multi-task benchmark/training dashboard output
- Validation:
  - `source/isaaclab/test/backends/test_mac_benchmark_suite.py`
  - local `logs/benchmarks/mlx/full-smoke-dashboard.json`
  - focused backend suite

## Phase G: CI And Release

### MLX-CI-001

- Status: `DONE`
- Title: Add MLX macOS smoke workflow

### MLX-CI-002

- Status: `DONE`
- Title: Add benchmark smoke run and artifact upload to MLX macOS workflow
- Validation:
  - `.github/workflows/mlx-macos.yml`
  - `source/isaaclab/test/backends/test_mac_benchmark_suite.py`
  - local benchmark artifact `logs/benchmarks/mlx/shared-cli-smoke.json`

### MLX-CI-003

- Status: `DONE`
- Depends on: `MLX-CI-002`
- Title: Add import-safety lane that proves no Isaac Sim install is required
- Validation:
  - `.github/workflows/mlx-macos.yml`
  - local import-safety artifact `logs/benchmarks/mlx/import-safety/local-specs.json`
  - focused backend suite

### MLX-CI-004

- Status: `DONE`
- Title: Add nightly drift checks against selected upstream task semantics
- Validation:
  - `.github/workflows/mlx-nightly-drift.yml`
  - `source/isaaclab/test/backends/test_mac_semantic_drift.py`
  - local `logs/benchmarks/mlx/full-smoke-semantic-report.json`

### MLX-CI-005

- Status: `DONE`
- Title: Publish support matrix and benchmark expectations in README
- Validation:
  - `README.md`
  - `.github/workflows/mlx-macos.yml`
  - focused backend suite

### MLX-CI-006

- Status: `DONE`
- Title: Archive benchmark trend JSON for M-series comparisons
- Validation:
  - `.github/workflows/mlx-macos.yml`
  - `source/isaaclab/test/backends/test_mac_benchmark_suite.py`
  - local `logs/benchmarks/mlx/full-smoke-trend.json`

## Phase H: Follow-On Compatibility

### MLX-ROS-001

- Status: `DONE`
- Title: Define planner compatibility seam to replace cuRobo progressively
- Validation:
  - `source/isaaclab/test/backends/test_runtime.py`
  - `source/isaaclab/test/backends/test_planner_compat.py`
  - local `logs/planner/mac-planner-smoke.json`

### MLX-ROS-002

- Status: `DONE`
- Title: Start plain ROS 2 process/message interoperability without CUDA assumptions
- Validation:
  - `source/isaaclab/test/backends/test_ros2_bridge.py`
  - `.github/workflows/mlx-macos.yml`
  - local `logs/hardware/ros2-bridge-smoke-summary.json`

### MLX-ROS-003

- Status: `DONE`
- Title: Document future CPU/Metal transport path for Isaac ROS compatibility
- Validation:
  - `README.md`
  - `.github/workflows/mlx-macos.yml`
  - focused backend suite

## Continuous Work Queue

This queue exists so work can continue without waiting for a new plan. The documented v1 board above is now closed for the current public MLX/mac slice, so the next queue is follow-on parity work:

- Hardware validation is now done for the backend-local stereo path against live ZED 2i capture through a camera-authorized Terminal host plus `zed-sdk-mlx`; retained host-local probe artifacts include `/tmp/isaaclab-zed-probe-live-final.json` and `/tmp/isaaclab-zed-probe-live-final.yuv`.
- Port the next manipulation milestone beyond the current six trainable Franka slices, likely a richer cabinet/drawer variant or the next multi-object manipulation workflow.
- Replace the next remaining locomotion or contact/support `mx.compile` helper with a true custom Metal kernel only after the root-step tranche proves benchmark-positive and semantically stable.
- Grow the planner/ROS prototypes carefully: richer process/message interoperability layers around the new world-state and joint-trajectory envelopes while still avoiding CUDA/NITROS assumptions.
- Keep the generic runtime metadata honest: only advertise generic sensor/runtime capabilities that are actually exposed through backend-neutral APIs, and push task-specific or tooling-only support into explicit diagnostic fields instead of broad parity flags.
- Keep manipulation compatibility aliasing honest: widen upstream task-ID coverage only where the reduced mac-native slice still matches the observation/action/checkpoint contract, and keep heavier visuomotor / blueprint / skillgen families explicitly gated instead of quietly remapping them.
- The next manipulation milestone should be a genuinely new reduced mac-native task, not more aliasing. The honest alias/gating boundary for the current Franka family is now in place.

## Validation Commands

```bash
PYTHONPATH=.:source/isaaclab:source/isaaclab_rl .venv/bin/pytest \
  scripts/tools/test/test_bootstrap_isaac_sources.py \
  scripts/tools/test/test_bootstrap_uv_mlx.py \
  source/isaaclab_rl/test/test_import_safety.py \
  source/isaaclab_rl/test/test_mlx_wrapper.py \
  source/isaaclab/test/backends/test_runtime.py \
  source/isaaclab/test/backends/test_task_registry.py \
  source/isaaclab/test/backends/test_kernel_inventory.py \
  source/isaaclab/test/backends/test_kernel_compat.py \
  source/isaaclab/test/backends/test_mac_hotpath.py \
  source/isaaclab/test/backends/test_mac_runtime_diagnostics.py \
  source/isaaclab/test/backends/test_planner_compat.py \
  source/isaaclab/test/backends/test_ros2_bridge.py \
  source/isaaclab/test/backends/test_mac_benchmark_suite.py \
  source/isaaclab/test/backends/test_mac_semantic_drift.py \
  source/isaaclab/test/backends/test_portability_utils.py \
  source/isaaclab/test/backends/test_mac_state_primitives.py \
  source/isaaclab/test/backends/test_mac_phase_b_support.py \
  source/isaaclab/test/backends/test_mac_cartpole.py \
  source/isaaclab/test/backends/test_mac_cartpole_camera.py \
  source/isaaclab/test/backends/test_mac_cartpole_showcase.py \
  source/isaaclab/test/backends/test_mac_cart_double_pendulum.py \
  source/isaaclab/test/backends/test_mac_quadcopter.py \
  source/isaaclab/test/backends/test_mac_anymal_c.py \
  source/isaaclab/test/backends/test_mac_anymal_c_rough.py \
  source/isaaclab/test/backends/test_mac_franka_reach.py \
  source/isaaclab/test/backends/test_mac_franka_lift.py \
  source/isaaclab/test/backends/test_mac_franka_stack.py \
  source/isaaclab/test/backends/test_mac_franka_open_drawer.py \
  source/isaaclab/test/backends/test_mac_h1.py -q
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/benchmarks/mlx/benchmark_mac_tasks.py \
  --task-group full \
  --json-out logs/benchmarks/mlx/smoke.json
```
