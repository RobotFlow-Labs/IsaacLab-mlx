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

- Status: `READY`
- Depends on: `MLX-IMPORT-003`
- Title: Audit `isaaclab.markers` config surfaces for any remaining heavy imports
- Acceptance:
  - config modules use lightweight constants/helpers where possible

### MLX-IMPORT-006

- Status: `READY`
- Depends on: `MLX-IMPORT-003`
- Title: Audit `isaaclab.devices.openxr` retargeter config/helpers for constant-only imports
- Acceptance:
  - config constants come from lightweight modules
  - heavy runtime helpers stay lazy

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

- Status: `READY`
- Depends on: `MLX-IMPORT-007`
- Title: Publish a definitive “import-safe on mac” module surface
- Acceptance:
  - documented in README
  - kept in sync with tests

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

- Status: `ACTIVE`
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
  - remaining high-value families include shadow-hand vision and additional manager-based manipulation/locomanipulation clusters

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

- Status: `ACTIVE`
- Depends on: `MLX-PKG-003`
- Title: Add fresh-env install smoke for documented public paths in CI
- Acceptance:
  - MLX base install smoke
  - MLX + RL base install smoke
  - upstream CUDA path remains unaffected

## Phase B: Shared mac-sim Substrate

### MLX-SIM-001

- Status: `READY`
- Title: Extract shared articulated joint-state container from cartpole/cart-double-pendulum
- Acceptance:
  - no duplicated joint state reset/write logic across those envs

### MLX-SIM-002

- Status: `READY`
- Depends on: `MLX-SIM-001`
- Title: Extract shared root-state container from quadcopter path
- Acceptance:
  - root pose/velocity read/write helpers centralized

### MLX-SIM-003

- Status: `READY`
- Depends on: `MLX-SIM-001`, `MLX-SIM-002`
- Title: Create reusable mac-sim batched state primitives module
- Acceptance:
  - cartpole/cart-double-pendulum/quadcopter use the shared substrate

### MLX-SIM-004

- Status: `READY`
- Depends on: `MLX-SIM-003`
- Title: Add reusable environment origin/grid manager

### MLX-SIM-005

- Status: `READY`
- Depends on: `MLX-SIM-003`
- Title: Add terrain representation for the first locomotion task

### MLX-SIM-006

- Status: `BLOCKED`
- Depends on: `MLX-SIM-005`
- Title: Add contact approximation model sufficient for first locomotion bring-up

### MLX-SIM-007

- Status: `BLOCKED`
- Depends on: `MLX-SIM-006`
- Title: Add contact-oriented reward/termination utilities

### MLX-SIM-008

- Status: `READY`
- Depends on: `MLX-SIM-003`
- Title: Centralize reset sampling helpers and determinism controls

### MLX-SIM-009

- Status: `READY`
- Depends on: `MLX-SIM-003`
- Title: Add capability reporting from concrete mac-sim adapters into benchmarks and diagnostics

### MLX-SIM-010

- Status: `READY`
- Depends on: `MLX-SIM-003`
- Title: Add shared replay/rollout helpers for mac-native tasks

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

- Status: `BLOCKED`
- Depends on: `MLX-SIM-005`, `MLX-SIM-006`, `MLX-SIM-007`
- Title: Port first quadruped locomotion task
- Suggested target:
  - pick the simplest existing velocity locomotion task with minimal sensor coupling

### MLX-TASK-005

- Status: `BLOCKED`
- Depends on: `MLX-TASK-004`
- Title: Add quadruped replay smoke and benchmark

### MLX-TASK-006

- Status: `BLOCKED`
- Depends on: `MLX-TASK-004`
- Title: Add quadruped training smoke on MLX

### MLX-TASK-007

- Status: `LATER`
- Depends on: `MLX-TASK-004`
- Title: Port first humanoid locomotion task

### MLX-TASK-008

- Status: `LATER`
- Depends on: `MLX-SIM-003`
- Title: Port first manipulation reach task

### MLX-TASK-009

- Status: `LATER`
- Depends on: `MLX-TASK-008`
- Title: Port first manipulation lift task

### MLX-TASK-010

- Status: `LATER`
- Depends on: `MLX-SENSOR-001`
- Title: Port first raycast-driven task

### MLX-TASK-011

- Status: `READY`
- Title: Keep all current task slices benchmarked on every major substrate change

### MLX-TASK-012

- Status: `READY`
- Title: Keep checkpoint/replay contracts stable across all mac-native tasks

## Phase D: Kernel Replacement

### MLX-KERNEL-001

- Status: `READY`
- Title: Inventory first real Warp/custom CUDA kernels needed by planned locomotion target

### MLX-KERNEL-002

- Status: `READY`
- Depends on: `MLX-KERNEL-001`
- Title: Implement MLX-op replacements for non-hot helper kernels

### MLX-KERNEL-003

- Status: `BLOCKED`
- Depends on: `MLX-KERNEL-001`
- Title: Implement Metal-backed replacements for locomotion hot loops

### MLX-KERNEL-004

- Status: `READY`
- Depends on: `MLX-KERNEL-001`
- Title: Add per-kernel parity tests against upstream outputs where feasible

### MLX-KERNEL-005

- Status: `READY`
- Title: Add benchmark reporting that detects accidental CPU fallback

### MLX-KERNEL-006

- Status: `LATER`
- Title: Inventory raycast kernels for future sensor port

### MLX-KERNEL-007

- Status: `LATER`
- Title: Inventory camera/tiled-camera reshape kernels

### MLX-KERNEL-008

- Status: `LATER`
- Title: Create a shared kernel-compat layer instead of scattered replacements

## Phase E: Sensors

### MLX-SENSOR-001

- Status: `LATER`
- Title: Implement `mac-sensors` raycast substrate

### MLX-SENSOR-002

- Status: `LATER`
- Depends on: `MLX-SENSOR-001`
- Title: Add raycast parity tests and benchmark

### MLX-SENSOR-003

- Status: `LATER`
- Title: Define minimal depth output contract for task-usable cameras

### MLX-SENSOR-004

- Status: `LATER`
- Depends on: `MLX-SENSOR-003`
- Title: Add basic camera/depth path for non-RTX tasks

### MLX-SENSOR-005

- Status: `READY`
- Title: Ensure unsupported camera/RTX features fail explicitly via capability checks

### MLX-SENSOR-006

- Status: `LATER`
- Title: Add benchmark coverage for sensor-heavy mac-native tasks

## Phase F: RL And Training

### MLX-RL-001

- Status: `READY`
- Title: Extract reusable PPO trainer substrate from cartpole-specific code

### MLX-RL-002

- Status: `READY`
- Depends on: `MLX-RL-001`
- Title: Define shared MLX policy/checkpoint format for mac-native tasks

### MLX-RL-003

- Status: `BLOCKED`
- Depends on: `MLX-TASK-004`, `MLX-RL-001`
- Title: Train first locomotion task on MLX

### MLX-RL-004

- Status: `READY`
- Title: Add shared replay/eval scripts for all MLX task slices

### MLX-RL-005

- Status: `LATER`
- Title: Define MLX-native wrapper surface instead of relying on torch-centric RL wrappers

### MLX-RL-006

- Status: `LATER`
- Title: Add multi-task benchmark/training dashboard output

## Phase G: CI And Release

### MLX-CI-001

- Status: `DONE`
- Title: Add MLX macOS smoke workflow

### MLX-CI-002

- Status: `READY`
- Title: Add benchmark smoke run and artifact upload to MLX macOS workflow

### MLX-CI-003

- Status: `READY`
- Depends on: `MLX-CI-002`
- Title: Add import-safety lane that proves no Isaac Sim install is required

### MLX-CI-004

- Status: `LATER`
- Title: Add nightly drift checks against selected upstream task semantics

### MLX-CI-005

- Status: `READY`
- Title: Publish support matrix and benchmark expectations in README

### MLX-CI-006

- Status: `LATER`
- Title: Archive benchmark trend JSON for M-series comparisons

## Phase H: Follow-On Compatibility

### MLX-ROS-001

- Status: `LATER`
- Title: Define planner compatibility seam to replace cuRobo progressively

### MLX-ROS-002

- Status: `LATER`
- Title: Start plain ROS 2 process/message interoperability without CUDA assumptions

### MLX-ROS-003

- Status: `LATER`
- Title: Document future CPU/Metal transport path for Isaac ROS compatibility

## Continuous Work Queue

This queue exists so work can continue without waiting for a new plan:

- If `MLX-IMPORT-003` is incomplete, keep taking the next import blocker from `isaaclab.sim`.
- If `MLX-IMPORT-003` is complete, move directly to `MLX-IMPORT-004`.
- Once the import-safety pass stops yielding high-value wins, start `MLX-SIM-001`.
- Once `MLX-SIM-003` is complete, immediately start `MLX-TASK-004`.

## Validation Commands

```bash
PYTHONPATH=.:source/isaaclab:source/isaaclab_rl .venv/bin/pytest \
  source/isaaclab_rl/test/test_import_safety.py \
  source/isaaclab/test/backends/test_runtime.py \
  source/isaaclab/test/backends/test_portability_utils.py \
  source/isaaclab/test/backends/test_mac_cartpole.py \
  source/isaaclab/test/backends/test_mac_cartpole_showcase.py \
  source/isaaclab/test/backends/test_mac_cart_double_pendulum.py \
  source/isaaclab/test/backends/test_mac_quadcopter.py -q
```

```bash
PYTHONPATH=.:source/isaaclab .venv/bin/python \
  scripts/benchmarks/mlx/benchmark_mac_tasks.py \
  --tasks cartpole cart-double-pendulum quadcopter train-cartpole
```
