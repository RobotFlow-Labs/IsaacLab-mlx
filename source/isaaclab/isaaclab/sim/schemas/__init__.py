# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for schemas used in Omniverse.

We wrap the USD schemas for PhysX and USD Physics in a more convenient API for setting the parameters from
Python. This is done so that configuration objects can define the schema properties to set and make it easier
to tune the physics parameters without requiring to open Omniverse Kit and manually set the parameters into
the respective USD attributes.

.. caution::

    Schema properties cannot be applied on prims that are prototypes as they are read-only prims. This
    particularly affects instanced assets where some of the prims (usually the visual and collision meshes)
    are prototypes so that the instancing can be done efficiently.

    In such cases, it is assumed that the prototypes have sim-ready properties on them that don't need to be modified.
    Trying to set properties into prototypes will throw a warning saying that the prim is a prototype and the
    properties cannot be set.

The schemas are defined in the following links:

* `UsdPhysics schema <https://openusd.org/dev/api/usd_physics_page_front.html>`_
* `PhysxSchema schema <https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/index.html>`_

Locally, the schemas are defined in the following files:

* ``_isaac_sim/extsPhysics/omni.usd.schema.physics/plugins/UsdPhysics/resources/UsdPhysics/schema.usda``
* ``_isaac_sim/extsPhysics/omni.usd.schema.physx/plugins/PhysxSchema/resources/generatedSchema.usda``

"""

from __future__ import annotations

import importlib

from isaaclab.backends import UnsupportedBackendError, current_runtime

_SAFE_EXPORTS = {
    "ArticulationRootPropertiesCfg": (".schemas_cfg", "ArticulationRootPropertiesCfg"),
    "BoundingCubePropertiesCfg": (".schemas_cfg", "BoundingCubePropertiesCfg"),
    "BoundingSpherePropertiesCfg": (".schemas_cfg", "BoundingSpherePropertiesCfg"),
    "CollisionPropertiesCfg": (".schemas_cfg", "CollisionPropertiesCfg"),
    "ConvexDecompositionPropertiesCfg": (".schemas_cfg", "ConvexDecompositionPropertiesCfg"),
    "ConvexHullPropertiesCfg": (".schemas_cfg", "ConvexHullPropertiesCfg"),
    "DeformableBodyPropertiesCfg": (".schemas_cfg", "DeformableBodyPropertiesCfg"),
    "FixedTendonPropertiesCfg": (".schemas_cfg", "FixedTendonPropertiesCfg"),
    "JointDrivePropertiesCfg": (".schemas_cfg", "JointDrivePropertiesCfg"),
    "MassPropertiesCfg": (".schemas_cfg", "MassPropertiesCfg"),
    "MeshCollisionPropertiesCfg": (".schemas_cfg", "MeshCollisionPropertiesCfg"),
    "RigidBodyPropertiesCfg": (".schemas_cfg", "RigidBodyPropertiesCfg"),
    "SDFMeshPropertiesCfg": (".schemas_cfg", "SDFMeshPropertiesCfg"),
    "SpatialTendonPropertiesCfg": (".schemas_cfg", "SpatialTendonPropertiesCfg"),
    "TriangleMeshPropertiesCfg": (".schemas_cfg", "TriangleMeshPropertiesCfg"),
    "TriangleMeshSimplificationPropertiesCfg": (".schemas_cfg", "TriangleMeshSimplificationPropertiesCfg"),
}
_ISAACSIM_EXPORTS = {
    "MESH_APPROXIMATION_TOKENS": (".schemas", "MESH_APPROXIMATION_TOKENS"),
    "PHYSX_MESH_COLLISION_CFGS": (".schemas", "PHYSX_MESH_COLLISION_CFGS"),
    "USD_MESH_COLLISION_CFGS": (".schemas", "USD_MESH_COLLISION_CFGS"),
    "activate_contact_sensors": (".schemas", "activate_contact_sensors"),
    "define_articulation_root_properties": (".schemas", "define_articulation_root_properties"),
    "define_collision_properties": (".schemas", "define_collision_properties"),
    "define_deformable_body_properties": (".schemas", "define_deformable_body_properties"),
    "define_mass_properties": (".schemas", "define_mass_properties"),
    "define_mesh_collision_properties": (".schemas", "define_mesh_collision_properties"),
    "define_rigid_body_properties": (".schemas", "define_rigid_body_properties"),
    "modify_articulation_root_properties": (".schemas", "modify_articulation_root_properties"),
    "modify_collision_properties": (".schemas", "modify_collision_properties"),
    "modify_deformable_body_properties": (".schemas", "modify_deformable_body_properties"),
    "modify_fixed_tendon_properties": (".schemas", "modify_fixed_tendon_properties"),
    "modify_joint_drive_properties": (".schemas", "modify_joint_drive_properties"),
    "modify_mass_properties": (".schemas", "modify_mass_properties"),
    "modify_mesh_collision_properties": (".schemas", "modify_mesh_collision_properties"),
    "modify_rigid_body_properties": (".schemas", "modify_rigid_body_properties"),
    "modify_spatial_tendon_properties": (".schemas", "modify_spatial_tendon_properties"),
}

__all__ = [*_SAFE_EXPORTS.keys(), *_ISAACSIM_EXPORTS.keys()]


def __getattr__(name: str):
    target = _SAFE_EXPORTS.get(name)
    if target is not None:
        module = importlib.import_module(target[0], __name__)
        value = getattr(module, target[1])
        globals()[name] = value
        return value

    if current_runtime().sim_backend != "isaacsim":
        raise UnsupportedBackendError(
            f"`isaaclab.sim.schemas.{name}` currently requires `sim-backend=isaacsim`."
            " Schema configuration objects are available in the `mac-sim` bootstrap path, but USD/PhysX schema"
            " mutation utilities remain Isaac Sim only."
        )

    target = _ISAACSIM_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
