from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

# Import terrain generator configurations. 
from Blind_Locomotion_Go1.tasks.manager_based.blind_locomotion_go1.Blind_Locomotion_Terrain import Blind_Locomotion_Terrains_config

@configclass
class Blind_Locomotion_sceneCfg(InteractiveSceneCfg):
    # Terrain configurations. 
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=Blind_Locomotion_Terrains_config,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # Specific robot will be left abstracted. 
    robot: ArticulationCfg = MISSING

    # Sensors.
    # Note here that the primitive base is only a descriptive name. 
    # Different robots have different names for the base. 
    # Unitree Go1 has it as: trunk. 
    # Remember to modify this in the robot specific environment or scene cfg. 
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    front_left_ray = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.04, size=[0.08, 0.08]),
        max_distance=1.0,
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    front_right_ray = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.04, size=[0.08, 0.08]),
        max_distance=1.0,
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    rear_left_ray = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RL_foot",
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.04, size=[0.08, 0.08]),
        max_distance=1.0,
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    rear_right_ray = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RR_foot",
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.04, size=[0.08, 0.08]),
        max_distance=1.0,
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # Lighting. 
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
