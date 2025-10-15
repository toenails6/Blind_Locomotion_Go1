# Library imports. 
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG

# Import training environment definition. 
from Blind_Locomotion_Go1.tasks.manager_based.blind_locomotion_go1.Blind_Locomotion_env import BlindLocomotionCfg


@configclass
class UnitreeGo1_BlindLocomotionEnvCfg(BlindLocomotionCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Assign Unitree Go1 Robot assets. 
        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Correct the primitive name for the ray-casters. 
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        self.scene.rear_left_ray.prim_path = "{ENV_REGEX_NS}/Robot/RL_foot"
        self.scene.rear_right_ray.prim_path = "{ENV_REGEX_NS}/Robot/RR_foot"
        self.scene.front_left_ray.prim_path = "{ENV_REGEX_NS}/Robot/FL_foot"
        self.scene.front_right_ray.prim_path = "{ENV_REGEX_NS}/Robot/FR_foot"

        # Scale down terrains according to robot scale. 
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # Reduce action scale. 
        self.actions.joint_pos.scale = 0.25

        # Modify events. 
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Modify rewards and set reward weights. 
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # Correct the primitive name for base contact termination definition. 
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"

# Example play configuration from official repositories. 
# Smaller scene and no randomization.
# @configclass
# class UnitreeGo1RoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         # make a smaller scene for play
#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.5
#         # spawn the robot randomly in the grid (instead of their terrain levels)
#         self.scene.terrain.max_init_terrain_level = None
#         # reduce the number of terrains to save memory
#         if self.scene.terrain.terrain_generator is not None:
#             self.scene.terrain.terrain_generator.num_rows = 5
#             self.scene.terrain.terrain_generator.num_cols = 5
#             self.scene.terrain.terrain_generator.curriculum = False

#         # disable randomization for play
#         self.observations.policy.enable_corruption = False
#         # remove random pushing event
#         self.events.base_external_force_torque = None
#         self.events.push_robot = None
