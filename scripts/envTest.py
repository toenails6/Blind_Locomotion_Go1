# Import blind locomotion environment. 
import os
current_working_directory = os.getcwd()
print(f"Current working directory: {current_working_directory}")
from source.Blind_Locomotion_Go1.Blind_Locomotion_Go1.tasks.manager_based.blind_locomotion_go1.blind_locomotion_go1_env_cfg import UnitreeGo1_BlindLocomotionEnvCfg

# Isaaclab pseudo command line imports. 
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Quadruped blind locomotion environment. ")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Library imports. 
from isaaclab.envs import ManagerBasedRLEnv
import torch


def main(): 
    print("Testing blind locomotion environment for Unitree Go1. ")
    envCfg = UnitreeGo1_BlindLocomotionEnvCfg()
    envCfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=envCfg)

    # Environment and robot preview. 
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Reset. 
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            # Sample random actions. 
            joint_vel = torch.randn_like(env.action_manager.action)

            # Step the environment. 
            obs, rew, terminated, truncated, info = env.step(joint_vel)

            # Legacy cartpole print views. 
            # # Print current orientation of pole
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())

            # Update counter. 
            count += 1

    # Close the environment. 
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
