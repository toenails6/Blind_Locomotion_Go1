# Pseudo command line. 
import argparse

# Import isaac lab and sim app launcher. 
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Quadruped blind locomotion environment. ")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go1-v0", help="Name of the task/environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Library imports. 
from isaaclab_tasks.utils import parse_env_cfg
import gymnasium as gym
import torch


def main(): 
    print("Preview blind locomotion environment for Unitree Go1. ")
    # Create the environment configuration. 
    envCfg = parse_env_cfg(
        args_cli.task, 
        num_envs=args_cli.num_envs
    )

    # Create the environment. 
    env = gym.make(args_cli.task, cfg=envCfg)

    # Reset the environment
    env.reset()

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
            actions = torch.randn_like(env.action_manager.action)

            # Step the environment. 
            obs, rew, terminated, truncated, info = env.step(actions)

            # Legacy cartpole print views. 
            # # Print current orientation of pole
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())

            # Update counter. 
            count += 1

    # Close the environment and simulation app. 
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
