# Pseudo command line. 
import argparse

# Import isaac lab and sim app launcher. 
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Quadruped blind locomotion environment. ")
parser.add_argument("--task", type=str, default="Template-Blind-Locomotion-Go1-v0", help="Name of the task/environment.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to spawn.")

# Append cli args to app launcher. 
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments to a local variable. 
args_cli = parser.parse_args()

# Launch omniverse app. 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Library imports. 
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg
import gymnasium as gym
import torch

# Import the module to register it to the gym environment. 
import Blind_Locomotion_Go1.tasks  # noqa: F401

# Define entry point. 
agent_cfg_entry_point = "skrl_cfg_entry_point"


# Main function. 
@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict): 
    # Parse the environment from pseudo cli. 
    print("Preview blind locomotion environment for Unitree Go1. ")
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Create the environment. 
    env = gym.make(args_cli.task, cfg=env_cfg)

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
            actions = 2 * torch.rand(env.action_space.shape) - 1

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
