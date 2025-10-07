# Library imports
import torch
# from omni.isaac.lab.sim import SimulationContext, SimulationCfg
# from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.envs import ManagerBasedEnv

# Import environment configuration.
from Blind_Locomotion_Go1.tasks.manager_based.blind_locomotion_go1.Blind_Locomotion_Robots import UnitreeGo1_BlindLocomotionEnvCfg

def main():
    """Main function to test the blind locomotion environment."""
    # Simulation control handles. 
    # Remember to adjust the device as needed.
    sim_cfg = SimulationCfg(device="cuda:0")
    sim = SimulationContext(sim_cfg)

    # Configure the scene with the blind locomotion environment.
    env_cfg = UnitreeGo1_BlindLocomotionEnvCfg()
    env_cfg.scene.num_envs = 2
    env_cfg.sim.device = "cuda:0"
    env = ManagerBasedEnv(cfg=env_cfg)

    # Reset and play the simulation
    env.reset()
    print("[INFO]: Scene setup complete. Environment spawned in active stage.")

    # sim_dt = sim.get_physics_dt()
    step_count = 0
    max_steps = 1E3

    while sim.is_playing() and step_count < max_steps:
        if step_count % 100 == 0:
            # Periodic reset for consistent testing
            env.reset()  
            print(f"[INFO]: Reset at step {step_count}")

        # Sample random actions. 
        joint_pos = torch.randn_like(env.action_manager.action)
        # step the environment
        env.step(joint_pos)
        
        step_count += 1

        pass

    print("[INFO]: Interactive test ended.")
    env.close()


if __name__ == "__main__":
    main()
