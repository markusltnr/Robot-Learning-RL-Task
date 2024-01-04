from rl_modules.rl_agent import RLAgent, PPO
from env.go_env import GOEnv
from rl_modules.storage import Storage
from rl_modules.actor_critic import ActorCritic
import wandb
from datetime import datetime
import os
import torch

def train():
    if torch.cuda.is_available():
        device ='cuda'
    else:
        device = 'cpu'
    log_name = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    # create environment
    go_env = GOEnv(render_mode="rgb-array")
    # create actor critic
    actor_critic = ActorCritic(state_dim=go_env.obs_dim, action_dim=go_env.action_dim).to(device)
    # create storage to save data
    storage = Storage(obs_dim=go_env.obs_dim, action_dim=go_env.action_dim, max_timesteps=1000, gamma=0.95)
    rl_agent = PPO(env=go_env, actor_critic=actor_critic, storage=storage, device=device, lr=0.00025, action_scale=0.3)
    #rl_agent.load_model('/home/markus/Documents/Uni/RobotLearning/RL_Project/checkpoints/2024-01-04/00-11-36/30500.pt')

    
    #save_dir = f'checkpoints/continued_00-11-36-30500-{log_name}'
    save_dir = f'checkpoints/{log_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    rl_agent.learn(save_dir, num_learning_iterations=50001, num_steps_per_val=500)


if __name__ == '__main__':
    train()
