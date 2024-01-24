from rl_modules.rl_agent import RLAgent, PPO
from env.go_env import GOEnv
from rl_modules.storage import Storage
from rl_modules.actor_critic import ActorCritic
from datetime import datetime
import os
import torch
import yaml
import argparse

def train():
    parser = argparse.ArgumentParser(
        description='sum the integers at the command line')
    parser.add_argument(
        '--config', default='configs/run1.yaml', type=str,
        help='config.yaml file that contains the hyperparameters for the experiment')
    args = parser.parse_args()
    #load hyperparameters from config file
    params = yaml.safe_load(open(args.config))
    if torch.cuda.is_available():
        device ='cuda'
    else:
        device = 'cpu'

    exp_name = os.path.basename(args.config).split('.')[0]

    log_name = exp_name+"-"+datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    # create environment
    go_env = GOEnv(**params["env"])
    # create actor critic
    actor_critic = ActorCritic(state_dim=go_env.obs_dim, action_dim=go_env.action_dim).to(device)
    # create storage to save data
    storage = Storage(obs_dim=go_env.obs_dim, action_dim=go_env.action_dim, **params["storage"])
    rl_agent = PPO(env=go_env, actor_critic=actor_critic, storage=storage, device=device, **params["agent"], train=True, exp_name=exp_name)
    
    save_dir = f'checkpoints/{log_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        yaml.dump(params, open(os.path.join(save_dir, "config.yaml"), "w"))
    rl_agent.learn(save_dir, **params["learn"])


if __name__ == '__main__':
    train()
