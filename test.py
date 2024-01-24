from rl_modules.rl_agent import RLAgent, PPO
from env.go_env import GOEnv
from rl_modules.storage import Storage
from rl_modules.actor_critic import ActorCritic
import torch
import yaml
import os


def test():
    path = '/home/markus/Documents/Uni/RobotLearning/Robot-Learning-RL-Task/checkpoints/base-2024-01-24/17-36-07/6750.pt'
    config_path = os.path.dirname(path)+"/config.yaml"
    print(config_path)
    
    if torch.cuda.is_available():
        device ='cuda'
    else:
        device = 'cpu'
    #load hyperparameters from config file
    
    if os.path.exists(config_path):
        config = yaml.safe_load(open(config_path))
        config["env"]["render_mode"] = "human"
        go_env = GOEnv(**config["env"])
    else:
        go_env = GOEnv(render_mode="human", scene='go/scene.xml', torque=False)
    # create actor critic
    actor_critic = ActorCritic(state_dim=go_env.obs_dim, action_dim=go_env.action_dim).to(device)
    # create storage to save data
    storage = Storage(obs_dim=go_env.obs_dim, action_dim=go_env.action_dim, max_timesteps=1000)
    rl_agent = PPO(env=go_env, actor_critic=actor_critic, storage=storage, device=device, train=False)

    rl_agent.load_model(path)
    rl_agent.play(is_training=False)


if __name__ == '__main__':
    test()
