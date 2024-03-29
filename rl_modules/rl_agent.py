import os.path
from env.go_env import GOEnv
from rl_modules.storage import Storage
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from rl_modules.actor_critic import ActorCritic

class RLAgent(nn.Module):
    def __init__(self,
                 env: GOEnv,
                 storage: Storage,
                 actor_critic: ActorCritic,
                 lr=1e-3,
                 value_loss_coef=0.5,
                 num_batches=1,
                 num_epochs=1,
                 device='cpu',
                 action_scale=0.3,
                 eps_clip=0.2,
                 entropy_coef = 0.01,
                 train=True,
                 exp_name=None
                 ):
        super().__init__()
        self.env = env
        self.storage = storage
        self.actor_critic = actor_critic
        self.num_batches = num_batches
        self.num_epochs = num_epochs
        self.value_loss_coef = value_loss_coef
        self.device = device
        self.action_scale = action_scale
        self.transition = Storage.Transition()
        self.eps_clip = eps_clip
        # create the normalizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        if train:
            self.writer = SummaryWriter("runs/"+exp_name)
        self.total_reward_counter = 1
        self.mean_total_reward = 0
        self.train = train
        self.entropy_coef = entropy_coef

    def act(self, obs):
        # Compute the actions and values
        action = self.actor_critic.act(obs).squeeze()
        self.transition.action = action.detach().cpu().numpy()
        self.transition.value = self.actor_critic.evaluate(obs).squeeze().detach().cpu().numpy()
        self.transition.action_log_prob = self.actor_critic.get_actions_log_prob(action).detach().cpu().numpy()
        return self.transition.action

    def inference(self, obs):
        return self.actor_critic.act_inference(obs).squeeze().detach().cpu().numpy()

    def store_data(self, obs, reward, done):
        self.transition.obs = obs
        self.transition.reward = reward
        self.transition.done = done

        # Record the transition
        self.storage.store_transition(self.transition)
        self.transition.clear()

    def compute_returns(self, last_obs):
        last_values = self.actor_critic.evaluate(last_obs).detach().cpu().numpy()
        return self.storage.compute_returns(last_values)

    def update(self):
        mean_value_loss = 0
        mean_actor_loss = 0
        mean_loss = 0
        generator = self.storage.mini_batch_generator(self.num_batches, self.num_epochs, device=self.device)

        for obs_batch, actions_batch, target_values_batch, advantages_batch in generator:
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

            actor_loss = (-advantages_batch * actions_log_prob_batch - self.entropy_coef*self.actor_critic.entropy()).mean()
            critic_loss = advantages_batch.pow(2).mean()
            loss = actor_loss + self.value_loss_coef * critic_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_value_loss += critic_loss.item()
            mean_actor_loss += actor_loss.item()
            mean_loss += loss.item()

        num_updates = self.num_epochs * self.num_batches
        mean_value_loss /= num_updates
        mean_actor_loss /= num_updates
        mean_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_actor_loss

    def play(self, is_training=True, early_termination=True):
        obs, _ = self.env.reset()
        infos = []
        total_reward = 0
        for _ in range(self.storage.max_timesteps):
            obs_tensor = torch.from_numpy(obs).to(self.device).float().unsqueeze(dim=0)
            with torch.no_grad():
                if is_training:
                    action = self.act(obs_tensor)
                else:
                    action = self.inference(obs_tensor)
            obs_next, reward, terminate, info = self.env.step(action*self.action_scale)
            infos.append(info)
            total_reward += reward
            if is_training:
                self.store_data(obs, reward, terminate)
            if terminate and early_termination:
                obs, _ = self.env.reset()
                #print("Total reward: ", total_reward)
                self.total_reward_counter += 1
                self.mean_total_reward += total_reward
                total_reward = 0
            else:
                obs = obs_next
        if is_training:
            self.compute_returns(torch.from_numpy(obs_next).to(self.device).float())

        return infos

    def learn(self, save_dir, num_learning_iterations=1000, num_steps_per_val=50):
        for it in range(num_learning_iterations):
            # play games to collect data
            self.total_reward_counter = 1
            infos = self.play(is_training=True)
            # improve policy with collected data
            mean_value_loss, mean_actor_loss = self.update()
            print(f'Iteration {it}: mean value loss = {mean_value_loss}, mean actor loss = {mean_actor_loss}')
            print("Mean total reward: ", self.mean_total_reward/self.total_reward_counter)
            self.writer.add_scalar('Loss/mean_value_loss', mean_value_loss, it)
            self.writer.add_scalar('Loss/mean_actor_loss', mean_actor_loss, it)
            self.writer.add_scalar('Reward/mean_total_reward', self.mean_total_reward/self.total_reward_counter, it)
            self.mean_total_reward = 0
            if it % num_steps_per_val == 0:
                infos = self.play(is_training=False)
                self.save_model(os.path.join(save_dir, f'{it}.pt'))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


class PPO(RLAgent):
    def __init__(self,
                    env: GOEnv,
                    storage: Storage,
                    actor_critic: ActorCritic,
                    lr=1e-3,
                    value_loss_coef=0.5,
                    num_batches=1,
                    num_epochs=1,
                    device='cpu',
                    action_scale=0.3,
                    entropy_coef = 0.01,
                    train=True,
                    exp_name=""
                    ):
            super().__init__(env, storage, actor_critic, lr, value_loss_coef, num_batches, num_epochs, device, action_scale, entropy_coef=entropy_coef, train=train, exp_name=exp_name)
            self.env = env
            self.storage = storage
            self.actor_critic = actor_critic
            self.num_batches = num_batches
            self.num_epochs = num_epochs
            self.value_loss_coef = value_loss_coef
            self.device = device
            self.action_scale = action_scale
            self.transition = Storage.Transition()
            self.MseLoss = nn.MSELoss()
            # create the normalizer
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

    def update(self):
        mean_actor_loss = 0
        mean_value_loss = 0
        mean_loss = 0
        generator = self.storage.ppo_mini_batch_generator(self.num_batches, self.num_epochs, device=self.device)

        for old_obs_batch, old_actions_batch, old_target_values_batch, old_advantages_batch, old_log_probs_batch, old_rewards_batch in generator:
            self.actor_critic.act(old_obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(old_actions_batch)

            ratios = torch.exp(actions_log_prob_batch - old_log_probs_batch.detach())
            surr1 = ratios * old_advantages_batch
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * old_advantages_batch

            # final loss of clipped objective PPO
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (self.MseLoss(old_target_values_batch, old_rewards_batch)).mean()
            #actor_loss = (-old_advantages_batch * actions_log_prob_batch).mean()
            #critic_loss = old_advantages_batch.pow(2).mean()
            loss = actor_loss + self.value_loss_coef * critic_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_value_loss += critic_loss.item()
            mean_actor_loss += actor_loss.item()
            mean_loss += loss.item()
        num_updates = self.num_epochs * self.num_batches
        mean_value_loss /= num_updates
        mean_actor_loss /= num_updates
        mean_loss /= num_updates

        self.storage.clear()

        return mean_value_loss, mean_actor_loss
    