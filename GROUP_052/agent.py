import numpy as np
import torch
import torch.nn.functional as F
from GROUP_052.sac.sac import *
import wandb

class Agent():
    '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
    '''

    def __init__(self, env_specs):
        self.env_specs = env_specs

    def load_weights(self, root_path):
        # Add root_path in front of the path of the saved network parameters
        # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters
        pass

    def act(self, curr_obs, mode='eval'):
        return self.env_specs['action_space'].sample()

    def update(self, curr_obs, action, reward, next_obs, done, timestep):
        pass

    def reset(self):
        pass


class SACAgent(Agent):
    """SAC algorithm."""

    def __init__(self, env_specs, obs_dim=11, action_dim=3, hidden_dim=256, hidden_depth=2, action_range=[-1, 1], device='cuda',
                 discount=0.99, init_temperature=0.2, alpha_lr=3e-4, alpha_betas=[0.9, 0.999],
                 actor_lr=1e-3, actor_betas=[0.9, 0.999], actor_update_frequency=1, critic_lr=1e-3,
                 critic_betas=[0.9, 0.999], critic_tau=0.005, critic_target_update_frequency=2,
                 batch_size=256, learnable_temperature=True):
        super().__init__(env_specs)

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth

        self.critic = DoubleQCritic(obs_dim=obs_dim,
                                    action_dim=action_dim,
                                    hidden_dim=hidden_dim,
                                    hidden_depth=hidden_depth).to(device)
        self.critic_target = DoubleQCritic(obs_dim=obs_dim,
                                           action_dim=action_dim,
                                           hidden_dim=hidden_dim,
                                           hidden_depth=hidden_depth).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(obs_dim=obs_dim,
                                       action_dim=action_dim,
                                       hidden_dim=hidden_dim,
                                       hidden_depth=hidden_depth,
                                       log_std_bounds=[-10, 10]).to(device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True

        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, use_wandb, step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        if use_wandb:
            wandb.log({'iter':step, 'critic train loss': critic_loss.detach().cpu().item()})

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, use_wandb, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        if use_wandb:
            wandb.log({'iter':step, 'actor train loss': actor_loss.detach().cpu().item(), 'actor entropy':-log_prob.mean().detach().cpu().item()})

        # logger.log('train_actor/loss', actor_loss, step)
        # logger.log('train_actor/target_entropy', self.target_entropy, step)
        # logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            if use_wandb:
                wandb.log({'iter': step, 'alpha train loss': alpha_loss.detach().cpu().item(),
                           'alpha value': self.alpha})
            #
            # logger.log('train_alpha/loss', alpha_loss, step)
            # logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, use_wandb, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size)
        if use_wandb:
            wandb.log({'iter': step, 'train batch reward': reward.mean().detach().cpu().item()})

        # logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max, use_wandb,
                           step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, use_wandb, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)