<<<<<<< HEAD
import numpy as np
import torch
import torch.nn.functional as F
from .sac.sac import *
from .sac import utils as utils
import zipfile
# import wandb

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'

class BaseAgent():
    '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
    '''

    def __init__(self, env_specs):
        self.env_specs = env_specs

    def load_weights(self, root_path):
        # Add root_path in front of the path of the saved network parameters
        # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters
        # checkpoint = torch.load(root_path + 'model_best.pt')

        pass

    def act(self, curr_obs, mode='eval'):
        return self.env_specs['action_space'].sample()

    def update(self, curr_obs, action, reward, next_obs, done, timestep):
        pass

    def reset(self):
        pass

class Agent(BaseAgent):
    """SAC algorithm."""

    def __init__(self, env_specs, obs_dim=11, action_dim=3, hidden_dim=512, hidden_depth=2, action_range=[-1, 1], device=device,
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
        self.obs_dim = obs_dim
        self.init_temperature = init_temperature
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.alpha_lr = alpha_lr
        self.alpha_betas = alpha_betas
        self.actor_lr = actor_lr
        self.actor_betas = actor_betas
        self.critic_lr = critic_lr
        self.critic_betas = critic_betas

        self.replay_buffer = utils.ReplayBuffer(self.env_specs['observation_space'].shape,
                                self.env_specs['action_space'].shape,
                                1000000,
                                self.device)

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
        self.timestep = 0

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def load_weights(self, root_path):

    # Add root_path in front of the path of the saved network parameters
    # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters

        with zipfile.ZipFile(root_path + 'model_final.zip', 'r') as zip_ref:
            zip_ref.extractall(root_path + 'model_final_unzipped')

        checkpoint = torch.load(root_path + 'model_final_unzipped/model_final.pt', map_location=device)
        self.critic = checkpoint['critic']
        self.critic_target = checkpoint['critic_target']
        self.actor = checkpoint['actor']
        self.log_alpha = checkpoint['log_alpha']

        if self.timestep == 0:
            self.timestep = 1000000


    # def act(self, obs, sample=False):
    #     obs = torch.FloatTensor(obs).to(self.device)
    #     obs = obs.unsqueeze(0)
    #     dist = self.actor(obs)
    #     action = dist.sample() if sample else dist.mean
    #     action = action.clamp(*self.action_range)
    #     assert action.ndim == 2 and action.shape[0] == 1
    #     return utils.to_np(action[0])

    def act(self, obs, mode='train'):
        if self.timestep < 2000:
            action = self.env_specs['action_space'].sample()
            return action
        else:
            with utils.eval_mode(self):
                obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
                dist = self.actor(obs)
                action = dist.sample() if mode == 'train' else dist.mean
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

    # def update(self, replay_buffer, use_wandb, step, step_per_inter=1):
    #     obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
    #         self.batch_size)
    #
    #     if use_wandb:
    #         wandb.log({'iter': step, 'train batch reward': reward.mean().detach().cpu().item()})
    #
    #     # logger.log('train/batch_reward', reward.mean(), step)
    #     for i in range(step_per_inter):
    #         self.update_critic(obs, action, reward, next_obs, not_done_no_max, use_wandb,
    #                            step)
    #
    #         if step % self.actor_update_frequency == 0:
    #             self.update_actor_and_alpha(obs, use_wandb, step)
    #
    #         if step % self.critic_target_update_frequency == 0:
    #             utils.soft_update_params(self.critic, self.critic_target,
    #                                      self.critic_tau)
    def reinit_model(self):
        print("PERFORM RESET")
        self.critic = DoubleQCritic(obs_dim=self.obs_dim,
                                    action_dim=self.action_dim,
                                    hidden_dim=self.hidden_dim,
                                    hidden_depth=self.hidden_depth).to(self.device)
        self.critic_target = DoubleQCritic(obs_dim=self.obs_dim,
                                           action_dim=self.action_dim,
                                           hidden_dim=self.hidden_dim,
                                           hidden_depth=self.hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(obs_dim=self.obs_dim,
                                       action_dim=self.action_dim,
                                       hidden_dim=self.hidden_dim,
                                       hidden_depth=self.hidden_depth,
                                       log_std_bounds=[-10, 10]).to(self.device)

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True

        # set target entropy to -|A|
        self.target_entropy = -self.action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.actor_lr,
                                                betas=self.actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 betas=self.critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.alpha_lr,
                                                    betas=self.alpha_betas)

        self.train()
        self.critic_target.train()


    def update(self, curr_obs, action, reward, next_obs, done, timestep):
        self.timestep = timestep

        if timestep < 2000:
            self.replay_buffer.add(curr_obs, action, reward, next_obs, done, done)
        else:
            self.replay_buffer.add(curr_obs, action, reward, next_obs, done, done)
            for i in range(1):
                obs, action, reward, next_obs, not_done, not_done_no_max = self.replay_buffer.sample(
                    self.batch_size)

                self.update_critic(obs, action, reward, next_obs, not_done_no_max, use_wandb=False,
                                   step=timestep)

            if timestep % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, use_wandb=False, step=timestep)

            if timestep % self.critic_target_update_frequency == 0:
                utils.soft_update_params(self.critic, self.critic_target,
                                         self.critic_tau)



=======
import os
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import torch.optim as optim
import math

def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action


def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) \
                  - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)


def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length


def kl_divergence(new_actor, old_actor, states):
    mu, std, logstd = new_actor(torch.Tensor(states))
    mu_old, std_old, logstd_old = old_actor(torch.Tensor(states))
    mu_old = mu_old.detach()
    std_old = std_old.detach()
    logstd_old = logstd_old.detach()

    # kl divergence between old policy and new policy : D( pi_old || pi_new )
    # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
    # be careful of calculating KL-divergence. It is not symmetric metric
    kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
         (2.0 * std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, num_outputs)
        # self.init_weight()
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def init_weight(self, dis=0.003):
        self.fc1.weight.data = self.fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = self.fanin_init(self.fc2.weight.data.size())
        # self.fc3.weight.data.uniform_(-dis, dis)

    def fanin_init(size, fanin=None):
        fanin = fanin or size[0]
        v = 1. / np.sqrt(fanin)
        return torch.Tensor(size).uniform_(-v, v)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        # x = nn.ReLU(self.fc1(x))
        # x = nn.ReLU(self.fc2(x))
        mu = self.fc3(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std, logstd


class Critic(nn.Module):
    def __init__(self, num_inputs):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        # self.init_weight()

    def init_weight(self, dis=0.003):
        self.fc1.weight.data = self.fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = self.fanin_init(self.fc2.weight.data.size())
        # self.fc3.weight.data.uniform_(-dis, dis)

    def fanin_init(size, fanin=None):
        fanin = fanin or size[0]
        v = 1. / np.sqrt(fanin)
        return torch.Tensor(size).uniform_(-v, v)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        # x = nn.ReLU(self.fc1(x))
        # x = nn.ReLU(self.fc2(x))
        v = self.fc3(x)
        return v

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n

    @property
    def mean(self):
        return self._M

    @mean.setter
    def mean(self, M):
        self._M = M

    @property
    def sum_square(self):
        return self._S

    @sum_square.setter
    def sum_square(self, S):
        self._S = S

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape

class hp:
    gamma = 0.99
    lamda = 0.98
    hidden = 64
    critic_lr = 0.0003
    actor_lr = 0.0003
    batch_size = 64
    l2_rate = 0.001
    max_kl = 0.01
    clip_param = 0.2

class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
  '''

  num_inputs = 11 #State
  num_actions = 3 #Action
  # hp = HyperParams

  def __init__(self, env_specs):
    # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.env_specs = env_specs
    self.num_inputs = 11 #State
    self.num_actions = 3
    self.actor = Actor(self.num_inputs, self.num_actions)
    self.critic = Critic(self.num_inputs)
    self.running_state = ZFilter((self.num_inputs,), clip=5)
    self.actor_optim = optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
    self.critic_optim = optim.Adam(self.critic.parameters(), lr=hp.critic_lr, weight_decay=hp.l2_rate)
    self.memory = deque()
    self.episode = 0

  def load_weights(self, root_path):
    self.actor = torch.load(root_path+"actor.pt")
    self.critic = torch.load(root_path+"critic.pt")

  def act(self, curr_obs, mode='eval'):
    self.actor.eval(), self.critic.eval()
    mu, std, _ = self.actor(torch.Tensor(self.running_state(curr_obs)).unsqueeze(0))
    action = get_action(mu, std)[0]
    return action

  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    curr_obs = self.running_state(curr_obs)
    if done:
        mask = 0
    else:
        mask = 1
    self.memory.append([curr_obs, action, reward, mask])
    if done:
      self.episode+=1
    if done and self.episode%200==0:
      print("Episode: ", self.episode)
    if done and (self.episode+1) % 100 == 0:
      self.actor.train(), self.critic.train()

      mem = np.array(self.memory)
      states = np.vstack(mem[:, 0])
      actions = list(mem[:, 1])
      rewards = list(mem[:, 2])
      masks = list(mem[:, 3])

      returns = self.get_returns(rewards, masks)
      self.train_actor(returns, states, actions)
      self.train_critic(states, returns)
      
      self.memory = deque()
    if timestep % 10000 == 0:
      torch.save(self.actor, "actor.pt")
      torch.save(self.critic, "critic.pt")

  def get_returns(self, rewards, masks):
      rewards = torch.Tensor(rewards)
      masks = torch.Tensor(masks)
      returns = torch.zeros_like(rewards)

      running_returns = 0

      for t in reversed(range(0, len(rewards))):
          running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
          returns[t] = running_returns

      returns = (returns - returns.mean()) / returns.std()
      return returns


  def get_loss(self, returns, states, actions):
      mu, std, logstd = self.actor(torch.Tensor(states))
      log_policy = log_density(torch.Tensor(actions), mu, std, logstd)
      returns = returns.unsqueeze(1)

      objective = returns * log_policy
      objective = objective.mean()
      return - objective


  def train_critic(self, states, returns):
      criterion = torch.nn.MSELoss()
      n = len(states)
      arr = np.arange(n)

      for epoch in range(5):
          np.random.shuffle(arr)

          for i in range(n // hp.batch_size):
              batch_index = arr[hp.batch_size * i: hp.batch_size * (i + 1)]
              batch_index = torch.LongTensor(batch_index)
              inputs = torch.Tensor(states)[batch_index]
              target = returns.unsqueeze(1)[batch_index]

              values = self.critic(inputs)
              loss = criterion(values, target)
              self.critic_optim.zero_grad()
              # print(loss)
              loss.backward()
              # print(loss)
              self.critic_optim.step()


  def train_actor(self, returns, states, actions):
      loss = self.get_loss(returns, states, actions)
      self.actor_optim.zero_grad()
      # print(loss)
      loss.backward()
      # print(loss)
      self.actor_optim.step()
>>>>>>> 60ebeb2c183a5bf7a7a60d0f9e9173cd71f39c30
