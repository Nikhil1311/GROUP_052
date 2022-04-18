# import os
# import gym
# import torch
# import numpy as np
# from hparams import HyperParams as hp
# # from tensorboardX import SummaryWriter
# import torch.optim as optim
# from model import Actor, Critic
# from utils import get_action, save_checkpoint
# from collections import deque
# from running_state import ZFilter
# from vanila_pg import train_model

# class Agent():
#   '''The agent class that is to be filled.
#      You are allowed to add any method you
#      want to this class.
#   '''
#   # eps = 0.3

#   # def __init__(self, env_specs):
#   #   self.env_specs = env_specs
#   #   self.reward_list = [(env_specs['action_space'].sample(), 0)]

#   # def load_weights(self):
#   #   pass

#   # def act(self, curr_obs, mode='eval'):
#   #   if random.uniform(0, 1) > self.eps:
#   #     # return sorted(self.reward_list, key=lambda item: item[1], reverse=True)[0][0] try this a[np.argmax(np.array(a), axis=0)[1]][0]
#   #     return self.reward_list[np.argmax(self.reward_list, axis=0)[1]][0]
#   #   else:
#   #     return self.env_specs['action_space'].sample()

#   # def update(self, curr_obs, action, reward, next_obs, done, timestep):
#   #   self.reward_list.append((action, reward))

#   num_inputs = 11 #State
#   num_actions = 3 #Action

#   def __init__(self, env_specs):
#     self.env_specs = env_specs
#     self.num_inputs = 11 #State
#     self.num_actions = 3
#     self.actor = Actor(self.num_inputs, self.num_actions)
#     self.critic = Critic(self.num_inputs)
#     self.running_state = ZFilter((self.num_inputs,), clip=5)
#     self.actor_optim = optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
#     self.critic_optim = optim.Adam(self.critic.parameters(), lr=hp.critic_lr, weight_decay=hp.l2_rate)
#     self.memory = deque()
#     self.running_state = ZFilter((self.num_inputs,), clip=5)


#   def load_weights(self, root_path):
#     # Add root_path in front of the path of the saved network parameters
#     # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters
#     pass

#   def act(self, curr_obs, mode='eval'):
#     if mode=="train":
#       self.actor.eval(), self.critic.eval()
#       mu, std, _ = self.actor(torch.Tensor(curr_obs).unsqueeze(0))
#       action = get_action(mu, std)[0]
#       return action
#     else:
#       pass

#   def update(self, curr_obs, action, reward, next_obs, done, timestep):
#     curr_obs = self.running_state(curr_obs)
#     self.actor.train(), self.critic.train()
#     next_state = self.running_state(next_obs)
#     if done:
#         mask = 0
#     else:
#         mask = 1
#     self.memory.append([curr_obs, action, reward, mask])
#     if done:
#       train_model(self.actor, self.critic, self.memory, self.actor_optim, self.critic_optim)


import os
import gym
import torch
import numpy as np
from hparams import HyperParams as hp
import torch.optim as optim
from model import Actor, Critic
from utils import get_action, save_checkpoint
from collections import deque
from running_state import ZFilter
from vanila_pg import train_model

class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
  '''

  num_inputs = 11 #State
  num_actions = 3 #Action

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
    self.running_state = ZFilter((self.num_inputs,), clip=5)


  def load_weights(self, root_path):
    self.actor = torch.load(root_path+"actor.pt")
    self.critic = torch.load(root_path+"critic.pt")

  def act(self, curr_obs, mode='eval'):
    # if mode=="train":
    self.actor.eval(), self.critic.eval()
    mu, std, _ = self.actor(torch.Tensor(curr_obs).unsqueeze(0))
    # print(mu, std)
    # action = get_action(mu, std)[0]
    action = torch.normal(mu, std)
    # print(action)
    action = action.data.numpy()
    # print(action)
    # test = self.env_specs['action_space'].sample()
    # print(test)
    return action
    # else:
    #   pass

  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    curr_obs = self.running_state(curr_obs)
    next_state = self.running_state(next_obs)
    if done:
        mask = 0
    else:
        mask = 1
    self.memory.append([curr_obs, action, reward, mask])
    if done and timestep%2000==0:
      self.actor.train(), self.critic.train()
      train_model(self.actor, self.critic, self.memory, self.actor_optim, self.critic_optim)
      self.memory = deque()
    if timestep % 100000 == 0:
      torch.save(self.actor, "actor.pt")
      torch.save(self.critic, "critic.pt")
    # if timestep % 100000 == 0:
    #   torch.save(self.actor, "/content/assignment/actor_"+str(timestep)+".pt")
    #   torch.save(self.critic, "/content/assignment/critic_"+str(timestep)+".pt")