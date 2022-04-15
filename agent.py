import numpy as np

class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
  '''
  eps = 0.3

  def __init__(self, env_specs):
    self.env_specs = env_specs
    self.reward_list = [(env_specs['action_space'].sample(), 0)]

  def load_weights(self):
    pass

  def act(self, curr_obs, mode='eval'):
    if random.uniform(0, 1) > self.eps:
      # return sorted(self.reward_list, key=lambda item: item[1], reverse=True)[0][0] try this a[np.argmax(np.array(a), axis=0)[1]][0]
      return self.reward_list[np.argmax(self.reward_list, axis=0)[1]][0]
    else:
      return self.env_specs['action_space'].sample()

  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    self.reward_list.append((action, reward))
