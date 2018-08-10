#ARS

# Importing the libraries
import os
import numpy as np

# Setting the Hyper Parameters

class Hp():
  
  def __init__(self):
    self.nb_steps = 1000
    self.episode_length = 1000
    self.learning_rate = 0.02
    self.nb_directions = 16
    self.nb_best_directions = 16
    assert self.nb_best_directions <= self.nb_directions
    self.noise = 0.03
    self.seed = 1
    self.env_name = ''

# Normalizing the states
    
class Normalizer():
  def __init__(self, nb_inputs):
    self.n = np.zeros(nb_inputs)
    self.mean = np.zeros(nb_inputs)
    self.mean_diff = np.zeros(nb_inputs)
    self.var = np.zeros(nb_inputs)
    
    
  def observe(self, x):
    self.n += 1.0
    last_mean = self.mean.copy()
    self.mean += (x - self.mean) / self.n
    self.mean_diff += (x - last_mean) * (x - self.mean)
    self.var = (self.mean_diff / self.n).clip(min = le-2) #makes sure var via clip is not equal to zero
  
  def normalize(self, inputs):
    obs_mean = self.mean
    obs_std = np.sqrt(self.var)
    return (inputs - obs_mean) / obs_std
  
# Building the AI 
    
class Policy():
  
  def __init__(self, input_size, output_size):
    self.theta = np.zeros((output_size, input_size))
    
  def evaluate(self, input, delta = None, direction = None):
    if direction is None: 
      return self.theta.dot(input)
    elif direction == "positive":
      return (self.theta + hp.noise*delta).dot(input)
    else:
      return (self.theta - hp.noise*delta).dot(input)
    
   def sample_deltas(self):
       return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]
   
    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta, sigma.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_direction * sigma_r) * step
        
# Exploring the policy on one specific direction and over one direction

def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    num_plays = 0.0
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards

# Training the AI
    
def train(env, policy, normalizer, hp):
    
    for step in range(hp.nb_steps):
        
        

    
    

    
    
    
    
    
    
    