import os
import sys
import pathlib
import numpy as np
import copy
from tqdm import tqdm
from pymdp.agent import Agent
from pymdp.utils import plot_beliefs, plot_likelihood
from pymdp import utils
from pymdp.envs import TMazeEnv

reward_probabilities = [1, 0] # probabilities used in the original SPM T-maze demo
env = TMazeEnv(reward_probs = reward_probabilities)
# generative process
A_gp = env.get_likelihood_dist()
B_gp = env.get_transition_dist()
# generative model
A_gm = copy.deepcopy(A_gp) # make a copy of the true observation likelihood to initialize the observation model
B_gm = copy.deepcopy(B_gp) # make a copy of the true transition likelihood to initialize the transition model
# define agent
controllable_indices = [0] # this is a list of the indices of the hidden state factors that are controllable

A_shapes = [A_gp[i].shape for i in range(len(A_gp))]
A = utils.obj_array_uniform(A_shapes)
pA = utils.dirichlet_like(A, scale = 1.0)
agent = Agent(A=A, B=B_gm, pA=pA, control_fac_idx=controllable_indices, save_belief_hist=True, use_states_info_gain=False, use_param_info_gain=True, lr_pA=100, policy_len=2)
agent.D[0] = utils.onehot(0, agent.num_states[0])
agent.C[1][1] = 3.0
agent.C[1][2] = -3.0

# Run active inference
T = 2# number of timesteps
loc_obs, reward_obs, cue_obs = env.reset()
history_of_locs = [loc_obs]
obs = [loc_obs, reward_obs, cue_obs]
# these are useful for displaying read-outs during the loop over time
# reward_conditions = ["Right", "Left"]
# location_observations = ['CENTER','RIGHT ARM','LEFT ARM','CUE LOCATION']
# reward_observations = ['No reward','Reward!','Loss!']
# cue_observations = ['Cue Right','Cue Left']
# msg = """ === Starting experiment === \n Reward condition: {}, Observation: [{}, {}, {}]"""
# print(msg.format(reward_conditions[env.reward_condition], location_observations[obs[0]], reward_observations[obs[1]], cue_observations[obs[2]]))

trials = 1000# number of trials
for n in tqdm(range(trials)):
    for t in range(T):
        qx_prev = agent.qs.copy()
        qx = agent.infer_states(obs)
        q_pi, efe = agent.infer_policies()
        action = agent.sample_action()
    #    msg = """[Step {}] Action: [Move to {}]"""
     #   print(msg.format(t, location_observations[int(action[0])]))
        loc_obs, reward_obs, cue_obs = env.step(action)
        obs = [loc_obs, reward_obs, cue_obs]    
        history_of_locs.append(loc_obs)
    #    msg = """[Step {}] Observation: [{},  {}, {}]"""
    #    print(msg.format(t, location_observations[obs[0]], reward_observations[obs[1]], cue_observations[obs[2]]))
        qa = agent.update_A(obs)
    agent.reset()

plot_likelihood(agent.A[1][:,:,0],'Reward Right')
# plot_likelihood(A_gp[1][:,:,0],'Reward Right')