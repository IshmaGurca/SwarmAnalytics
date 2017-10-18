'''
Based on OPEN AI A2C implementation in the "Baselines" package
https://github.com/openai/baselines
'''

import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger


from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse

from baselines.a2c.a2c_Model import Model
from baselines.a2c.a2c_Runner import Runner



def learn(policy, env, modelsavepath, nsteps=5, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):    
    tf.reset_default_graph()
    update = 1
    #ob_space = env.ObsSpace
    #ac_space =  env.ActSpace
    num_procs = 1 #len(env.remotes) # HACK

    model = Model(policy=policy, env=env, nsteps=nsteps, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, lrschedule=lrschedule)

    runner = Runner(env, model,modelsavepath = modelsavepath,  nsteps=nsteps, gamma=gamma)
    time.sleep(1)
    runner.listen()