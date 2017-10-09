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
from baselines.a2c.policies import CnnPolicy
from baselines.a2c.utils import cat_entropy, mse


try:
    from SwarmAnalyticsUtility.MessageInterface import MESSAGETYP, MessageInterface
    from SwarmAnalyticsUtility.CommunicationEnviroment import CommunicationEnviroment
except ImportError:
    import pip
    pip.main(['install','-e','/shared/MessageUtilities'])
    #print('Pip install SwarmAnalyticsUtility')
    from SwarmAnalyticsUtility.MessageInterface import MESSAGETYP, MessageInterface
    from SwarmAnalyticsUtility.CommunicationEnviroment import CommunicationEnviroment

class Runner(object):

    def __init__(self, env, model, nsteps=5, gamma=0.99):
        self.env = env
        self.model = model
        obs_space = self.env.ObsSpace
        nh = sum(obs_space)
        self.batch_ob_shape = (nsteps, nh)
        self.obs = np.zeros((1, nh), dtype=np.uint8)
        #obs = env.reset()
        #self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(1)]
        self.mb_Classes = []

        def on_Message(observation, reward, done, comID, step):
            #Update  self.observation
            observation = observation.reshape(self.obs.shape)        
            self.update_obs(observation)
            #set reward and done to arrays
            reward = [reward]
            done = [done]
            self.dones = done

            #if first message for communication create new mbs class
            # if not first step append reward and done
            if step == 1:
                mbs = mb_class(comID, self.gamma)
                self.mb_Classes.append(mbs)
            else:
                mbs = [m for m in self.mb_Classes if m.ID == comID][0]
                #for n, done in enumerate(done):
                if done:
                    self.obs = self.obs*0
                mbs.mb_rewards.append(reward)
            #Add Done to mb class => dones length +1 then other
            mbs.mb_dones.append(self.dones)

            #decide if to train model, call callback train function
            # if response to nth action ............   OR if done ???
            if step != 1 and step%self.nsteps == 1:
                #transofrom arrays and call learn callback function
                lastValues = self.model.value(self.obs, self.states, self.dones).tolist()
                mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values = mbs.transform(self.states, self.batch_ob_shape,lastValues)

                self.learn_cb(mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values)
                #reset nbs class
                mbs.reset()
                #Add last done statement back to mbs class
                mbs.mb_dones.append(self.dones)

            actions, values, states = self.model.step(self.obs, self.states, self.dones)
            mbs.mb_obs.append(np.copy(self.obs))
            mbs.mb_actions.append(actions)
            mbs.mb_values.append(values)
            #store last state --------- shouldn't it be stored in mbs class since it depends on comID??????????
            self.states = states
            #mbs.mb_dones.append(self.dones)

            #call action
            self.env.emit(comID,actions)

        self.env.SetCallbackFunction(on_Message)

        self.update = 1
        self.train_writer = tf.summary.FileWriter('/usr/src/app/log')
        def learning(obs, states, rewards, masks, actions, values):
            log_interval=100
            update = self.update
            policy_loss, value_loss, policy_entropy,summary = model.train(obs, states, rewards, masks, actions, values)
            nbatch = 1*nsteps
            tstart = time.time()
            nseconds = time.time()-tstart
            fps = int((update*nbatch)/nseconds)
            self.train_writer.add_summary(summary,update)

            if update % log_interval == 0 or update == 1:
                logger.record_tabular("nupdates", update)
                logger.record_tabular("total_timesteps", update*nbatch)
                logger.record_tabular("fps", fps)
                logger.record_tabular("policy_entropy", float(policy_entropy))
                logger.record_tabular("value_loss", float(value_loss))
                logger.dump_tabular()
            self.update = update+ 1


        self.learn_cb = learning
    
    def listen(self):
        self.env.ListenToSocketIO()
        #print('ListenCom')

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        #print(str(obs))
        self.obs = np.roll(self.obs, shift=-1, axis=3)
        self.obs[:, :, :, -1] = obs[:, :, :, 0]
        #self.obs[:,-1] = obs[:,0]

class mb_class():
    def __init__(self,id, gamma):
        self.ID = id
        self.gamma = gamma
        self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones = [],[],[],[],[]

    def reset(self):
        self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones = [],[],[],[],[]

    def transform(self, mb_states, batch_ob_shape, last_values):
        mb_obs = self.mb_obs
        mb_rewards = self.mb_rewards
        mb_actions = self.mb_actions
        mb_values = self.mb_values
        mb_dones = self.mb_dones
        

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        #last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        #print('mbrewards: ' +str(mb_rewards))
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            #print('before disc: ' +str(rewards))
            #print('dones: ' +str(dones))
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1] #Don't get it why not use last?
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            #print('after disc: ' +str(rewards))
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values



