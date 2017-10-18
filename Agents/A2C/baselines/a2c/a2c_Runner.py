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

import random

class Runner(object):

    def __init__(self, env, model,modelsavepath, nsteps=5, gamma=0.99):
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
        self.sumOfRewards = 0
        
        self.modelsavepath = modelsavepath
        
        
        # Load Model params if save exist
        
        self.model.load(self.modelsavepath)
        
        
        self.ActionStats = self.ResetActionStats()
        


        def on_Message(observation, reward, done, comID, step):
            #Update  self.observation
            observation = observation.reshape(self.obs.shape)      

            #print("OBS_SINGLE:" + str(observation))  
            self.update_obs(observation)
            #print("OBS_SINGLE__SELF:" + str(self.obs))
            #set reward and done to arrays
            self.sumOfRewards = self.sumOfRewards + reward
            reward = [reward]
            done = [done]
            self.dones = done

            #if first message for communication create new mbs class
            # if not first step append reward and done
            if step == 1:
                mbs = mb_class(comID, self.gamma, self.nsteps)
                self.mb_Classes.append(mbs)
            else:
                mbs = [m for m in self.mb_Classes if m.ID == comID][0]
                #for n, done in enumerate(done):
                #if done:
                    #self.obs = self.obs*0
                mbs.mb_rewards.append(reward)
            #Add Done to mb class => dones length +1 then other
            mbs.mb_dones.append(self.dones)

            #decide if to train model, call callback train function
            # if response to nth action ............   OR if done ???
            if (step != 1 and step%self.nsteps == 1) or done[0]:
                #transofrom arrays and call learn callback function
                lastValues = self.model.value(self.obs, self.states, self.dones).tolist()
                '''
                RESHAPE ACTION AND REWARD AND  VALUES!!!!!
                '''


                mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_advantages = mbs.transform(self.states, self.batch_ob_shape,lastValues)
                
                self.learn_cb(mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values,mb_advantages,self.sumOfRewards )
                #reset nbs class
                mbs.reset()
                #Add last done statement back to mbs class
                mbs.mb_dones.append(self.dones)

            actions, values, states = self.model.step(self.obs, self.states, self.dones)

            actions = self.exploration(actions)

            mbs.mb_obs.append(np.copy(self.obs))
            mbs.mb_actions.append(actions)
            mbs.mb_values.append(values)
            #store last state --------- shouldn't it be stored in mbs class since it depends on comID??????????
            mbs.mb_states.append(np.copy(self.states))
            self.states = states
            #mbs.mb_dones.append(self.dones)

            self.actionStats(actions)
            if done[0]:
                #remove mbs from list
                self.mb_Classes.remove(mbs)    
                self.states = model.initial_state

            #call action
            self.env.emit(comID,actions)

        self.env.SetCallbackFunction(on_Message)

        self.update = 1
        self.train_writer = tf.summary.FileWriter('/usr/src/app/log')
        def learning(obs, states, rewards, masks, actions, values,advantages,sumOfRewards):
            
            #print("OBS:" + str(obs))
            #print("REW:" +str(rewards))
            #print("ACT:"+str(actions))


            log_interval=100
            save_interval = 1000
            update = self.update
            policy_loss, value_loss, policy_entropy,summary = model.train(obs, states, rewards, masks, actions, values,advantages)
            nbatch = 1*nsteps
            tstart = time.time()
            nseconds = time.time()-tstart
            fps = int((update*nbatch)/nseconds)
            self.train_writer.add_summary(summary,update)

            if update % log_interval == 0:
                actStat = self.ActionStats
                logger.record_tabular("action_CTYP_AGENT",actStat['ChatType'][0])
                logger.record_tabular("action_CTYP_WORKER",actStat['ChatType'][1])
                logger.record_tabular("action_CTYP_KNOWLEDGE",actStat['ChatType'][2])
                #logger.record_tabular("action_CTYP_REVIEW",actStat['ChatType'][3])
                logger.record_tabular("action_CNR_0",actStat['ChatNr'][0])
                logger.record_tabular("action_CNR_1",actStat['ChatNr'][1])
                logger.record_tabular("action_MTYP_QUESTION",actStat['MessageType'][0])
                logger.record_tabular("action_MTYP_ANSWER",actStat['MessageType'][1])
                logger.record_tabular("action_MTYP_OBSERVATION",actStat['MessageType'][2])
                logger.record_tabular("nupdates", update)
                logger.record_tabular("avg_reward",sumOfRewards/(nbatch*log_interval))
                logger.record_tabular("sum_reward",sumOfRewards)
                logger.record_tabular("total_timesteps", update*nbatch)
                logger.record_tabular("fps", fps)
                logger.record_tabular("policy_entropy", float(policy_entropy))
                logger.record_tabular("value_loss", float(value_loss))
                logger.dump_tabular()
                self.sumOfRewards = 0
                self.ResetActionStats()

            if update % save_interval == 0:
                self.model.save(self.modelsavepath)
            
            self.update = update+ 1

        self.learn_cb = learning
    
    def listen(self):
        self.env.ListenToSocketIO()
        #print('ListenCom')

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        #print(str(obs))
        #self.obs = np.roll(self.obs, shift=-1, axis=3)
        #self.obs[:, :, :, -1] = obs[:, :, :, 0]
        self.obs = obs
        #self.obs[:,-1] = obs[:,0]

    def exploration(self,actions):
        epsilon = 0.8
        if random.random() > epsilon:
            ret =  actions
        else:
            ncty = self.env.ChatType.n
            ncnr = self.env.ChatNumber.n
            nmty = self.env.MessageType.n
            nmsg = self.env.MessageText.n
            nrew = self.env.FeedbackReward.n

            actspace =  self.env.ActSpace

            cty = [random.sample(range(ncty),1)[0] for _ in range(actspace[0])]
            cnr = [random.sample(range(ncnr),1)[0] for _ in range(actspace[1])]
            mty = [random.sample(range(nmty),1)[0] for _ in range(actspace[2])]
            msg = [random.sample(range(nmsg),1)[0] for _ in range(actspace[3])]
            rew = [random.sample(range(nrew),1)[0] for _ in range(actspace[4])]
            
            ret = np.reshape(np.concatenate((cty,cnr,mty,msg,rew)),(1,int(np.sum(actspace))))
        return ret

    def ResetActionStats(self):
        ncty = self.env.ChatType.n
        ncnr = self.env.ChatNumber.n
        nmty = self.env.MessageType.n
        nrew = self.env.FeedbackReward.n

        dic = {}
            
        dic['ChatType'] = np.zeros(ncty)
        dic['ChatNr'] = np.zeros(ncnr)
        dic['MessageType'] = np.zeros(nmty)
        dic['Reward'] = np.zeros(nrew)
        self.ActionStats = dic
        return dic

    def actionStats(self, actions):
        ctyi,cnri, mtyi, mtxt, reward = np.split(actions,np.cumsum(self.env.ActSpace), axis=1)[:-1]
        ctyi = ctyi[0][0]
        cnri = cnri[0][0]
        mtyi = mtyi[0][0]
        reward = reward[0][0]
        self.ActionStats['ChatType'][ctyi] += 1
        self.ActionStats['ChatNr'][cnri] += 1
        self.ActionStats['MessageType'][mtyi] += 1
        self.ActionStats['Reward'][reward] += 1




class mb_class():
    def __init__(self,id, gamma, nsteps):
        self.ID = id
        self.gamma = gamma
        self.nsteps  = nsteps
        self.gae_lambda = 0.96
        self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_states = [],[],[],[],[], []

    def reset(self):
        self.mb_obs, self.mb_rewards, self.mb_actions, self.mb_values, self.mb_dones, self.mb_states = [],[],[],[],[], []

    def transform(self, mb_states, batch_ob_shape, last_values):
        if len(self.mb_obs) < self.nsteps:
            short = True
        else:
            short = False
        mb_obs = self.mb_obs
        mb_rewards = self.mb_rewards
        mb_actions = self.mb_actions
        mb_values = self.mb_values
        mb_dones = self.mb_dones
        mb_states = self.mb_states

        mb_masks = []
        [mb_masks.append([1]) for i in range(len(mb_obs))]
        [mb_masks.append([0]) for i in range(self.nsteps-len(mb_masks))]
        #append missing nsteps
        [mb_obs.append(np.zeros(mb_obs[0].shape)) for i in range(self.nsteps-len(mb_obs))]
        [mb_rewards.append([0]) for i in range(self.nsteps-len(mb_rewards))]
        [mb_actions.append(np.zeros(mb_actions[0].shape)) for i in range(self.nsteps-len(mb_actions))]
        [mb_values.append(np.zeros(mb_values[0].shape)) for i in range(self.nsteps-len(mb_values))]
        [mb_dones.append([True]) for i in range(self.nsteps +1-len(mb_dones))]
        [mb_states.append(np.zeros(mb_states[0].shape)) for i in range(self.nsteps-len(mb_states))]


        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = np.asarray(mb_masks, dtype=np.float32).swapaxes(1, 0)
        #mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        #last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        #print('mbrewards: ' +str(mb_rewards))

        mb_advantages = self.GenrelaziedAdvantageEstimate(mb_rewards,mb_values,mb_dones,last_values)

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
        mb_rewards = np.squeeze(mb_rewards, axis=0)
        mb_advantages = np.squeeze(mb_advantages, axis=0)
        mb_actions = np.squeeze(mb_actions, axis=0)
        mb_values = np.squeeze(mb_values, axis=0)
        mb_masks = np.squeeze(mb_masks, axis=0)
        mb_states = np.squeeze(mb_states, axis=0)
        #if short:
            #print('Dones: ' + str(mb_dones))
            #print('Mask: ' + str(mb_masks))
        #print(mb_rewards)
        #print(mb_advantages)
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_advantages

    def GenrelaziedAdvantageEstimate(self,mb_rewards,mb_values,mb_dones,last_values ):
        gamma = self.gamma
        gae_lambda = self.gae_lambda
        mb_advantages = np.zeros(mb_rewards.shape)


        for n, (rewards,values,dones, lastvalue) in enumerate(zip(mb_rewards,mb_values, mb_dones, last_values)):
            rewards = rewards.tolist()
            values = values.tolist()
            dones = dones.tolist()
            values = values + [lastvalue]
            adv = np.asarray(rewards) + gamma*np.asarray(values[1:])  - np.asarray(values[:-1])
            advantages = discount_with_dones(adv.tolist(), dones, gamma * gae_lambda)

            mb_advantages[n] = advantages

        return mb_advantages



