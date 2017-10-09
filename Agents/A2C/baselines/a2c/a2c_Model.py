'''
Based on OPEN AI A2C implementation in the "Baselines" package
https://github.com/openai/baselines
'''


import os.path as osp
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


class Model(object):

    def __init__(self, policy, env, nsteps, num_procs,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
        
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        
        nbatch = nsteps
        ac_space =  env.ActSpace

        #A = tf.placeholder(tf.int32, [nbatch])
        A = tf.placeholder(tf.int32, [nbatch,sum(ac_space)])
        A_cty, A_cnr, A_mty, A_msg, A_rew = tf.split(A,num_or_size_splits = ac_space, axis = 1)

        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, env, 1, reuse=False)
        train_model = policy(sess, env, nsteps, reuse=True)

        '''
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        '''

        #loss chat type
        cty_coef = 1/5
        neglogpac_cty = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.cty, labels=A_cty)
        pg_loss_cty = tf.reduce_mean(ADV * neglogpac_cty)
        entropy_cty = tf.reduce_mean(cat_entropy(train_model.cty))
        loss_cty = (pg_loss_cty - entropy_cty*ent_coef) * cty_coef

        #loss chat nr
        cnr_coef = 1/5
        neglogpac_cnr = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.cnr, labels=A_cnr)
        pg_loss_cnr = tf.reduce_mean(ADV * neglogpac_cnr)
        entropy_cnr = tf.reduce_mean(cat_entropy(train_model.cnr))
        loss_cnr = (pg_loss_cnr - entropy_cnr*ent_coef) * cnr_coef

        #loss message type
        mty_coef = 1/5
        neglogpac_mty = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.mty, labels=A_mty)
        pg_loss_mty = tf.reduce_mean(ADV * neglogpac_mty)
        entropy_mty = tf.reduce_mean(cat_entropy(train_model.mty))
        loss_mty = (pg_loss_mty - entropy_mty*ent_coef) * mty_coef



        ''' NEED TO BE ADJUSTED '''     
        #loss message
        mes_coef = 1/5
        # needs to be adjusted
        neglogpac_mes_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.mes, labels=A_msg)
        neglogpac_mes = tf.reduce_sum(neglogpac_mes_1 *  train_model.mask, [1,2]) /tf.reduce_sum(train_model.mask,[1,2])
        pg_loss_mes = tf.reduce_mean(ADV * neglogpac_mes)
        # needs to be adjusted
        entropy_mes_1 = tf.reduce_mean(cat_entropy(train_model.mes, dim = 2) * train_model.mask,[1,2])
        entropy_mes = tf.reduce_mean(cat_entropy(train_model.mes))
        loss_mes = (pg_loss_mes - entropy_mes*ent_coef) * mes_coef

        #loss feedback reward
        rew_coef = 1/5
        neglogpac_rew = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.rew, labels=A_rew)
        pg_loss_rew = tf.reduce_mean(ADV * neglogpac_rew)
        entropy_rew = tf.reduce_mean(cat_entropy(train_model.rew))
        loss_rew = (pg_loss_rew - entropy_rew*ent_coef) * rew_coef

        pg_loss = loss_cty + loss_cnr + loss_mty + loss_mes + loss_rew
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        
        #loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        loss = pg_loss + vf_loss * vf_coef
        entropy = entropy_cty + entropy_cnr + entropy_mty + entropy_mes + entropy_rew

        
        tf.summary.scalar('loss', loss)
        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        merged = tf.summary.merge_all()

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ ,summary= sess.run(
                [pg_loss, vf_loss, entropy, _train,merged],
                td_map
            )
            return policy_loss, value_loss, policy_entropy, summary

        def save(save_path):
            ps = sess.run(params)
            make_path(save_path)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load

        tf.global_variables_initializer().run(session=sess)
