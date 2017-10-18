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
from baselines.a2c.utils import cat_entropy, mse



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
        M =  tf.placeholder(tf.float32, [nbatch])

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
        neglogpac_cty = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.cty, labels=tf.squeeze(A_cty,[1]))
        #pg_loss_cty = tf.reduce_mean(ADV * neglogpac_cty)
        pg_loss_cty = tf.reduce_sum(-1*ADV * neglogpac_cty*M)/tf.reduce_sum(M)
        #entropy_cty = tf.reduce_mean(cat_entropy(train_model.cty))
        entropy_cty= tf.reduce_sum(cat_entropy(train_model.cty)*M)/tf.reduce_sum(M)
        loss_cty = (pg_loss_cty - entropy_cty*ent_coef) * cty_coef

        #loss chat nr
        cnr_coef = 1/5
        neglogpac_cnr = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.cnr, labels=tf.squeeze(A_cnr,[1]))
        #pg_loss_cnr = tf.reduce_mean(ADV * neglogpac_cnr)
        pg_loss_cnr = tf.reduce_sum(-1*ADV * neglogpac_cnr*M)/tf.reduce_sum(M)
        #entropy_cnr = tf.reduce_mean(cat_entropy(train_model.cnr))
        entropy_cnr= tf.reduce_sum(cat_entropy(train_model.cnr)*M)/tf.reduce_sum(M)
        loss_cnr = (pg_loss_cnr - entropy_cnr*ent_coef) * cnr_coef

        #loss message type
        mty_coef = 1/5
        neglogpac_mty = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.mty, labels=tf.squeeze(A_mty,[1]))
        #pg_loss_mty = tf.reduce_mean(ADV * neglogpac_mty)
        pg_loss_mty = tf.reduce_sum(-1*ADV * neglogpac_mty*M)/tf.reduce_sum(M)
        #entropy_mty = tf.reduce_mean(cat_entropy(train_model.mty))
        entropy_mty= tf.reduce_sum(cat_entropy(train_model.mty)*M)/tf.reduce_sum(M)
        loss_mty = (pg_loss_mty - entropy_mty*ent_coef) * mty_coef



        ''' NEED TO BE ADJUSTED '''     
        #loss message
        mes_coef = 1/5
        # needs to be adjusted
        neglogpac_mes_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.mes, labels=A_msg)
        mask = tf.to_float(train_model.mask)
        neglogpac_mes = tf.reduce_sum(neglogpac_mes_1 *  mask, [1]) /tf.reduce_sum(mask,[1])
        #pg_loss_mes = tf.reduce_mean(ADV * neglogpac_mes)
        pg_loss_mes = tf.reduce_sum(-1*ADV * neglogpac_mes*M)/tf.reduce_sum(M)
        # needs to be adjusted
        cat_ent_mes = cat_entropy(train_model.mes, dim = 2) 
        entropy_mes_1 = tf.reduce_sum(cat_ent_mes * mask,[1]) /tf.reduce_sum(mask,[1])
        #entropy_mes = tf.reduce_mean(cat_entropy(train_model.mes))
        entropy_mes = tf.reduce_sum(entropy_mes_1*M)/tf.reduce_sum(M)
        loss_mes = (pg_loss_mes - entropy_mes*ent_coef) * mes_coef

        #loss feedback reward
        rew_coef = 1/5
        neglogpac_rew = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.rew, labels=tf.squeeze(A_rew,[1]))
        #pg_loss_rew = tf.reduce_mean(ADV * neglogpac_rew)
        pg_loss_rew = tf.reduce_sum(-1*ADV * neglogpac_rew*M)/tf.reduce_sum(M)
        #entropy_rew = tf.reduce_mean(cat_entropy(train_model.rew))
        entropy_rew= tf.reduce_sum(cat_entropy(train_model.rew)*M)/tf.reduce_sum(M)
        loss_rew = (pg_loss_rew - entropy_rew*ent_coef) * rew_coef

        pg_loss = loss_cty + loss_cnr + loss_mty + loss_mes + loss_rew
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        
        #loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        loss = pg_loss + vf_loss * vf_coef
        entropy = entropy_cty + entropy_cnr + entropy_mty + entropy_mes + entropy_rew

        tf.summary.scalar('neglogpac_cty',tf.reduce_mean(neglogpac_cty))
        tf.summary.scalar('neglogpac_cnr',tf.reduce_mean(neglogpac_cnr))
        tf.summary.scalar('neglogpac_mty',tf.reduce_mean(neglogpac_mty))
        tf.summary.scalar('neglogpac_mes',tf.reduce_mean(neglogpac_mes))
        tf.summary.scalar('neglogpac_rew',tf.reduce_mean(neglogpac_rew))
        tf.summary.scalar('loss_cty',loss_cty)
        tf.summary.scalar('loss_cnr',loss_cnr)
        tf.summary.scalar('loss_mty',loss_mty)
        tf.summary.scalar('loss_mes',loss_mes)
        tf.summary.scalar('loss_rew',loss_rew)
        tf.summary.scalar('entropy_cty',entropy_cty)
        tf.summary.scalar('entropy_cnr',entropy_cnr)
        tf.summary.scalar('entropy_mty',entropy_mty)
        tf.summary.scalar('entropy_mes',entropy_mes)
        tf.summary.scalar('entropy_rew',entropy_rew)
        tf.summary.scalar('vf_loss',vf_loss)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('avgDiscReward',tf.reduce_mean(R))
        tf.summary.scalar('avgAdvantage',tf.reduce_mean(ADV))

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        merged = tf.summary.merge_all()

        def train(obs, states, rewards, masks, actions, values,advantages):
            advs = advantages #rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr, M: masks}
            if states != []:
                td_map[train_model.S] = states
                #td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ ,summary= sess.run(
                [pg_loss, vf_loss, entropy, _train,merged],
                td_map
            )
            return policy_loss, value_loss, policy_entropy, summary

        def save(save_path, filename= 'saved.pkl'):
            path = save_path + "/" + filename
            ps = sess.run(params)
            make_path(save_path)
            joblib.dump(ps, path)
            print('Model is saved')

        def load(load_path, filename= 'saved.pkl'):
            path = load_path  + "/" +  filename
            if osp.isfile(path):
                loaded_params = joblib.load(path)
                restores = []
                for p, loaded_p in zip(params, loaded_params):
                    restores.append(p.assign(loaded_p))
                ps = sess.run(restores)
                print('Model is loading')

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load

        tf.global_variables_initializer().run(session=sess)
