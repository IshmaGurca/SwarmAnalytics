
'''
Based on OPEN AI A2C implementation in the "Baselines" package
https://github.com/openai/baselines
'''


import numpy as np
import tensorflow as tf
from baselines.a2c.utils import mylstm, memCell, conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample, check_shape
import gym


class MyPolicy(object):

    def __init__(self, sess, env, nsteps, reuse=False):
        
        nbatch = nsteps
        obs_space = env.ObsSpace
        ob_shape = (nbatch, sum(obs_space))
        ac_space =  env.ActSpace
        msg_length = ac_space[3]
        ncty = env.ChatType.n
        ncnr = env.ChatNumber.n
        nmty = env.MessageType.n
        nmsg = env.MessageText.n
        nrew = env.FeedbackReward.n
        MemNH = sum(obs_space)
        #ob_shape = (nbatch, nh)
        #nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        S = tf.placeholder(tf.float32,[nbatch,MemNH*2])
        with tf.variable_scope("model", reuse=reuse):

            ''' 
            Convert X in form of indices e.g. [[1,2,3,4],[3,4,5,2]]
            to OneHot and concenate one hot to 1 array
            '''
            X_cty, X_cnr, X_mty, X_msg = tf.split(X,num_or_size_splits = obs_space, axis = 1)
            X_cty_oh = tf.squeeze(env.ChatType.IndexToOneHot(X_cty),[1])
            X_cnr_oh = tf.squeeze(env.ChatNumber.IndexToOneHot(X_cnr),[1])
            X_mty_oh = tf.squeeze(env.MessageType.IndexToOneHot(X_mty),[1])
            X_msg_oh = env.MessageText.IndexToOneHot(X_msg)
            m_shape = X_msg_oh.shape
            X_msg_oh = tf.reshape(X_msg_oh,[int(m_shape[0]),int(m_shape[1])*int(m_shape[2])])
    

            X_oh = tf.concat([X_cty_oh,X_cnr_oh,X_mty_oh,X_msg_oh], axis = 1)

            '''
            Memory Cell to use X and inner State of prior message
            '''

            h,s0 = memCell(X,S,'memcell', nh=MemNH)

            h3 = fc(h,'fc', nh=MemNH, init_scale=np.sqrt(2))

            fh = fc(h3, 'fc1', nh=MemNH, init_scale=np.sqrt(2))
            '''
            Break down hidden state to:
                1. Channel Type (output 5 classes)
                2. Channel Nr. (output 5 classes)
                3. Message Type (output x classes)
                4. Message ... LSTM to genearate 500 output sequence
                5. Reward (output  x classe eg. [-100, 100])
            '''
            cty = fc(fh, 'cty', ncty, act=lambda x:x) #act=lambda x:x)
            cnr = fc(fh, 'cnr', ncnr, act=lambda x:x)
            mty = fc(fh, 'mty', nmty, act=lambda x:x)

            #Gen message with LSTM
            #1. generate inital hidden state
            mesh = fc(fh,'mesh', nmsg)
            mesc = np.zeros(mesh.shape)
            mess = tf.concat(axis=1, values=[mesc, mesh])
            #2. inital x as  <start> one hot
            inital_x = env.MessageText.IndexSartMessage(nbatch)
            mes, state, mask = mylstm(inital_x,msg_length, mess, "meslstm",nmsg,env)
            
            rew = fc(fh,'rew',nrew, act=lambda x:x)

            #pi = fc(h4, 'pi', nact, act=lambda x:x)
            vf = fc(fh, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]

        # action as combination of output

        cty0 = tf.reshape(sample(cty),[nbatch,1])
        cnr0 = tf.reshape(sample(cnr),[nbatch,1])
        mty0 = tf.reshape(sample(mty),[nbatch,1])
        mes0 = sample(mes,2)
        rew0 = tf.reshape(sample(rew),[nbatch,1])

        #mes0 = tf.reshape(mes0,[int(mes0.shape[0]),int(mes0.shape[1])])

        #a0 = sample(pi)
        a0 = tf.concat([cty0,cnr0,mty0,mes0,rew0], axis = 1)
        
        #self.initial_state = [] #not stateful
        self.initial_state = np.zeros([nbatch,MemNH*2])


        def step(ob,state, *_args, **_kwargs):
            a, v, s = sess.run([a0, v0, s0], {X:ob,S:state})
            return a, v, s #[] #dummy state

        def value(ob,state, *_args, **_kwargs):
            return sess.run(v0, {X:ob,S:state})

        self.X = X
        self.S = S
        self.cty = cty
        self.cnr = cnr
        self.mty = mty
        self.mes = mes
        self.mask = mask
        self.rew = rew
        #self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value