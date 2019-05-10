import tensorflow as tf
import numpy as np
import gym
import time
MAX_EPISODES=200
MAX_EP_STEPS=200
LR_A=0.001
LR_C=0.002
GAMMA=0.9
TAU=0.01
MEMORY_CAPACITY=10000
BATCH_SIZE=32

RENDER=False
ENV_NAME='Pendulum-v0'

class DDPG(object):
    def __init__(self,a_dim,s_dim,a_bound):
        self.memory=np.zeros((MEMORY_CAPACITY,s_dim*2+a_dim+1),dtype=np.float32)
        self.pointer=0
        self.sess=tf.Session()
        self.a_dim,self.s_dim,self.a_bound=a_dim,s_dim,a_bound
        self.S=tf.placeholder(tf.float32,[None,s_dim],'s')
        self.S_=tf.placeholder(tf.float32,[None,s_dim],'s_')
        self.R=tf.placeholder(tf.float32,[None,1],'r')

        with tf.variable_scope('Actor'):
            self.a=self._build_a(self.S,scope='eval',trainable=True)
            a_=self._build_a(self.S_,scope='target',trainable=False)
        with tf.variable_scope('Critic'):
            q=self._build_c(self.S,self.a,scope='eval',trainable=True)
            q_=self._build_c(self.S_,a_,scope='target',trainable=False)

        self.ae_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Actor/eval')
        self.at_params=