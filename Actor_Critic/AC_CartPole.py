import numpy as np
import tensorflow as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)

OUTPUT_GRAPH=False
MAX_EPISODE=3000
DISPLAY_REWARD_THRESHOLD=200
MAX_EP_STEPS=1000
RENDER=False
GAMMA=0.9
LA_A=0.001
LR_C=0.01

env=gym.make('CartPole-vo')
env.seed(1)
env=env.unwrapped

N_F=env.observation_space.shape[0]
N_A=env.action_space.n
class Actor(object):
    def __init__(self,sess,n_features,n_actions,lr=0.001):
        self.sess=sess

        self.s=tf.placeholder(tf.float32,[1,n_features],'state')
        self.a=tf.placeholder(tf.int32,None,'act')
        self.tf_error=tf.placeholder(tf.float32,None,"td_error")

        with tf.variable_scope('Actor')
            l1=tf.layer.dense(inputs=self.s,units=20,activation=tf.random_normal_initializer(0.,.1),
                              bias_initializer=tf.constant_initializer(0.1),name='l1')
            self.acts_prob=tf.layers.dense(
                input(l1,units=n_actions,activation=tf.nn.softmax,kernel_initializer=tf.random_normal_initializer(0.,.1),
                      bias_initializer=tf.constant_initializer(0.1),name='acts_prob')
            )
        with tf.variable_scope('exp_v'):
            log_prob=tf.log(self.acts_prob[0,self.a])
            self.exp_v=tf.reduce_mean(log_prob*self.td_error)
        with tf.variable_scope('train'):
            self.train_op=tf.train.AdamOptimizer(lr).minimize(-self.exp_v)
    def learn(self,s,a,td):
        s=s[np.newaxis,:]
        feed_dict={self.s:s,self.a:a,self.td_error:td}
        _,exp_v=self.sess.run([self.train_op,self.exp_v],feed_dict)
        return exp_v

    def choose_action(self,s):
        s=s[np.newaxis,:]
        probs=self.sess.run(self.acts_prob,{self.s:s})
        return np.random.choice(np.arange(probs.shape[1]),p=probs.ravel())
