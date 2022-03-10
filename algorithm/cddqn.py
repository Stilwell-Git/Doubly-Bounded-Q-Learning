import numpy as np
import tensorflow as tf
from .deepq import DeepQ

class ClippedDDQN(DeepQ):
    def __init__(self, args):
        super().__init__(args)

    def create_network(self):
        with tf.variable_scope('main'):
            with tf.variable_scope('value_1'):
                self.q = self.value_net(self.obs_ph)
                self.q_pi = tf.reduce_max(self.q, axis=1, keepdims=True)
            with tf.variable_scope('value_2'):
                self.q_2 = self.value_net(self.obs_ph)
            self.q_train_list = [self.q, self.q_2]

        with tf.variable_scope('target'):
            with tf.variable_scope('value_1'):
                self.q_t_1 = self.value_net(self.obs_next_ph)
            with tf.variable_scope('value_2'):
                self.q_t_2 = self.value_net(self.obs_next_ph)
            self.pi_next = tf.one_hot(tf.argmax(self.q_t_1, axis=1), self.acts_num, dtype=tf.float32)
            self.q_t = tf.reduce_sum(tf.minimum(self.q_t_1, self.q_t_2)*self.pi_next, axis=1, keepdims=True)
