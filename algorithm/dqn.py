import numpy as np
import tensorflow as tf
from .deepq import DeepQ

class DQN(DeepQ):
    def __init__(self, args):
        super().__init__(args)

    def create_network(self):
        with tf.variable_scope('main'):
            with tf.variable_scope('value'):
                self.q = self.value_net(self.obs_ph)
                self.q_pi = tf.reduce_max(self.q, axis=1, keepdims=True)
            self.q_train_list = [self.q]
            if self.args.double:
                with tf.variable_scope('value', reuse=True):
                    self.q_next = self.value_net(self.obs_next_ph)
                    self.pi_next = tf.one_hot(tf.argmax(self.q_next, axis=1), self.acts_num, dtype=tf.float32)

        with tf.variable_scope('target'):
            with tf.variable_scope('value'):
                if self.args.double:
                    self.q_t = tf.reduce_sum(self.value_net(self.obs_next_ph)*self.pi_next, axis=1, keepdims=True)
                else:
                    self.q_t = tf.reduce_max(self.value_net(self.obs_next_ph), axis=1, keepdims=True)
