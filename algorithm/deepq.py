import numpy as np
import tensorflow as tf
from utils.tf_utils import get_vars

class DeepQ(object):
    def __init__(self, args):
        self.args = args
        self.acts_num = args.acts_dims[0]
        self.use_db_target = (self.args.learn[-5:]=='dbadp')
        self.create_model()

        self.train_info = {
            'Q_loss': self.q_loss
        }
        self.step_info = {
            'Q_average': self.q_pi
        }

    def create_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def create_inputs(self):
        self.obs_ph = tf.placeholder(tf.float32, [None]+self.args.obs_dims)
        self.obs_next_ph = tf.placeholder(tf.float32, [None]+self.args.obs_dims)
        self.acts_ph = tf.placeholder(tf.float32, [None]+self.args.acts_dims)
        self.rews_ph = tf.placeholder(tf.float32, [None, 1])
        self.done_ph = tf.placeholder(tf.float32, [None, 1])

        if self.use_db_target:
            self.q_lb_ph = tf.placeholder(tf.float32, [None, 1])

    def value_net(self, obs_ph):
        # the default architecture of value network
        with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
            q_conv1 = tf.layers.conv2d(obs_ph, 32, 8, 4, 'same', activation=tf.nn.relu, name='q_conv1')
            q_conv2 = tf.layers.conv2d(q_conv1, 64, 4, 2, 'same', activation=tf.nn.relu, name='q_conv2')
            q_conv3 = tf.layers.conv2d(q_conv2, 64, 3, 1, 'same', activation=tf.nn.relu, name='q_conv3')
            q_conv3_flat = tf.layers.flatten(q_conv3)

            q_dense_act = tf.layers.dense(q_conv3_flat, 512, activation=tf.nn.relu, name='q_dense_act')
            q_act = tf.layers.dense(q_dense_act, self.acts_num, name='q_act')

            if self.args.dueling:
                q_dense_base = tf.layers.dense(q_conv3_flat, 512, activation=tf.nn.relu, name='q_dense_base')
                q_base = tf.layers.dense(q_dense_base, 1, name='q_base')
                q = q_base + q_act - tf.reduce_mean(q_act, axis=1, keepdims=True)
            else:
                q = q_act
        return q

    def create_network(self):
        raise NotImplementedError

    def create_operators(self):
        self.target = tf.stop_gradient(self.rews_ph+(1.0-self.done_ph)*(self.args.gamma**self.args.nstep)*self.q_t)
        if self.use_db_target:
            self.target = tf.maximum(self.target, self.q_lb_ph)
        self.q_acts_train_list = [tf.reduce_sum(q*self.acts_ph, axis=1, keepdims=True) for q in self.q_train_list]
        self.q_loss = tf.add_n([tf.losses.huber_loss(self.target, q_acts) for q_acts in self.q_acts_train_list])
        if self.args.optimizer=='adam':
            self.q_optimizer = tf.train.AdamOptimizer(self.args.q_lr, epsilon=self.args.Adam_eps)
        elif self.args.optimizer=='rmsprop':
            self.q_optimizer = tf.train.RMSPropOptimizer(self.args.q_lr, decay=self.args.RMSProp_decay, epsilon=self.args.RMSProp_eps)
        self.q_train_op = self.q_optimizer.minimize(self.q_loss, var_list=get_vars('main/value'))

        self.target_update_op = tf.group([
            v_t.assign(v)
            for v, v_t in zip(get_vars('main'), get_vars('target'))
        ])

        self.saver=tf.train.Saver()
        self.init_op = tf.global_variables_initializer()
        self.target_init_op = tf.group([
            v_t.assign(v)
            for v, v_t in zip(get_vars('main'), get_vars('target'))
        ])

    def create_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_session()
            self.create_inputs()
            self.create_network()
            self.create_operators()
        self.init_network()

    def init_network(self):
        self.sess.run(self.init_op)
        self.sess.run(self.target_init_op)

    def step(self, obs, explore=False, test_info=False):
        if (not test_info) and (self.args.buffer.steps_counter<self.args.warmup):
            return np.random.randint(self.acts_num)

        # eps-greedy exploration
        if explore and np.random.uniform()<=self.args.eps_act:
            return np.random.randint(self.acts_num)

        feed_dict = {
            self.obs_ph: [obs/255.0]
        }
        q_value, info = self.sess.run([self.q, self.step_info], feed_dict)
        action = np.argmax(q_value[0])

        if test_info: return action, info
        return action

    def feed_dict(self, batch):
        def one_hot(idx):
            idx = np.array(idx)
            batch_size = idx.shape[0]
            res = np.zeros((batch_size, self.acts_num), dtype=np.float32)
            res[np.arange(batch_size),idx] = 1.0
            return res

        feed_dict = {
            self.obs_ph: np.array(batch['obs']),
            self.obs_next_ph: np.array(batch['obs_next']),
            self.acts_ph: one_hot(batch['acts']),
            self.rews_ph: np.array(batch['rews']),
            self.done_ph: batch['done']
        }

        if self.use_db_target:
            feed_dict[self.q_lb_ph]= batch['rets']

        return feed_dict

    def train(self, batch):
        feed_dict = self.feed_dict(batch)
        info, _ = self.sess.run([self.train_info, self.q_train_op], feed_dict)
        return info

    def target_update(self):
        self.sess.run(self.target_update_op)

    def save_model(self, save_path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, save_path)

    def load_model(self, load_path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, load_path)
