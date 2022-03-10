import numpy as np
from test import Tester
from envs import make_env, envs_collection
from learner import create_learner, learner_collection
from algorithm import create_agent, algorithm_collection
from algorithm.replay_buffer import create_buffer, buffer_collection
from utils.os_utils import get_arg_parser, get_logger, str2bool

def get_args():
    parser = get_arg_parser()

    # basic arguments
    parser.add_argument('--tag', help='the terminal tag in logger', type=str, default='')
    parser.add_argument('--env', help='gym env id', type=str, default='Alien')
    parser.add_argument('--alg', help='backend algorithm', type=str, default='dqn', choices=algorithm_collection.keys())
    parser.add_argument('--learn', help='the type of training method', type=str, default='atari_dbadp', choices=learner_collection.keys())

    args, _ = parser.parse_known_args()

    # env arguments
    parser.add_argument('--gamma', help='discount factor', type=np.float32, default=0.99)

    def atari_args():
        parser.add_argument('--sticky', help='whether to use sticky actions', type=str2bool, default=False)
        parser.add_argument('--noop', help='the number of noop actions while starting a new episode', type=np.int32, default=30)
        parser.add_argument('--frames', help='the number of stacked frames', type=np.int32, default=4)
        parser.add_argument('--rews_scale', help='the clip scale of rewards', type=np.float32, default=1.0)
        parser.add_argument('--test_eps', help='the scale of random action noise in atari testing', type=np.float32, default=0.001)

    env_args_collection = {
        'atari': atari_args
    }
    env_args_collection[envs_collection[args.env]]()

    # training arguments
    parser.add_argument('--epochs', help='the number of epochs', type=np.int32, default=10)
    parser.add_argument('--cycles', help='the number of cycles per epoch', type=np.int32, default=20)
    parser.add_argument('--iterations', help='the number of iterations per cycle', type=np.int32, default=500)
    parser.add_argument('--timesteps', help='the number of timesteps per iteration', type=np.int32, default=100)

    # testing arguments
    parser.add_argument('--test_rollouts', help='the number of rollouts to test per cycle', type=np.int32, default=5)
    parser.add_argument('--test_timesteps', help='the number of timesteps per rollout', type=np.int32, default=27000)
    parser.add_argument('--save_rews', help='whether to save cumulative rewards', type=str2bool, default=True)

    # buffer arguments
    parser.add_argument('--buffer', help='the type of replay buffer', type=str, default='framestack', choices=buffer_collection.keys())
    parser.add_argument('--buffer_size', help='the number of transitions in replay buffer', type=np.int32, default=1000000)
    parser.add_argument('--batch_size', help='the size of sample batch', type=np.int32, default=32)
    parser.add_argument('--warmup', help='the number of timesteps for buffer warmup', type=np.int32, default=20000)

    # algorithm arguments
    def q_learning_args():
        parser.add_argument('--train_batches', help='the number of batches to train per iteration', type=np.int32, default=25)
        parser.add_argument('--train_target', help='the frequency of target network updating', type=np.int32, default=8000)

        parser.add_argument('--eps_l', help='the beginning percentage of epsilon greedy explorarion', type=np.float32, default=1.00)
        parser.add_argument('--eps_r', help='the final percentage of epsilon greedy explorarion', type=np.float32, default=0.01)
        parser.add_argument('--eps_decay', help='the number of steps for epsilon decaying', type=np.int32, default=250000)

        parser.add_argument('--optimizer', help='the optimizer to use', type=str, default='adam', choices=['adam', 'rmsprop'])
        args, _ = parser.parse_known_args()
        if args.optimizer=='adam':
            parser.add_argument('--q_lr', help='the learning rate of value network', type=np.float32, default=0.625e-4)
            parser.add_argument('--Adam_eps', help='the epsilon factor of Adam optimizer', type=np.float32, default=1.5e-4)
        elif args.optimizer=='rmsprop':
            parser.add_argument('--q_lr', help='the learning rate of value network', type=np.float32, default=2.5e-4)
            parser.add_argument('--RMSProp_decay', help='the decay factor of RMSProp optimizer', type=np.float32, default=0.95)
            parser.add_argument('--RMSProp_eps', help='the epsilon factor of RMSProp optimizer', type=np.float32, default=1e-2)

        parser.add_argument('--nstep', help='the parameter of n-step bootstrapping', type=np.int32, default=1)

    def dqn_args():
        q_learning_args()
        parser.add_argument('--double', help='whether to use double trick', type=str2bool, default=True)
        parser.add_argument('--dueling', help='whether to use dueling trick', type=str2bool, default=False)

    def cddqn_args():
        q_learning_args()
        parser.add_argument('--dueling', help='whether to use dueling trick', type=str2bool, default=False)

    algorithm_args_collection = {
        'dqn': dqn_args,
        'cddqn': cddqn_args,
    }
    algorithm_args_collection[args.alg]()

    args = parser.parse_args()

    logger_name = args.alg+'-'+args.env+'-'+args.learn
    if args.tag!='': logger_name = args.tag+'-'+logger_name
    args.logger = get_logger(logger_name)

    for key, value in args.__dict__.items():
        if key!='logger':
            args.logger.info('{}: {}'.format(key,value))

    return args

def experiment_setup(args):
    env = make_env(args)
    args.acts_dims = env.acts_dims
    args.obs_dims = env.obs_dims

    args.buffer = buffer = create_buffer(args)
    args.agent = agent = create_agent(args)
    args.learner = learner = create_learner(args)
    args.logger.info('*** network initialization complete ***')
    args.tester = tester = Tester(args)
    args.logger.info('*** tester initialization complete ***')

    return env, agent, buffer, learner, tester
