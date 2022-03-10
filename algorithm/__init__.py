from .dqn import DQN
from .cddqn import ClippedDDQN

algorithm_collection = {
	'dqn': DQN,
	'cddqn': ClippedDDQN,
}

def create_agent(args):
	return algorithm_collection[args.alg](args)
