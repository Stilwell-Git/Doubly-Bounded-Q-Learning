from .atari import AtariLearner
from .atari_dbadp import DBADPAtariLearner

learner_collection = {
	'atari': AtariLearner,
	'atari_dbadp': DBADPAtariLearner,
}

def create_learner(args):
	return learner_collection[args.learn](args)
