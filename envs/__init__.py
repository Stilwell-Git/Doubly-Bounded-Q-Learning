import gym
from .atari import AtariEnv

# this list is obtained from gym/envs/__init__.py
atari_list = [
	'Adventure', 'AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis',
	'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Carnival',
	'Centipede', 'ChopperCommand', 'CrazyClimber', 'Defender', 'DemonAttack', 'DoubleDunk',
	'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar',
	'Hero', 'IceHockey', 'Jamesbond', 'JourneyEscape', 'Kangaroo', 'Krull', 'KungFuMaster',
	'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall', 'Pong', 'Pooyan',
	'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest', 'Skiing',
	'Solaris', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'UpNDown',
	'Venture', 'VideoPinball', 'WizardOfWor', 'YarsRevenge', 'Zaxxon'
]

envs_collection = {
	# Atari envs
	**{
		atari_name : 'atari'
		for atari_name in atari_list
	}
}

make_env_collection = {
	'atari': AtariEnv
}

def make_env(args):
	return make_env_collection[envs_collection[args.env]](args)
