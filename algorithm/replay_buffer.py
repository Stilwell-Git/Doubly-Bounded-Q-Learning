import numpy as np
import copy
from envs import envs_collection

class Episode_FrameStack:
	def __init__(self, info, gamma):
		self.gamma = gamma
		self.common_info = [
			'obs', 'obs_next', 'frame_next',
			'acts', 'rews', 'done'
		]
		self.ep = {
			'obs': [],
			'acts': [],
			'rews': [],
			'done': []
		}
		for key in info.keys():
			if not(key in self.common_info):
				self.ep[key] = []
		self.ep_len = 0
		self.frames = info['obs'].shape[-1]
		for i in range(self.frames):
			self.ep['obs'].append(copy.deepcopy(info['obs'][:,:,i]))

	def insert(self, info):
		self.ep_len += 1
		self.ep['obs'].append(copy.deepcopy(info['frame_next']))
		self.ep['acts'].append(copy.deepcopy(info['acts']))
		self.ep['rews'].append(copy.deepcopy(info['rews']))
		self.ep['done'].append(copy.deepcopy(info['done']))
		for key in info.keys():
			if not(key in self.common_info):
				self.ep[key].append(copy.deepcopy(info[key]))

	def get_obs(self, idx):
		idx += 1
		obs = np.stack(self.ep['obs'][idx:idx+self.frames], axis=-1)
		return obs.astype(np.float32)/255.0

	def sample(self, step=1):
		if step==1:
			idx = np.random.randint(self.ep_len)
			info = {
				'obs': self.get_obs(idx-1),
				'obs_next': self.get_obs(idx),
				'acts': copy.deepcopy(self.ep['acts'][idx]),
				'rews': [copy.deepcopy(self.ep['rews'][idx])],
				'done': [copy.deepcopy(self.ep['done'][idx])]
			}
		else:
			if not self.ep['done'][-1]:
				idx = np.random.randint(self.ep_len-step)
			else:
				idx = np.random.randint(self.ep_len)
			rews, gamma_prod, k = 0.0, 1.0, step-1
			for i in range(step):
				rews += gamma_prod * self.ep['rews'][idx+i]
				gamma_prod *= self.gamma
				if self.ep['done'][idx+i]:
					k = i
					break
			info = {
				'obs': self.get_obs(idx-1),
				'obs_next': self.get_obs(idx+k),
				'acts': copy.deepcopy(self.ep['acts'][idx]),
				'rews': [rews],
				'done': [copy.deepcopy(self.ep['done'][idx+k])]
			}
			if 'hash_next' in self.ep.keys():
				info['hash_next'] = self.ep['hash_next'][idx+k]
		for key in self.ep.keys():
			if (not(key in self.common_info)) and (not(key in info.keys())):
				info[key] = copy.deepcopy(self.ep[key][idx])
		return info

class ReplayBuffer_FrameStack:
	def __init__(self, args):
		self.args = args
		self.in_head = True
		self.counter = 0
		self.steps_counter = 0
		self.buffer_size = self.args.buffer_size

		self.ep = []
		self.length = 0
		self.head_idx = 0
		self.ram_idx = []

	def store_transition(self, info):
		if self.in_head:
			new_ep = Episode_FrameStack(info, self.args.gamma)
			self.ep.append(new_ep)
		self.ep[-1].insert(info)
		self.ram_idx.append(self.counter)
		self.length += 1

		if self.length>self.buffer_size:
			del_len = self.ep[0].ep_len
			self.ep.pop(0)
			self.head_idx += 1
			self.length -= del_len
			self.ram_idx = self.ram_idx[del_len:]

		self.steps_counter += 1
		self.in_head = info['done']
		if info['done']: self.counter += 1

	def sample_batch(self, batch_size=-1):
		if batch_size==-1: batch_size = self.args.batch_size
		batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[])

		for i in range(batch_size):
			idx = self.ram_idx[np.random.randint(self.length-self.args.nstep)]-self.head_idx
			info = self.ep[idx].sample(step=self.args.nstep)
			if i==0:
				batch = { key: [info[key]] for key in info.keys() }
			else:
				for key in info.keys():
					batch[key].append(info[key])

		return batch


buffer_collection = {
	'framestack': ReplayBuffer_FrameStack,
}

def create_buffer(args):
	return buffer_collection[args.buffer](args)
