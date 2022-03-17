import numpy as np

class AtariLearner:
	def __init__(self, args):
		self.target_count = 0
		self.steps_counter = 0
		self.learner_info = [
			'Epsilon'
		]

		args.eps_act = args.eps_l
		self.eps_decay = (args.eps_l-args.eps_r)/args.eps_decay

	def learn(self, args, env, agent, buffer):
		for _ in range(args.iterations):
			obs = env.get_obs()
			for timestep in range(args.timesteps):
				obs_pre = obs
				action = agent.step(obs, explore=True)
				args.eps_act = max(args.eps_r, args.eps_act-self.eps_decay)
				obs, reward, done, _ = env.step(action)
				self.steps_counter += 1
				frame = env.get_frame()
				transition = {
					'obs': obs_pre,
					'obs_next': obs,
					'frame_next': frame,
					'acts': action,
					'rews': np.clip(reward, -args.rews_scale, args.rews_scale),
					'done': done
				}
				buffer.store_transition(transition)
				if done:
					obs = env.reset()
			args.logger.add_record('Epsilon', args.eps_act)

			if buffer.steps_counter>=args.warmup:
				for _ in range(args.train_batches):
					info = agent.train(buffer.sample_batch())
					args.logger.add_dict(info)
					self.target_count += 1
					if self.target_count%args.train_target==0:
						agent.target_update()
