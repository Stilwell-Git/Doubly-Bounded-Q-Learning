import time
import numpy as np
from algorithm.adp import AbstractedDynamicProgramming

class DiscountedQueue:
    def __init__(self, args):
        self.args = args
        self.buffer = args.buffer
        self.transitions = []

    def store_transition(self, transition):
        self.transitions.append(transition)
        if transition['done']:
            self.pop()

    def pop(self):
        next_state_id, rets = 0, 0.0
        for transition in self.transitions[::-1]:
            rews = transition['rews'] # This reward value has been clipped.
            assert (rews>=-self.args.rews_scale) and (rews<=self.args.rews_scale)
            rets = rews+self.args.gamma*rets
            transition['hash_next'] = next_state_id
            current_state_id = self.args.adp_lib.get_state_id(transition['obs'])

            self.args.adp_lib.add_transition(current_state_id, transition['acts'], rews, next_state_id)
            self.args.adp_lib.update_state(current_state_id)
            next_state_id = current_state_id

        for transition in self.transitions:
            self.buffer.store_transition(transition)

        self.transitions = []

class DBADPAtariLearner:
    def __init__(self, args):
        self.steps_counter = 0
        self.target_count = 0
        self.learner_info = [
            'Epsilon',
            'TimeCost_ADP'
        ]

        args.eps_act = args.eps_l
        self.eps_decay = (args.eps_l-args.eps_r)/args.eps_decay

        self.queue = DiscountedQueue(args)
        args.adp_lib = AbstractedDynamicProgramming(args)

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
                self.queue.store_transition(transition)
                if done:
                    obs = env.reset()

            if buffer.steps_counter>=args.warmup:
                for _ in range(args.train_batches):
                    batch = buffer.sample_batch()
                    batch['rets'] = []
                    for r, done, hash_next in zip(batch['rews'], batch['done'], batch['hash_next']):
                        avg_ret = r[0] + (1.0-done[0]) * (args.gamma**args.nstep) * args.adp_lib.get_state_value(hash_next)
                        batch['rets'].append([avg_ret])
                    info = agent.train(batch)
                    args.logger.add_dict(info)
                    self.target_count += 1
                    if self.target_count%args.train_target==0:
                        agent.target_update()
                        start_time = time.time()
                        self.args.adp_lib.update_buffer()
                        args.logger.add_record('TimeCost_ADP', time.time()-start_time)

        args.logger.add_record('Epsilon', args.eps_act)
