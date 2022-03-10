import ctypes
import numpy as np
from utils.c_utils import load_c_lib, c_ptr, c_int, c_double

class AbstractedDynamicProgramming:
    def __init__(self, args):
        self.args = args
        self.adp_lib = load_c_lib('algorithm/adp_lib.cpp')
        self.adp_lib.get_state_tot.restype = ctypes.c_int
        self.adp_lib.get_state_id.restype = ctypes.c_int
        self.adp_lib.get_state_value.restype = ctypes.c_double
        self.adp_lib.init(c_double(np.float64(args.gamma)))

    def get_state_tot(self):
        return self.adp_lib.get_state_tot()

    def get_state_id(self, obs):
        obs = obs.astype(np.uint8)
        return self.adp_lib.get_state_id(c_ptr(obs))

    def add_transition(self, state_id, action, reward, next_state_id):
        self.adp_lib.add_transition(c_int(state_id), c_int(action), c_double(np.float64(reward)), c_int(next_state_id))

    def update_state(self, state_id):
        # update the value of a single state
        self.adp_lib.update_state(c_int(state_id))

    def update_buffer(self):
        # perform one iteration of value iteration
        self.adp_lib.update_buffer()

    def get_state_value(self, state_id):
        return self.adp_lib.get_state_value(c_int(state_id))
