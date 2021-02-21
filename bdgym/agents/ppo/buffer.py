"""PPO Buffer """
import torch
import numpy as np
import scipy.signal


def discount_cumsum(x, discount):
    """Calculate discounted cumulative sum """
    return scipy.signal.lfilter(
        [1], [1, float(-discount)], x[::-1], axis=0
    )[::-1]


class PPOBuffer:
    """PPO Buffer """

    def __init__(self, capacity, obs_dim, gamma=0.99, lam=0.95, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.o_buf = np.zeros((capacity, *obs_dim), dtype=np.float32)
        self.a_buf = np.zeros((capacity, ), dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32)
        self.logp_buf = np.zeros(capacity, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx = 0, 0

    def store(self, o, a, r, v, logp):
        """Store a step """
        assert self.ptr < self.capacity
        self.o_buf[self.ptr] = o
        self.a_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.val_buf[self.ptr] = v
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """Call this at end of trajectory """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE - advantage estimate
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(
            deltas, self.gamma * self.lam
        )

        # Reward-to-go targets
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """Get all trajectories currently stored"""
        assert self.ptr == self.capacity
        self.ptr, self.path_start_idx = 0, 0

        # normalize advantage
        norm_adv = self.adv_buf - np.mean(self.adv_buf)
        norm_adv /= np.std(self.adv_buf)

        data = dict(
            obs=self.o_buf,
            act=self.a_buf,
            ret=self.ret_buf,
            adv=norm_adv,
            logp=self.logp_buf
        )
        return {
            k: torch.from_numpy(v).to(self.device) for k, v in data.items()
        }
