"""
To balance actor and critic losses, the rewards are divided through by the standard deviation of a rolling discounted sum of the rewards (without subtracting and re-adding the mean).

Code is based on: https://github.com/openai/phasic-policy-gradient/blob/master/phasic_policy_gradient/reward_normalizer.py

Reference: https://arxiv.org/pdf/2005.12729.pdf

"""

import numpy as np


class RunningMeanStd:
    def __init__(
        self,
        epsilon=1e-4,  # initial count (with mean=0 ,var=1)
        shape=(),  # unbatched shape of data, shape[0] is the batch size
    ):
        super().__init__()
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = M2 / (tot_count - 1)
        self.count = tot_count


class RunningRewardScaler:
    """
    Pseudocode can be found in https://arxiv.org/pdf/1811.02553.pdf
    section 9.3 (which is based on our Baselines code, haha)
    Motivation is that we'd rather normalize the returns = sum of future rewards,
    but we haven't seen the future yet. So we assume that the time-reversed rewards
    have similar statistics to the rewards, and normalize the time-reversed rewards.
    """

    def __init__(self, num_envs, cliprew=10.0, gamma=0.99, epsilon=1e-8, per_env=False):
        ret_rms_shape = (num_envs,) if per_env else ()
        self.ret_rms = RunningMeanStd(shape=ret_rms_shape)
        self.cliprew = cliprew
        self.ret = np.zeros(num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.per_env = per_env

    def __call__(self, reward, first):
        rets = backward_discounted_sum(
            prevret=self.ret, reward=reward, first=first, gamma=self.gamma
        )
        self.ret = rets[:, -1]
        self.ret_rms.update(rets if self.per_env else rets.reshape(-1))
        return self.transform(reward)

    def transform(self, reward):
        return np.clip(
            reward / np.sqrt(self.ret_rms.var + self.epsilon),
            -self.cliprew,
            self.cliprew,
        )


def backward_discounted_sum(
    prevret,  # value predictions
    reward,  # reward
    first,  # mark beginning of episodes"
    gamma,  # discount
):
    assert first.ndim == 2
    _, nstep = reward.shape
    ret = np.zeros_like(reward)
    for t in range(nstep):
        prevret = ret[:, t] = reward[:, t] + (1 - first[:, t]) * gamma * prevret
    return ret
