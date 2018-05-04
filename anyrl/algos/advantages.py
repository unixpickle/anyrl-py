"""
Advantage estimation routines.
"""

from abc import ABC, abstractmethod


class AdvantageEstimator(ABC):
    """
    A technique for using a value function to estimate
    the advantage function.
    """
    @abstractmethod
    def advantages(self, rollouts):
        """
        Compute the advantages for the rollouts and return
        a list of advantage lists of the same shape as the
        rollouts.
        """
        pass

    def targets(self, rollouts):
        """
        Compute new targets for the value function.

        The result is the same shape as advantages().
        """
        res = [x.copy() for x in self.advantages(rollouts)]
        for rollout_idx, rollout in enumerate(rollouts):
            for timestep, model_out in enumerate(rollout.step_model_outs):
                value_out = model_out['values'][0]
                res[rollout_idx][timestep] += value_out
        return res


class GAE(AdvantageEstimator):
    """
    An implementation of Generalized Advantage Estimation.
    """

    def __init__(self, lam, discount, target_lam=None):
        self.lam = lam
        self.discount = discount
        self.target_lam = target_lam

    def advantages(self, rollouts):
        res = []
        for rollout in rollouts:
            adv = 0
            advs = []
            for i in range(rollout.num_steps)[::-1]:
                delta = rollout.rewards[i] - rollout.predicted_value(i)
                if i+1 < len(rollout.model_outs):
                    delta += self.discount * rollout.predicted_value(i + 1)
                adv *= self.lam * self.discount
                adv += delta
                advs.append(adv)
            res.append(advs[::-1])
        return res

    def targets(self, rollouts):
        if self.target_lam is None:
            return super(GAE, self).targets(rollouts)
        proxy = GAE(lam=self.target_lam, discount=self.discount)
        return proxy.targets(rollouts)
