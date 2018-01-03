from abc import ABC, abstractmethod

import numpy as np

import copy
import math
import time

class TreeNode(object):
    """
    Is an abstract node data structure that has a parent (or not, i.e. root nodes), an action and a
    value that is derived by the wrapped gym environment, which can be a direct copy of the
    parent's environment.

    The node may also have children as well as a rollout associated with the parent.
    """
    def __init__(self, parent=None, value=None, action=None, env=None, rollout=None):
        self.parent = parent
        self.value = value
        self.action = action
        self.env = env
        self.rollout = rollout
        self.children = {}

    def __next__(self):
        """Creates a copy of the parent environment and updates this node."""
        self.rollout = copy.deepcopy(self.parent.rollout)
        self.env = copy.deepcopy(self.parent.env)
        obs, rew, done, info = self.env.step(self.action[0])
        self.rollout.rewards.append(rew)
        self.rollout.infos.append(info)
        self.rollout.observations.append(obs)
        return obs, done

    next = __next__

class TreeSearch(ABC):
    """
    Describes abstract tree search algorithms, such as Monte Carlo Tree Search.
    """
    @abstractmethod
    def search(self, env, rollout):
        pass

    @abstractmethod
    def select_leaf(self, node):
        pass

    @abstractmethod
    def select_action(self, node):
        pass

class MonteCarloTreeSearch(TreeSearch):
    """
    Naive implementation of Monte Carlo Tree Search for gym environments that uses TreeNode's.

    This implementation supports both time-constrained and iteration-based search.
    """
    def __init__(self,
                 use_time_search=False,
                 search_time_secs=.5,
                 search_iterations=1600,
                 uct_c=1.,
                 tau=1.,
                 max_search_depth=200):
        super(MonteCarloTreeSearch, self).__init__()
        self._search = self.uct_search if use_time_search else self.iter_search
        self.search_time_secs = search_time_secs
        self.search_iterations = search_iterations
        self.uct_c = uct_c
        assert tau != 0., 'temperature `tau` must not be zero, choose a value close to 0 (i.e. 1e-9).'
        self.tau = tau
        self.max_search_depth = max_search_depth

    def expand(self, node, actions):
        node.children = {}
        for action, value in np.ndenumerate(actions):
            child = TreeNode(parent=node, value=value, action=action)
            child.n, child.prior, child.u, child.q = 0, value, value, node.q
            node.children[action] = child

    def search(self, model, env, obs, states, rollout):
        model_out = model.step([obs], states)
        states = model_out['states']
        tmp_states = states
        rollout.model_outs.append(model_out)
        value = model_out['action_params'][0]
        root = TreeNode(env=env, rollout=rollout)
        root.n, root.prior, root.u, root.q = 0, 0, 0, 0
        self.expand(root, value)
        n_searches = self._search(model, states, root, rollout)
        return self.select_action(root), tmp_states

    def select_action(self, node):
        sort = lambda action, node=node: node.children[action].n ** (1 / self.tau)
        sorted_moves = sorted(node.children.keys(), key=sort, reverse=True)
        return sorted_moves[0]

    def uct_search(self, model, states, root, rollout):
        continue_search = True
        start = time.time()
        it = 0
        while ((time.time() - start) < self.search_time_secs) and continue_search:
            continue_search, states = self.mcts(model, states, root, rollout)
            it += 1
        return it

    def iter_search(self, model, states, root, rollout):
        continue_search = True
        it = 0
        while (it < self.search_iterations) and continue_search:
            continue_search, states = self.mcts(model, states, root, rollout)
            it += 1
        return it

    def select_leaf(self, node):
        current = node
        while bool(current.children):
            current = max(current.children.values(), key=lambda n: self.action_score(n))
        return current

    def action_score(self, node):
        return node.q + node.u

    def backup_value(self, node, value):
        node.n += 1
        if node.parent is None:
            return
        node.q += (value - node.q) / node.n
        node.u = self.uct_c * math.sqrt(node.parent.n) * node.prior / node.n
        self.backup_value(node.parent, -value)

    def simulation(self, model, env, obs, total_reward, states, max_search_depth):
        terminal = False
        steps = max_search_depth

        while not terminal and steps:
            model_out = model.step([obs], states)
            states = model_out['states']
            act = model_out['action_params'][0].cumsum().searchsorted(np.random.random(), side='right')
            if not env.action_space.contains(act):
                act = env.action_space.sample()
            state, reward, terminal, info = env.step(act)
            total_reward += reward
            steps -= 1

        while not terminal:
            _, reward, terminal, info = env.step(env.action_space.sample())
            total_reward += reward
        return total_reward

    def mcts(self, model, states, node, rollout):
        """Performs monte-carlo tree search from the given node until we reach an impossible case."""
        if not node.children:
            return False, None

        chosen_leaf = self.select_leaf(node)
        obs, terminal = next(chosen_leaf)

        if terminal:
            if len(chosen_leaf.parent.children) > 1:
                del chosen_leaf.parent.children[chosen_leaf.action]
                return True, states
            else:
                # no where to go ¯\_(ツ)_/¯
                return False, states

        model_out = model.step([obs], states)
        self.expand(chosen_leaf, model_out['action_params'][0])

        value = self.simulation(
            model, copy.deepcopy(chosen_leaf.env),
            chosen_leaf.rollout.observations[-1],
            chosen_leaf.rollout.total_reward,
            states,
            self.max_search_depth)

        self.backup_value(chosen_leaf, value)
        return True, model_out['states']
