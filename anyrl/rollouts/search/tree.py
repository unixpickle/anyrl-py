
from abc import ABC, abstractmethod

import numpy as np

import copy
import math
import time

from anyrl.rollouts import empty_rollout, Roller


def index_action(env, action_coords):
  return action_coords[0]


class TreeNode(object):

  def __init__(self, parent=None, value=None, action=None, env=None, rollout=None):
    self.parent = parent
    self.value = value
    self.action = action
    self.env = env
    self.rollout = rollout
    self.children = {}

  @property
  def is_expanded(self):
    """Return if the node has been acted on or not."""
    return bool(self.children)

  def __next__(self):
    """Creates a copy of the parent environment and updates this node."""
    self.rollout = copy.deepcopy(self.parent.rollout)
    self.env = copy.deepcopy(self.parent.env)
    obs, rew, done, info = self.env.step(index_action(self.env, self.action))
    self.rollout.rewards.append(rew)
    self.rollout.infos.append(info)
    self.rollout.observations.append(obs)
    return obs, done

  next = __next__


class TreeSearch(ABC):

  @abstractmethod
  def search(self, env, rollout):
    pass

  @abstractmethod
  def select_leaf(self, node):
    pass

  @abstractmethod
  def select_action(self, node):
    pass

  def expand(self, node, actions):
    node.children = {action: TreeNode(parent=node, value=value, action=action)
                     for action, value in np.ndenumerate(actions)}
    return node.children


class MonteCarloTreeSearch(TreeSearch):

  def __init__(self, search_time_secs=5, uct_c=5, max_rollout_depth=200):
    super(MonteCarloTreeSearch, self).__init__()
    self.search_time_secs = search_time_secs
    self.uct_c = uct_c
    self.max_rollout_depth = max_rollout_depth

  def expand(self, node, actions):
    node.children = {}
    for action, value in np.ndenumerate(actions):
      child = TreeNode(parent=node, value=value, action=action)
      child.n, child.prior, child.u, child.q = 0, value, value, node.q
      node.children[action] = child
    return node.children

  def search(self, model, env, obs, states, rollout):
    model_out = model.step([obs], states)
    states = model_out['states']
    tmp_states = states
    rollout.model_outs.append(model_out)
    value = model_out['values'][0]
    root = TreeNode(env=env, rollout=rollout)
    root.n, root.prior, root.u, root.q = 0, 0, 0, 0
    self.expand(root, value)
    continue_search = True
    start = time.time()
    while ((time.time() - start) < self.search_time_secs) and continue_search:
      continue_search, states = self.mcts(model, states, root, rollout)
    return self.select_action(root), tmp_states

  def select_leaf(self, node):
    current = node
    while current.is_expanded:
      current = max(current.children.values(), key=lambda n: self.action_score(n))
    return current

  def select_action(self, node):
    sort = lambda action, node=node: node.children[action].n
    sorted_moves = sorted(node.children.keys(), key=sort, reverse=True)
    return sorted_moves[0]

  def action_score(self, node):
    return node.q + node.u

  def backup_value(self, node, value):
    node.n += 1
    if node.parent is None:
      return
    node.q += (value - node.q) / node.n
    node.u = self.uct_c * math.sqrt(node.parent.n) * node.prior / node.n
    self.backup_value(node.parent, -value)

  def simulation(self, model, env, obs, total_reward, states, max_rollout_depth):
    terminal = False
    steps = max_rollout_depth
    while not terminal and steps:
      model_out = model.step([obs], states)
      states = model_out['states']
      state, reward, terminal, info = env.step(model_out['actions'][0])
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

    # selection
    chosen_leaf = self.select_leaf(node)

    # expansion
    obs, terminal = next(chosen_leaf)
    if terminal:
      # repeat
      if len(chosen_leaf.parent.children) > 1:
        del chosen_leaf.parent.children[chosen_leaf.action]
        return True, states
      else:
        return False, states

    model_out = model.step([obs], states)
    self.expand(chosen_leaf, model_out['values'][0])

    # evaluation
    value = self.simulation(
        model, copy.deepcopy(chosen_leaf.env),
        chosen_leaf.rollout.observations[-1],
        chosen_leaf.rollout.rewards[-1],
        states,
        self.max_rollout_depth)

    # backup
    self.backup_value(chosen_leaf, value)
    return True, model_out['states']


class TreeRoller(Roller):
  def __init__(self, env, model, tree, min_episodes=1):
    self.env = env
    self.model = model
    self.tree = tree
    self.min_episodes = 1

  def rollouts(self):
    episodes = []

    while len(episodes) < self.min_episodes:
      states = self.model.start_state(1)
      rollout = empty_rollout(states)
      obs = self.env.reset()

      while True:
        rollout.observations.append(obs)
        selected_action, states = self.tree.search(
            self.model, copy.deepcopy(self.env), obs, states, rollout)
        obs, rew, done, info = self.env.step(index_action(self.env, selected_action))
        rollout.rewards.append(rew)
        rollout.infos.append(info)
        if done:
          break

      rollout.end_time = time.time()
      episodes.append(rollout)
    return episodes


import gym
from anyrl.models import Model

class RandomModel(Model):
  def stateful(self):
    return False

  def start_state(self, idx):
    return np.zeros((4))

  def step(self, obs, states):
    values = np.random.dirichlet(np.ones(2), size=1)
    return {
      'actions': [np.argmax(values)],
      'values': values,
      'states': None,
    }


roller = TreeRoller(
    gym.make('CartPole-v0'),
    RandomModel(),
    MonteCarloTreeSearch(
        search_time_secs=.5, uct_c=5, max_rollout_depth=100),
    min_episodes=5)
actual = roller.rollouts()
