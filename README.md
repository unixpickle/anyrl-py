# anyrl-py

This will be a remake of [anyrl](https://github.com/unixpickle/anyrl) in Python. It will likely support TensorFlow, and perhaps other frameworks as well.

# Motivation

Currently, most RL code out there is very restricted and not properly decoupled. The way things work in anyrl is fairly modular and flexible. The goal is to decouple agents, learning algorithms, trajectories, and things like GAE.

For example, anyrl decouples rollouts from the learning algorithm when possible. This way, you can gather a batch of rollouts and then pass it to TRPO, PPO, vanilla PG, etc. This makes it possible to gather rollouts in several different ways: with fixed-length trajectories, with full-length episodes, with certain wallclock constraints, etc. Further, and more obviously, it means that you don't have to rewrite rollout code for every RL algorithm. However, algorithms like A3C and Evolution Strategies may have specific ways of performing rollouts.
