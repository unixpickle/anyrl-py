"""
Use A2C to train a model on the CartPole-v0 environment in
OpenAI Gym.
"""

from anyrl.algos import A2C
from anyrl.distributions import gym_space_distribution
from anyrl.models import MLP, space_vectorizer
from anyrl.rollouts import BasicRoller, mean_total_reward
import gym
import tensorflow as tf

def main():
    """
    Entry-point for the program.
    """
    env = gym.make('CartPole-v0')
    action_dist = gym_space_distribution(env.action_space)
    obs_vectorizer = space_vectorizer(env.observation_space)

    with tf.Session() as sess:
        model = MLP(sess, action_dist, obs_vectorizer, layer_sizes=[32])

        # Deal with CartPole-v0 reward scale.
        model.scale_outputs(20)

        roller = BasicRoller(env, model, min_episodes=30)
        a2c = A2C(model)
        optimize = a2c.optimize(learning_rate=1e-2, max_grad_norm=None,
                                rms_decay=0.9)
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            rollouts = roller.rollouts()
            feed_dict = a2c.feed_dict(rollouts)
            nodes = (optimize, a2c.critic_loss, a2c.regularization)
            _, critic_loss, regularization = sess.run(nodes, feed_dict)
            print('batch %d: mean=%f reg=%f critic_mse=%f' %
                  (i, mean_total_reward(rollouts), regularization, critic_loss))

if __name__ == '__main__':
    main()
