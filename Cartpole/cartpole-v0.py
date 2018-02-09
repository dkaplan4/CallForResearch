import gym
import sys
import numpy as np
from gym import spaces
import utils_cartpole as util
import tensorflow as tf
from argparse import ArgumentParser


ACTION_CHOICE = '__Not Initialized__'
NUM_ITERATIONS = 10000
ENV_TYPE = 'CartPole-v0'
#BEST_REWARD = 200

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--action-choice', type=str,
                        dest='action_choice', help='random,hill-climb,policy-gradient,neural-network',
                        metavar='ACTION_CHOICE', required=True)
    parser.add_argument('--num-iterations', type=int,
                        dest='num_iterations', help='Number of iterations before termination',
                        metavar='NUM_ITERATIONS', default=NUM_ITERATIONS)
    parser.add_argument('--env-type', type=str,
                        dest='env_type', help='The Gym environment or it to run in',
                        metavar='ENV_TYPE', default=ENV_TYPE)
    return parser


def check_options(opts):
    if opts.action_choice != 'random' and \
       opts.action_choice != 'hill-climb' and \
       opts.action_choice != 'policy-gradient' and \
       opts.action_choice != 'neural-network':
       print('action choice "{}" not valid.'.format(opts.action_choice))
       sys.exit()
    assert opts.num_iterations > 0
    if opts.env_type != 'CartPole-v0':
        print('environment {} not valid.'.format(opts.env_type))

def main():
    parser = build_parser()
    options = parser.parse_args()
    check_options(options)

    #initialize environment
    env = gym.make(options.env_type)
    total_iterations = options.num_iterations

    #deterministic transition model
    if options.action_choice == 'random' or options.action_choice == 'hill-climb':
        print('Random configuration')
        #initialize the return variables
        best_configuration = np.random.rand(4)*2 - 1
        best_reward = -1
        for i in range(options.num_iterations):
            if i % 2000 == 0:
                print('{} out of {} iterations'.format(i,total_iterations))
            if options.action_choice == 'random':
                #randomly initialize parameters
                parameters = np.random.rand(4)*2 - 1 #randomly initialize parameters
            else:
                #add noise to parameters
                parameters = best_configuration + np.random.rand(4)*.01
            #run
            reward = util.run_episode(env,parameters)
            #compare rewards
            if reward > best_reward:
                best_configuration = parameters
                best_reward = reward
        print('Best configuration: {}'.format(best_configuration))
        print('Best reward: {}'.format(best_reward))
        print('Number of iterations: {}'.format(i))

    #stochastic transition model
    elif options.action_choice == 'policy-gradient':
        return 0


if __name__ == '__main__':
    main()


# '''
# In order to implement a policy gradient, we need a policy that can change little
# by little. In practice, this means switching from an absolute limit
# (move left if the totla is <0, otherwise move right) to probabilities.
#
# This changes our agent from a deterministic to a stochastic (random) policy
# Instead of only having one linear combination, we have two: one for each possible
# action. Passing these two values through a softmax function gives the probabilities
# of taking th respective actions, given a set of observations. This also generalizes to
# multiple actions, unlike the threshold we were using before
#
# Since we're going to be computing gradients, it's time to use tensorflow.
# '''
# if run_policy_gradient:
#     '''
#     In order to train this netowrk, we first need to run some episodes
#     to gather data. This is pretty similar to the loop in random-search or hill
#     climbing, except we want to to record transitions for each step, containing
#     what action we took from what state, and what reward we got for it
#     '''
#     # tensorflow operations to compute probabilities for eah action, given a state
#     cumm_reward = 0
#     pl_probs, pl_state = util.policy_grad()
#     obs = env.reset()
#     actions = []
#     transitions = []
#     for _ in range(200):
#         #calculate the policy
#         obs_vector = np.expand_dims(obs,axis=0)
#         probs = sess.run(pl_probs,feed_dict={pl_state: obs_vector})
#         action = 0 if np.random.uniform(0,1) < probs[0][0] else 1
#         #record the transition
#         states.append(obs)
#         actionblank = np.zeros(2)
#         actionblank[action] = 1
#         actions.append(actionblank)
#         #take the action in the environment
#         old_obs = obs
#         obs, reward, done, info = env.step(action)
#         transitions.append((old_obs,action,reward))
#         cumm_reward += reward=
#         if done:
#             break
#
#     '''
#     next, we compute the return of each transition, and update the neural
#     network to reflect this
#     We dont care about the specific action we took from each state, only what
#     the average return for the state over all actions is.
#     '''
#     vl_calculated, vl_state, vl_newvals, vl_optimizer = util.alue_gradient()
#     update_vals = []
#     for index, trans in enumerate(transitions):
#         obs,action,reward = trans
#         #calculate discounted monte-carlo return
#         future_reward = 0
#         future_transitions = len(transitions) - index
#         decrease = 1
#         for index2 in xrange(future_transitions):
#             future_reward += transitions[(index2) + index][2] * decrease
#             decrease = decrease* 0.97
#         update_vals.append(future_reward)
#     update_vals_vector = np.expand_dims(update_vals,axis=1)
#     #run the session
#     sess.run(vl_optimizer,feed_dict={vl_state:states,vl_newvals:update_vals_vector})
#     '''
#     If we let this run for 100+ episodes, the value of each state is represented
#     pretty accurately.
#     The decrease factor puts more of an emphasis on short-term reward than long
#     term reward.
#     This introduces a little biases but can reduce variance by a lot
#     How can we use the newly found values of states in order to update our policy
#     to reflect it? We want to favor actions that return a total reward than the average
#     of that state - this is called the __advantage__.
#     We can plug in the advantage as a scale and update our policy accordingly
#     '''
#     for index, trans in enumerate(transitions):
#         obs,action,reward = trans
#         obs_vector = np.expand_dims(obs,axis=0)
#         curr_val = sess.run(vl_calculated,feed_dict={vl_state:obs_vector})[0][0]
#         advatages.append(future_reward - curr_val)
#     advantages_vector = np.expand_dims(advantages,axis=1)
#     sess.run(pl_optimizer,feed_dict={pl_state:states,pl_advantages:advantages_vector,pl_actions:actions})
