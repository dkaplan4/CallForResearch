#Defines functions used in cartpole
import gym
import numpy as np
from gym import spaces
import tensorflow as tf




def run_episode(env,parameters,render_episode=False):
    observation = env.reset() #gets the process started
    #initialize variables
    cumm_reward = 0 #cummulative reward
    for t in range(200): #each time step within each episode
        if render_episode:
            env.render()
        #determine action based on parameters
        if np.matmul(observation,parameters) < 0:
            action = 0 #go left
        else:
            action = 1 #go right
        observation, reward, done, info = env.step(action) #get info about the time step
        cumm_reward += reward
        if done:
            break
    return cumm_reward


def policy_grad():
    params = tf.get_variable('policy_parameters',[4,2])
    state = tf.placeholder('float',[None,4])
    actions = tf.placeholder('float',[None,2]) #go left or go right
    advantages = tf.placeholder('float',[None,1])
    linear = tf.matmul(state,params)
    probs = tf.nn.softmax(linear)
    #we need a way to change the policy, increasing the probabilities of taking
    #a certain action given a certain state
    #Implement an optimizer that allows us to incrementally update our policy
    # Actions is a one-hot vector, with a one at the action we ant to increase the
    # probability of.
    good_probs = tf.reduce_sum(tf.mul(probs,actions),reduction_indecies=[1])
    # Maximize logprob
    logprobs = tf.log(good_probs)
    eligibility = logprobs*advantages
    loss = -tf.reduce_sum(eligibility)
    optimizer = tf.train.AdamOptimizer(.01).minimize(loss)

'''
we are trying to determine the best action for the state
first thing we need is a baseline to compare from
we define some value for each state, that contains the average
return starting from that state

In this example, we are using a 1 hidden layer neural network
'''
def value_grad():
    # sess.run(calculated) to calculate calue of state
    state = tf.placeholder('float',[None,4])
    w1 = tf.get_variable('w1',[4,10])
    b1 = tf.get_variable('b1',[10])
    h1 = tf.nn.relu(tf.matmul(state,w1)+b1)
    w2 = tf.get_variable('w2',[10,1])
    b2 = tf.get_variable('b2',[1])
    calculated = tf.matmul(h1,w2) + b2

    # sess.run(optimizer) to update the value of a state
    newvals = tf.placeholder('float',[None,1])
    diffs = calculated - newvals
    loss = tf.nn.l2_loss(diffs)
    optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
