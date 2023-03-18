import gymnasium as gym
import numpy as np
import pygame
import tensorflow as tf
import sys
from collections import deque
from helper import *


def initialize_model(learning_rate):
    """

    :return:
    """
    # TODO hyperparameter settings / how many nodes and layers do we need?
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(24, activation='relu', input_shape=env.observation_space.shape, kernel_initializer='random_uniform'),
      tf.keras.layers.Dense(12, activation='relu', kernel_initializer='random_uniform'),
      tf.keras.layers.Dense(2, activation='linear', kernel_initializer='random_uniform')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['accuracy'])

    return model


def train(replay_buffer):
    # take the last saved episode with [-1]
    observation = replay_buffer[-1][0]
    new_observation = replay_buffer[-1][3]

    predicted_q_values = model.predict(observation.reshape((1, 4))) # Q(S_t)
    new_predicted_q_values = model.predict(new_observation.reshape((1, 4))) # Q(S_t+1)

    action = replay_buffer[-1][1]
    reward = replay_buffer[-1][2]
    terminated, truncated = replay_buffer[-1][4], replay_buffer[-1][5]

    # update model weights
    if not terminated and not truncated:
        q_bellman = predicted_q_values - learning_rate * (predicted_q_values - reward - gamma * max(new_predicted_q_values))
    else:
        q_bellman = predicted_q_values - learning_rate * (predicted_q_values - reward)

    q_bellman[0][1-action] = predicted_q_values[0][1-action] # [0] comes from the fact that these two are arrays

    model.fit(x=observation.reshape(1, 4), y=q_bellman)


def main(amount_of_episodes, initial_exploration, final_exploration, percentage, decay_constant):
    episode_lengths = []
    replay_buffer = deque(maxlen=10000)
    current_episode_length = 0
    observation, info = env.reset()

    for episode in range(amount_of_episodes):
        terminated, truncated = False, False
        # annealing, done before the while loop because the first episode equals 0 so it returns the original epsilon back
        exploration_parameter = exponential_anneal(episode, amount_of_episodes, initial_exploration, final_exploration,
                                                   percentage, decay_constant)
        epsilon = exploration_parameter  # temporary while only using egreedy
        while not terminated and not truncated:
            current_episode_length += 1
            env.render()

            # let the main model predict the Q values based on the observation of the environment state
            # these are Q(S_t)
            predicted_q_values = model.predict(observation.reshape((1, 4)))

            # choose an action
            if np.random.random() < epsilon:
                # exploration
                action = np.random.randint(0, 2)
            else:
                # exploitation
                action = np.argmax(predicted_q_values)  # take action with highest associated Q value

            # for testing:
            # print(f'predicted Q values {predicted_q_values}')
            # print(f'Chosen action: {action}')

            new_observation, reward, terminated, truncated, info = env.step(action)
            replay_buffer.append([observation,action,reward,new_observation,terminated,truncated])

            train(replay_buffer)

            # roll over
            observation = new_observation

            if terminated or truncated:
                episode_lengths.append(current_episode_length)
                current_episode_length = 0
                observation, info = env.reset()

    # for episode length visualization
    print('episode lengths: ', episode_lengths)
    env.close()

learning_rate = 10**(-1)
gamma = 1  # discount factor
initial_epsilon = 1 # 100%
final_epsilon = 0.01 # 1%
amount_of_episodes = 10
percentage = 1 # percentage after which the annealing should finish
decay_constant = 0.01 # the amount with which the exploration parameter changes after each episode

env = gym.make('CartPole-v1', render_mode='human')

model = initialize_model(learning_rate=learning_rate)
main(amount_of_episodes=amount_of_episodes,initial_exploration=initial_epsilon,final_exploration=final_epsilon,percentage=percentage,decay_constant=decay_constant)