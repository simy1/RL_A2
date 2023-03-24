# import gymnasium as gym
import gym
import numpy as np
import pygame
import tensorflow as tf
from collections import deque


import visualize
from helper import *
import time
import random
import matplotlib.pyplot as plt
from visualize import plot_episode_length, average_episode_length


# next two lines can be removed after hyperparameter tuning
update_freq_TN = 100
gamma = 1

def initialize_model(learning_rate):
    # TODO hyperparameter settings / how many nodes and layers do we need?
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(24, activation='relu', input_shape=(4,), kernel_initializer='random_uniform'),
      tf.keras.layers.Dense(12, activation='relu', kernel_initializer='random_uniform'),
      tf.keras.layers.Dense(2, activation='linear', kernel_initializer='random_uniform')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['accuracy'])

    return model


def update_model(base_model, target_network):
    """
    Copies weights from base model to target network
    :param base_model: tf base model
    :param target_network: tf target network
    :return:
    """
    for layer_TN, layer_BM in zip(target_network.layers, base_model.layers):
        layer_TN.set_weights(layer_BM.get_weights())


def train(base_model, target_network, replay_buffer, activate_ER, activate_TN, learning_rate):
    last_element = -1
    terminated, truncated = replay_buffer[last_element][4], replay_buffer[last_element][5]

    if not activate_ER:                    # for the baseline: just take the last element
        sample_list = [last_element]    
    else:                                  # for the ER: check the conditions and then take a sample
        MIN_SIZE_BUFFER = 1_000
        BATCH_SIZE = 128

        # if len(replay_buffer) % FREQUENCY_ER != 0 and not terminated and not truncated:
        #     return
        if len(replay_buffer) < MIN_SIZE_BUFFER:
            return
        
        sample_list = random.sample(range(0, len(replay_buffer)), BATCH_SIZE)

    observation_list = list()
    new_observation_list = list()
    action_list = list()
    reward_list = list()
    terminated_list = list()
    truncated_list = list()
    for element in sample_list:
        observation_list.append(replay_buffer[element][0])
        new_observation_list.append(replay_buffer[element][3])
        action_list.append(replay_buffer[element][1])
        reward_list.append(replay_buffer[element][2])
        terminated_list.append(replay_buffer[element][4])
        truncated_list.append(replay_buffer[element][5])

    predicted_q_values = base_model.predict(np.array(observation_list))
    if activate_TN:
        new_predicted_q_values = target_network.predict(np.array(new_observation_list))
    else:
        new_predicted_q_values = base_model.predict(np.array(new_observation_list))

    q_bellman_list = list()
    for i in range(len(observation_list)):
        if not terminated_list[i] and not truncated_list[i]:
            q_bellman = predicted_q_values[i] - learning_rate * (predicted_q_values[i] - reward_list[i] - gamma * max(new_predicted_q_values[i]))
        else:
            q_bellman = predicted_q_values[i] - learning_rate * (predicted_q_values[i] - reward_list[i])
        q_bellman[1-action_list[i]] = predicted_q_values[i][1-action_list[i]]
        q_bellman_list.append(q_bellman)
    
    if activate_ER:
        base_model.fit(x=np.array(observation_list), y=np.array(q_bellman_list), batch_size=BATCH_SIZE, verbose=0)
    else:
        base_model.fit(x=np.array(observation_list), y=np.array(q_bellman_list), verbose=0)


def main(base_model, target_network, num_episodes, initial_exploration, final_exploration, decay_constant, activate_TN, activate_ER, learning_rate):
    env = gym.make('CartPole-v1') #, render_mode='human')  # TODO uncomment if you want to see the cartpole!!

    episode_lengths = []
    replay_buffer = deque(maxlen=10_000)
    current_episode_length = 0

    if activate_TN:
        # start by copying over the weights from TN to base model to ensure they are identical
        update_model(base_model=base_model, target_network=target_network)
        steps_TN = 0

    observation, info = env.reset()

    for episode in range(num_episodes):
        terminated, truncated = False, False
        # annealing, done before the while loop because the first episode equals 0 so it returns the original epsilon back
        exploration_parameter = exponential_anneal(episode, initial_exploration, final_exploration, decay_constant)
        epsilon = exploration_parameter  # temporary while only using egreedy
        while not terminated and not truncated:
            current_episode_length += 1


            # env.render()  # uncomment after hyperparameter tuning


            # let the main model predict the Q values based on the observation of the environment state
            # these are Q(S_t)
            predicted_q_values = target_network.predict(observation.reshape((1, 4)))

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
            replay_buffer.append([observation, action, reward, new_observation, terminated, truncated])

            if activate_TN:
                steps_TN += 1
                if current_episode_length % 4 == 0 or truncated or terminated:
                    train(base_model=base_model, target_network=target_network, replay_buffer=replay_buffer, activate_ER=activate_ER, activate_TN=activate_TN, learning_rate=learning_rate)
            else:
                train(base_model=base_model, target_network=target_network, replay_buffer=replay_buffer, activate_ER=activate_ER, activate_TN=activate_TN, learning_rate=learning_rate)

            # roll over
            observation = new_observation

            if terminated or truncated:
                episode_lengths.append(current_episode_length)
                current_episode_length = 0
                observation, info = env.reset()

                if activate_TN:
                    if steps_TN >= update_freq_TN:
                        update_model(base_model=base_model, target_network=target_network)  # copy over the weights
                        steps_TN = 0

    # for episode length visualization
    # print('episode lengths: ', episode_lengths)
    env.close()

    return episode_lengths






if __name__ == '__main__':
    start = time.time()

    # response = ckeckCMD()
    # if 'error' in response:
    #     printNotAcceptedCMD(response)
    #     exit()
    # activate_ER = False
    # activate_TN = False
    # if 'ER' in response:
    #     activate_ER = True
    # if 'TN' in response:
    #     activate_TN = True

    learning_rate = 10**(-1)
    gamma = 1  # discount factor
    initial_epsilon = 1  # 100%
    final_epsilon = 0.01  # 1%
    num_episodes = 500
    decay_constant = 0.01  # the amount with which the exploration parameter changes after each episode
    activate_ER = True
    activate_TN = True

    # env = gym.make('CartPole-v1', render_mode='human')

    # main DQN model (not target network)
    base_model = initialize_model(learning_rate=learning_rate)

    if activate_TN:
        target_network = initialize_model(learning_rate=learning_rate)
        update_freq_TN = 100  # steps




    end = time.time()
    print('Total time: {} seconds (number of episodes: {})'.format(end-start, num_episodes))






