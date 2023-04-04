# import gymnasium as gym
import gym
import numpy as np
import pygame
import tensorflow as tf
from collections import deque
from tqdm import tqdm
import time
import random
import matplotlib.pyplot as plt

from helper import *

update_freq_TN = 100
gamma = 1


def initialize_model(learning_rate):
    '''
    Build the model. It is a simple Neural Network consisting of 3 densely connected layers with Relu activation functions.
    The only argument is the learning rate.
    '''
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(6,),
                              kernel_initializer=tf.keras.initializers.GlorotUniform()),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform()),
        tf.keras.layers.Dense(3, activation='linear', kernel_initializer=tf.keras.initializers.GlorotUniform())
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    return model


def update_model(base_model, target_network):
    '''
    Copies weights from base model to target network.
    param base_model:       tf base model
    param target_network:   tf target network
    '''
    for layer_TN, layer_BM in zip(target_network.layers, base_model.layers):
        layer_TN.set_weights(layer_BM.get_weights())


def train(base_model, target_network, replay_buffer, activate_ER, activate_TN, learning_rate):
    '''
    Trains the model using the DQN algorithm. 
    The Replay Experience buffer (if enabled) is used to indicate which states we want to train the model on. 
    Otherwise, we use the last state observed in the list.
    Then, it predicts the new Q-values with the use of the Target Network (if enabled).
    Finally it fits the model (using a batch size if Replay Experience buffer is enabled).
    param base_model:       the constructed Model
    param target_network:   Target Network
    param replay_buffer:    Experience Replay buffer for storing states.
    param activate_ER:      True of False whether an Experience Replay Buffer is used 
    param activate_TN:      True of False whether a Target Network is used 
    param learning_rate:    learning rate hyperparameter
    '''    
    last_element = -1  # index for the last element of the buffer for if we update without batches
    terminated, truncated = replay_buffer[last_element][4], replay_buffer[last_element][5]

    if not activate_ER:  # for the baseline: just take the last element
        sample_list = [last_element]
    else:  # for the ER: check the conditions and then take a sample
        min_size_buffer = 1_000
        batch_size = 128

        if len(replay_buffer) < min_size_buffer:
            return

        sample_list = random.sample(range(0, len(replay_buffer)), batch_size)

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

    predicted_q_values = base_model.predict(np.array(observation_list), verbose=0)
    if activate_TN:
        new_predicted_q_values = target_network.predict(np.array(new_observation_list), verbose=0)
    else:
        new_predicted_q_values = base_model.predict(np.array(new_observation_list), verbose=0)

    q_bellman_list = list()
    for i in range(len(observation_list)):
        if (not terminated_list[i]) and (not truncated_list[i]):
            q_bellman = predicted_q_values[i] - learning_rate * (
                        predicted_q_values[i] - reward_list[i] - gamma * max(new_predicted_q_values[i]))
        else:
            q_bellman = predicted_q_values[i] - learning_rate * (predicted_q_values[i] - reward_list[i])
        for action in range(3):
            if action != action_list[i]:
                q_bellman[action] = predicted_q_values[i][action]
        q_bellman_list.append(q_bellman)

    if activate_ER:
        base_model.fit(x=np.array(observation_list), y=np.array(q_bellman_list), batch_size=batch_size, verbose=0)
    else:
        base_model.fit(x=np.array(observation_list), y=np.array(q_bellman_list), verbose=0)


def main(base_model, target_network, num_episodes, initial_exploration, final_exploration, decay_constant, activate_TN,
         activate_ER, learning_rate):
    '''
    For all the episodes, the agent selects an action (based on the given policy) and then trains the model.
    Experience Replacy and Target Network are used if they were specified when this function was called.
    same parameters as the train function: base_model, target_network, learning_rate, activate_TN, activate_ER
    param num_episodes:             integet number specifying the number of episodes
    param initial_exploration:      upper limit of epsilon value for annealing epsilon greedy
    param final_exploration:        lower limit of epsilon value for annealing epsilon greedy
    param decay_constant:           decreasing value for annealing epsilon greedy
    param temperature:              key parameter of boltzmann's policy
    param exploration_strategy:     by default is set to 'anneal_epsilon_greedy' but 'boltzmann' is also a valid option     
    '''
    env = gym.make('Acrobot-v1') # to see the acrobot, put inside gym.make this --> render_mode='human' <--

    episode_lengths = []
    replay_buffer = deque(maxlen=50_000)
    current_episode_length = 0

    if activate_TN:
        update_model(base_model=base_model, target_network=target_network)
        steps_TN = 0

    observation, info = env.reset()

    for episode in tqdm(range(num_episodes)):
        terminated, truncated = False, False
        # annealing, done before the while loop because the first episode equals 0 so it returns the original epsilon back
        exploration_parameter = exponential_anneal(episode, initial_exploration, final_exploration, decay_constant)
        epsilon = exploration_parameter  # temporary while only using egreedy
        while (not terminated) and (not truncated):
            current_episode_length += 1

            # env.render()  # uncomment after hyperparameter tuning

            # let the main model predict the Q values based on the observation of the environment state
            # these are Q(S_t)
            predicted_q_values = base_model.predict(observation.reshape((1, 6)), verbose=0)

            # choose an action
            if np.random.random() < epsilon:        # exploration
                action = np.random.randint(3)
            else:
                action = np.argmax(predicted_q_values)  # exploitation: take action with highest associated Q value

            # for testing:
            # print(f'predicted Q values {predicted_q_values}')
            # print(f'Chosen action: {action}')

            new_observation, reward, terminated, truncated, info = env.step(action)
            replay_buffer.append([observation, action, reward, new_observation, terminated, truncated])

            if activate_TN:
                steps_TN += 1
                if current_episode_length % 4 == 0 or truncated or terminated:
                    train(base_model=base_model, target_network=target_network, replay_buffer=replay_buffer,
                          activate_ER=activate_ER, activate_TN=activate_TN, learning_rate=learning_rate)
            else:
                train(base_model=base_model, target_network=target_network, replay_buffer=replay_buffer,
                      activate_ER=activate_ER, activate_TN=activate_TN, learning_rate=learning_rate)

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
                break

    plt.figure()
    plt.plot(range(len(episode_lengths)),episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.show()

    env.close()

    return episode_lengths


if __name__ == '__main__':
    ###########################################################################################################
    # Hyperparameters of the algorithm and other parameters of the program
    learning_rate = 0.01
    gamma = 1  # discount factor
    initial_epsilon = 0.5  # 50%
    final_epsilon = 0.01  # 1%
    num_episodes = 300
    decay_constant = 0.1  # the amount with which the exploration parameter changes after each episode
    activate_ER = True
    activate_TN = True
    ###########################################################################################################

    start = time.time()
    code=1
    response = ckeckCMD(code)
    if 'error' in response:
        printNotAcceptedCMD(code,response)
        exit()
    activate_ER = False
    activate_TN = False
    if 'ER' in response:
        activate_ER = True
    if 'TN' in response:
        activate_TN = True
    
    base_model = initialize_model(learning_rate=learning_rate)
    target_network = initialize_model(learning_rate=learning_rate)
    if activate_TN:
        update_freq_TN = 100  # steps

    returned_list = main(base_model, target_network, num_episodes, initial_epsilon, final_epsilon, decay_constant, activate_TN,
         activate_ER, learning_rate)
    
    file_handler = open('acrobot_results.txt','w')
    file_handler.writelines(str(returned_list))
    file_handler.close()

    end = time.time()
    print('Total time: {} seconds (number of episodes: {})'.format(end - start, num_episodes))