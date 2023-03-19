# import gymnasium as gym
import gym
import numpy as np
import pygame
import tensorflow as tf
from collections import deque
from helper import *
import time
import random

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
    last_element = -1
    terminated, truncated = replay_buffer[last_element][4], replay_buffer[last_element][5]

    if not activate_ER:                    # for the baseline: just take the last element
        sample_list = [last_element]    
    else:                                  # for the ER: check the conditions and then take a sample
        FREQUENCY_ER = 4 
        MIN_SIZE_BUFFER = 1_000
        BATCH_SIZE = 128

        if len(replay_buffer) % FREQUENCY_ER != 0 and not terminated and not truncated:
            return 
        if len(replay_buffer) < MIN_SIZE_BUFFER:
            return
        
        sample_list = random.sample(range(0,len(replay_buffer)), BATCH_SIZE)

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

    predicted_q_values = model.predict(np.array(observation_list))
    new_predicted_q_values = model.predict(np.array(new_observation_list))
    
    q_bellman_list = list()
    for i in range(len(observation_list)):
        if not terminated_list[i] and not truncated_list[i]:
            q_bellman = predicted_q_values[i] - learning_rate * (predicted_q_values[i] - reward_list[i] - gamma * max(new_predicted_q_values[i]))
        else:
            q_bellman = predicted_q_values[i] - learning_rate * (predicted_q_values[i] - reward_list[i])
        q_bellman[1-action_list[i]] = predicted_q_values[i][1-action_list[i]]
        q_bellman_list.append(q_bellman)
    
    if activate_ER:
        model.fit(x=np.array(observation_list), y=np.array(q_bellman_list), batch_size=BATCH_SIZE)
    else:
        model.fit(x=np.array(observation_list), y=np.array(q_bellman_list))    

    
def trainOriginal(replay_buffer):
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


def main(amount_of_episodes, initial_exploration, final_exploration, decay_constant):
    episode_lengths = []
    replay_buffer = deque(maxlen=10_000)
    current_episode_length = 0
    observation, info = env.reset()

    for episode in range(amount_of_episodes):
        terminated, truncated = False, False
        # annealing, done before the while loop because the first episode equals 0 so it returns the original epsilon back
        exploration_parameter = exponential_anneal(episode, initial_exploration, final_exploration, decay_constant)
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
initial_epsilon = 1 # 100%
final_epsilon = 0.01 # 1%
amount_of_episodes = 300
decay_constant = 0.01 # the amount with which the exploration parameter changes after each episode
activate_ER = True
activate_TN = False

env = gym.make('CartPole-v1', render_mode='human')
model = initialize_model(learning_rate=learning_rate)
main(amount_of_episodes=amount_of_episodes,initial_exploration=initial_epsilon,final_exploration=final_epsilon,decay_constant=decay_constant)

end = time.time()
print('Total time: {} seconds (number of episodes: {})'.format(end-start,amount_of_episodes))
