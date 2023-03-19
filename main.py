import gymnasium as gym
import numpy as np
import pygame
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

# TODO figure out
learning_rate = 10**(-1)
gamma = 1  # discount factor
epsilon = 0.25

env = gym.make('CartPole-v1', render_mode='human')

print(env.observation_space.shape)


def initialize_model(learning_rate=1e-3):
    """

    :return:
    """
    # TODO hyperparameter settings / how many nodes and layers do we need?
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(20, activation='relu', input_shape=env.observation_space.shape, kernel_initializer='random_uniform'),
      tf.keras.layers.Dense(10, activation='relu'),
      tf.keras.layers.Dense(2, activation='linear', kernel_initializer='random_uniform')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['accuracy'])

    return model



main = initialize_model(learning_rate=learning_rate)




episode_lengths = []
current_episode_length = 0

observation, info = env.reset()
predicted_q_values = main.predict(observation.reshape((1, 4)))
for episode in range(2000):

    if episode > 1000:
        epsilon = 0.1  # annealing

    current_episode_length += 1
    env.render()

    # let the main model predict the Q values based on the observation of the environment state
    # these are Q(S_t)
    if np.random.random() < epsilon:
        print('random')
        # exploration
        action = np.random.randint(0, 2)
    else:
        action = np.argmax(predicted_q_values)  # take action with highest associated Q value
    print(f'predicted Q values {predicted_q_values}')
    print(f'Chosen action: {action}')

    new_observation, reward, terminated, truncated, info = env.step(action)

    # these are Q(S_t+1)
    new_predicted_q_values = main.predict(new_observation.reshape((1, 4)))

    # update model weights
    q_bellman = predicted_q_values - learning_rate * (predicted_q_values - reward - gamma * max(new_predicted_q_values))
    q_bellman[not action] = predicted_q_values[not action]  # TODO CHECK
    main.fit(x=observation.reshape(1, 4), y=q_bellman)

    # roll over
    observation = new_observation
    predicted_q_values = new_predicted_q_values

    if terminated or truncated:
        print('it fell')
        print(f'reward = {reward}')
        sys.exit()
        episode_lengths.append(current_episode_length)
        current_episode_length = 0
        observation, info = env.reset()



env.close()

print('episode lengths: ', episode_lengths)

plt.title('Episode lengths over time')
plt.scatter(range(episode_lengths), episode_lengths)
plt.xlabel('episode')
plt.ylabel('length')
plt.show()
