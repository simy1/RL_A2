import numpy as np
import matplotlib.pyplot as plt
# import gymnasium as gym
import gym
import tensorflow as tf
import csv
import time
from tqdm import tqdm
import sys

# import main
import helper
import main_ER


def initialize_modelA(learning_rate, initialization, activation_func):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(24, activation=activation_func, input_shape=(4,), kernel_initializer=initialization),
      tf.keras.layers.Dense(12, activation=activation_func, kernel_initializer=initialization),
      tf.keras.layers.Dense(2, activation='linear', kernel_initializer=initialization)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['accuracy'])
    return model


def average_episode_length(episode_lengths):
    episode_lengths = np.array(episode_lengths)
    return np.cumsum(episode_lengths) / np.arange(1, len(episode_lengths)+1)


def plot_episode_length(episode_lengths, experiment_label):
    episode_lengths = np.array(episode_lengths)

    plt.scatter(range(len(episode_lengths)), episode_lengths, color='navy', label='episode length')

    average_ep_length = average_episode_length(episode_lengths)
    plt.plot(range(len(episode_lengths)), average_ep_length, color='chocolate', label='average episode length')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title(f'experiment {experiment_label}')
    plt.tight_layout()
    plt.savefig(experiment_label)
    # plt.show()


def test_hyperparams(num_episodes, activate_TN, activate_ER, learning_rate, initial_epsilon, final_epsilon, decay_constant, experiment_label=0, num_independent_runs=1):
    #env = gym.make('CartPole-v1', render_mode='human')

    colours = ['chocolate', 'slateblue', 'lime', 'orange', 'forestgreen']
    mean_ep_length_last_fifty_runs = 0

    # main DQN model (not target network)
    base_model = main_ER.initialize_model(learning_rate=learning_rate)

    # put it outside of the if because an error occurs "local variable 'target_network' referenced before assignment"
    # which means that we cannot call target in mainER.main without knowing the variable
    target_network = main_ER.initialize_model(learning_rate=learning_rate)
    if activate_TN:
        update_freq_TN = 100  # steps


    plt.figure(figsize=(15, 6))
    for run in range(num_independent_runs):
        print(f'run {run}')
        episode_lengths = main_ER.main(base_model=base_model, target_network=target_network, num_episodes=num_episodes, initial_exploration=initial_epsilon, final_exploration=final_epsilon, decay_constant=decay_constant, activate_TN=activate_TN, activate_ER=activate_ER, learning_rate=learning_rate)

        average_ep_length = average_episode_length(episode_lengths)
        # mean_ep_length_last_fifty_runs += np.mean(average_ep_length[:-50])
        mean_ep_length_last_fifty_runs += np.mean(average_ep_length[:5])
        plt.scatter(range(len(episode_lengths)), episode_lengths, label='episode length', color=colours[run])
        plt.plot(range(len(episode_lengths)), average_ep_length, label='average episode length', color=colours[run])

        # extra code - back up results in dictionaries
        central_path = helper.make_central_directory()
        dqn_version_path = helper.make_DQN_directory(central_path=central_path, activate_TN=activate_TN, activate_ER=activate_ER)
        helper.store_results_to_file(dqn_version_path=dqn_version_path,initial_exploration=initial_epsilon, final_exploration=final_epsilon, decay_constant=decay_constant, learning_rate=learning_rate, experiment_label=experiment_label, episode_lengths=episode_lengths)


    plt.title(f'experiment {experiment_label}')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'figure {experiment_label}')

    return mean_ep_length_last_fifty_runs/num_independent_runs




if __name__ == '__main__':
    # env = gym.make('CartPole-v1', render_mode='human')

    gamma = 1  # discount factor
    initial_epsilon = 1  # 100%
    final_epsilon = 0.01  # 1%
    num_episodes = 500

    # # learning_rates = [0.01, 0.03, 0.1, 0.3]
    # learning_rates = [0.01, 0.1]
    # decay_constants = [0.001, 0.1]
    # loss_functions = [tf.keras.losses.MeanSquaredError(), tf.keras.losses.Huber()]
    # # kernel_initialization = ['glorot_uniform', 'random_normal', tf.keras.initializers.HeUniform(), tf.keras.initializers.HeNormal()]
    # kernel_initialization = ['random_normal', tf.keras.initializers.HeUniform(), tf.keras.initializers.HeNormal()]
    # activation_functions = ['relu', 'tanh']
    # # activate_TN_options = [True, False]
    # activate_TN_options = [True]
    # # activate_ER_options = [True, False]
    # activate_ER_options = [True]    # learning_rates = [0.01, 0.03, 0.1, 0.3]


    learning_rates = [0.1, 0.01, 0.001]
    decay_constants = [0.01, 0.1]
    loss_functions = [tf.keras.losses.Huber()]
    kernel_initialization = [tf.keras.initializers.HeUniform()]
    activation_functions = ['relu']
    activate_TN_options = [True]
    activate_ER_options = [False]


    experiment_details = {}

    experiment_number = 0
    for activate_TN in activate_TN_options:
        for activate_ER in activate_ER_options:
            for learning_rate in learning_rates:
                for decay_constant in decay_constants:
                    for activation_func in activation_functions:
                        for initialization in kernel_initialization:
                            experiment_number += 1

                            # store all options in dictionary so we can start from another point later
                            experiment_details[experiment_number] = (learning_rate, decay_constant, activation_func, initialization, activate_TN, activate_ER)


    start = (int)(sys.argv[1])      #1
    end = start+1                   #len(experiment_details)+1
    for experiment_number in tqdm(range(start,end)):
        time.sleep(120)
        print(f'-----Experiment {experiment_number}-----')
        start = time.time()

        with open('experiment_specs.txt', mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=',')

            if experiment_number == 1:
                writer.writerow(['experiment_number', 'mean_ep_length_last_fifty_runs', 'num_episodes', 'activate_TN', 'activate_ER', 'learning_rate', 'decay_constant', 'activation_func', 'initialization'])


            learning_rate, decay_constant, activation_func, initialization, activate_TN, activate_ER = experiment_details[experiment_number]

            try:
                mean_ep_length_last_fifty_runs = test_hyperparams(num_episodes=num_episodes, 
                                                                  activate_TN=activate_TN, 
                                                                  activate_ER=activate_ER, 
                                                                  learning_rate=learning_rate, 
                                                                  initial_epsilon=initial_epsilon, 
                                                                  final_epsilon=final_epsilon, 
                                                                  decay_constant=decay_constant, 
                                                                  experiment_label=experiment_number)

                writer.writerow([experiment_number, np.round(mean_ep_length_last_fifty_runs, 3), num_episodes, activate_TN, activate_ER, learning_rate, decay_constant, activation_func, initialization])
            except Exception as error:
                print('>>>>>>> special error:',error)
                writer.writerow([experiment_number, 'run failed'])
        
        end = time.time()
        print('Total time: {} seconds (experiment_number: {})'.format(end-start, experiment_number))