import numpy as np
import matplotlib.pyplot as plt
# import gymnasium as gym
import gym
import tensorflow as tf
import time
from tqdm import tqdm
import sys
import os

import helper
import dqn
import visualize


def initialize_modelA(learning_rate, loss_init, neuron_layers):
    '''
    Build the model. It is a simple Neural Network consisting of 3 densely connected layers with Relu activation functions.
    Tune four parameters of the model (learning rate, loss function, initialization function and neurons in each layer).
    '''
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(neuron_layers[0], activation='relu', input_shape=(4,), kernel_initializer=loss_init[1]),
      tf.keras.layers.Dense(neuron_layers[1], activation='relu', kernel_initializer=loss_init[1]),
      tf.keras.layers.Dense(2, activation='linear', kernel_initializer=loss_init[1])
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss_init[0],
                  metrics=['accuracy'])
    return model


def test_hyperparams(num_episodes, activate_TN, activate_ER, learning_rate, initial_epsilon, final_epsilon, decay_constant, temperature, experiment_label=0, repetition=1, neurons_layers=[24,12], loss_init=[tf.keras.losses.Huber(),tf.keras.initializers.HeUniform()], exploration_strategy='epsilon_greedy_annealing'):
    '''
    Takes a certain combination of hyperparameters and run the model by calling the mainDQN.main.
    Finally, it stores the results for the specific repetition at a certain directory (actually two directories are used). 
    '''
    colours = ['chocolate', 'slateblue', 'lime', 'orange', 'forestgreen']
    mean_ep_length_last_fifty_runs = 0

    base_model = initialize_modelA(learning_rate=learning_rate, loss_init=loss_init, neuron_layers=neurons_layers)

    target_network = initialize_modelA(learning_rate=learning_rate, loss_init=loss_init, neuron_layers=neurons_layers)
    if activate_TN:
        update_freq_TN = 100  # steps

    print(f'run {repetition}')
    episode_lengths = dqn.main(base_model=base_model, target_network=target_network, num_episodes=num_episodes, initial_exploration=initial_epsilon, final_exploration=final_epsilon, learning_rate=learning_rate, decay_constant=decay_constant, temperature=temperature, activate_TN=activate_TN, activate_ER=activate_ER, exploration_strategy='anneal_epsilon_greedy')

    average_ep_length = visualize.average_episode_length(episode_lengths)
    mean_ep_length_last_fifty_runs += np.mean(average_ep_length[:5])

    # back up results in dictionaries
    if exploration_strategy == 'epsilon_greedy_annealing':
        central_path = helper.make_central_directory('/hyperparams-tuning_annealing')
        dqn_version_path = helper.make_DQN_directory(central_path=central_path, activate_TN=activate_TN, activate_ER=activate_ER)
        helper.store_results_to_file(dqn_version_path=dqn_version_path,initial_exploration=initial_epsilon, final_exploration=final_epsilon, decay_constant=decay_constant, learning_rate=learning_rate, experiment_label=experiment_label, episode_lengths=episode_lengths, repetition=repetition)
        
        central_path = helper.make_central_directory('/hyperparams-tuning_architecture')
        dqn_version_path = helper.make_DQN_directory(central_path=central_path, activate_TN=activate_TN, activate_ER=activate_ER)
        helper.store_results_to_file_v2(dqn_version_path=dqn_version_path, loss_init=loss_init, neurons_layers=neurons_layers, experiment_label=experiment_label, episode_lengths=episode_lengths, repetition=repetition)
    else:
        central_path = helper.make_central_directory('/hyperparams-tuning_boltzmann')
        dqn_version_path = helper.make_DQN_directory(central_path=central_path, activate_TN=activate_TN, activate_ER=activate_ER)
        helper.store_results_to_file_boltzmann(dqn_version_path=dqn_version_path, temperature=temperature, learning_rate=learning_rate, experiment_label=experiment_label, episode_lengths=episode_lengths, repetition=repetition)
        
        central_path = helper.make_central_directory('/hyperparams-tuning_architecture')
        dqn_version_path = helper.make_DQN_directory(central_path=central_path, activate_TN=activate_TN, activate_ER=activate_ER)
        helper.store_results_to_file_v2(dqn_version_path=dqn_version_path, loss_init=loss_init, neurons_layers=neurons_layers, experiment_label=experiment_label, episode_lengths=episode_lengths, repetition=repetition)

    return mean_ep_length_last_fifty_runs


if __name__ == '__main__':
    gamma = 1  # discount factor
    initial_epsilon = 1  # 100%
    final_epsilon = 0.01  # 1%
    num_episodes = 300
    total_repetitions = 3

    learning_rates = [0.1, 0.01, 0.001]
    decay_constants = [0.1, 0.01, 0.001, 0.99]
    temperatures = [10**-1, 10**0, 10**1]
    exploration_strategy = 'epsilon_greedy_annealing' # or 'boltzmann'
    loss_init_functions = [[tf.keras.losses.Huber(),tf.keras.initializers.HeUniform()],[tf.keras.losses.MeanSquaredError(),tf.keras.initializers.GlorotUniform()]]
    neurons_list = [[24,12],[64,128],[128,128]]
    activation_functions = ['relu']
    activate_TN_options = [True]
    activate_ER_options = [True]

    experiment_details = {}
    experiment_number = 0

    if exploration_strategy == 'epsilon_greedy_annealing':
        for activate_TN in activate_TN_options:
            for activate_ER in activate_ER_options:
                for learning_rate in learning_rates:
                    for decay_constant in decay_constants:
                        for activation_func in activation_functions:
                            for loss_init in loss_init_functions:
                                for neurons_layers in neurons_list:
                                    experiment_number += 1
                                    experiment_details[experiment_number] = (learning_rate, decay_constant, activation_func, loss_init, activate_TN, activate_ER, neurons_layers)
    else:
        for activate_TN in activate_TN_options:
            for activate_ER in activate_ER_options:
                for learning_rate in learning_rates:
                    for temperature in temperatures:
                        for activation_func in activation_functions:
                            for loss_init in loss_init_functions:
                                for neurons_layers in neurons_list:
                                    experiment_number += 1
                                    experiment_details[experiment_number] = (learning_rate, temperature, activation_func, loss_init, activate_TN, activate_ER, neurons_layers)


    start = 1                                       # (int)(sys.argv[1])
    end = len(experiment_details)+1                 # start+1
    for repetition in range(1,total_repetitions+1):     # number of independent experiment / repetition (int)(sys.argv[2])
        for experiment_number in tqdm(range(start,end)):
            time.sleep(10)
            print(f'-----Experiment {experiment_number}-----')
            start = time.time()

            if exploration_strategy == 'epsilon_greedy_annealing':
                learning_rate, decay_constant, activation_func, loss_init, activate_TN, activate_ER, neurons_layers = experiment_details[experiment_number]
                temperature = 0
            else:
                learning_rate, temperature, activation_func, loss_init, activate_TN, activate_ER, neurons_layers = experiment_details[experiment_number]
                decay_constant = 0

            try:
                mean_ep_length_last_fifty_runs = test_hyperparams(num_episodes=num_episodes, 
                                                                    activate_TN=activate_TN, 
                                                                    activate_ER=activate_ER, 
                                                                    learning_rate=learning_rate, 
                                                                    initial_epsilon=initial_epsilon, 
                                                                    final_epsilon=final_epsilon, 
                                                                    decay_constant=decay_constant,
                                                                    temperature=temperature, 
                                                                    experiment_label=experiment_number,
                                                                    repetition=repetition,
                                                                    neurons_layers = neurons_layers,
                                                                    loss_init=loss_init,
                                                                    exploration_strategy=exploration_strategy)

            except Exception as error:
                print('>>>> special error:',error)
            
            end = time.time()
            print('Total time: {} seconds (experiment_number: {})'.format(end-start, experiment_number))
