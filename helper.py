import numpy as np
import sys
import os
import pickle

def ckeckCMD(env):
    '''
    Ckeck the command line from the terminal for validity and print the corresponding message (if needed).
    '''
    if env == 0:    # cartpole
        acceptedTokens = ['dqn.py','--experience_replay','--target_network']
        if len(sys.argv)<1 or len(sys.argv)>3:
            return 'error_length'
        for term in sys.argv:
            if term not in acceptedTokens:
                return 'error_term'
            
        if len(sys.argv) == 1:
            return 'baseline'
        elif '--experience_replay' in sys.argv and '--target_network' not in sys.argv:
            return 'ER'
        elif '--target_network' in sys.argv and '--experience_replay' not in sys.argv:
            return 'TN'
        elif '--experience_replay' in sys.argv and '--target_network' in sys.argv:
            return 'ER_TN'
    else:       # acrobot
        acceptedTokens = ['acrobot.py','--experience_replay','--target_network']
        if len(sys.argv)<1 or len(sys.argv)>3:
            return 'error_length'
        for term in sys.argv:
            if term not in acceptedTokens:
                return 'error_term'
            
        if len(sys.argv) == 1:
            return 'baseline'
        elif '--experience_replay' in sys.argv and '--target_network' not in sys.argv:
            return 'ER'
        elif '--target_network' in sys.argv and '--experience_replay' not in sys.argv:
            return 'TN'
        elif '--experience_replay' in sys.argv and '--target_network' in sys.argv:
            return 'ER_TN'


def printNotAcceptedCMD(env,error_msg):
    '''
    Print error message in case checkCMD() function returns error code.
    param error_msg:    the error message from the given error command line 
    '''
    if env == 0:
        # cartpole
        if error_msg == 'error_length':
            print('The command was not accepted due to the following reason: too many/few arguments')
        if error_msg == 'error_term':
            print('The command was not accepted due to the following reason: wrong argument(s) given')
        print('Please follow one of the examples listed below:')
        print('$ python dqn.py')
        print('$ python dqn.py --experience_replay')
        print('$ python dqn.py --target_network')
        print('$ python dqn.py --experience_replay --target_network')
    else:
        # acrobot
        if error_msg == 'error_length':
            print('The command was not accepted due to the following reason: too many/few arguments')
        if error_msg == 'error_term':
            print('The command was not accepted due to the following reason: wrong argument(s) given')
        print('Please follow one of the examples listed below:')
        print('$ python acrobot.py')
        print('$ python acrobot.py --experience_replay')
        print('$ python acrobot.py --target_network')
        print('$ python acrobot.py --experience_replay --target_network')


def exponential_anneal(t, start, final, decay_constant):
    ''' 
    Exponential annealing scheduler for epsilon-greedy policy.
    param t:        current timestep
    param start:    initial value
    param final:    value after percentage*T steps
    '''
    return final + (start - final) * np.exp(-decay_constant*t)


def boltzmann_exploration(actions, temperature):
    '''
    Boltzmann exploration policy.
    param actions:      vector with possible actions
    param temperature:  exploration parameter
    return:             vector with probabilities for choosing each option
    '''
    # print(f'bolzmann exploration of {actions}')  # can remove this line once everything works
    actions = actions[0] / temperature  # scale by temperature
    a = actions - max(actions)  # substract maximum to prevent overflow of softmax
    return np.exp(a)/np.sum(np.exp(a))


def make_central_directory(target):
    '''
    Make the central directory for storing all the results.
    param target:   name of the central directory
    '''
    file_dir = os.path.dirname(__file__)
    central_path = file_dir + target
    try:
        os.mkdir(central_path)
    except OSError as error:
        print(error)
    return central_path


def make_DQN_directory(central_path, activate_TN, activate_ER, exploration='annealing_epsilon_greedy'):
    '''
    Make the directory for the corresponding DQN version you want to run.
    param central_path:     the name of central directory
    param activate_TN:      True of False whether we use a DQN version with a Target Network
    param activate_ER:      True of False whether we use a DQN version with an Experience Replay Buffer
    param exploration:      Defines the policy of experiments 
    '''
    dqn_version_path = central_path + '/DQN'
    if activate_TN == True:
        dqn_version_path += '-TN'
    if activate_ER == True:
        dqn_version_path += '-ER'
    if exploration == 'boltzmann':
        dqn_version_path += '-boltzmann'
    try:
        os.mkdir(dqn_version_path)
    except OSError as error:
        print(error)
    return dqn_version_path


def store_results_to_file(dqn_version_path,initial_exploration, final_exploration, decay_constant, learning_rate, experiment_label, episode_lengths, repetition):
    '''
    Make a dictionary for the results obtained and store them to the file.
    param dqn_version_path:     the path to store the results (combines both the central directory and the DQN version)
    param initial_exploration:  number of initial exploration
    param final_exploration:    number of final exploration
    param decay_constant:       decay constant selected
    param learning_rate:        learning rate chosen
    param experiment_label:     number that represents the combination of the above hyperparameters  
    param episode_lengths:      number for episode length
    param repetition:           number for repetition of the combination
    '''
    run_details_dict = dict()
    key = (('initial_exploration',initial_exploration),('final_exploration',final_exploration),('decay_constant',decay_constant),('learning_rate',learning_rate))
    run_details_dict[key] = episode_lengths

    # store the dictionary to a pickle file
    name = dqn_version_path + '/combination_' + str(experiment_label) + '__repetition_' + str(repetition) + '.pkl'
    a_file = open(name, "wb")
    pickle.dump(run_details_dict,a_file,pickle.HIGHEST_PROTOCOL)
    a_file.close()


def store_results_to_file_boltzmann(dqn_version_path, temperature, learning_rate, experiment_label, episode_lengths, repetition):
    '''
    Make a dictionary for the results obtained and store them to the file.
    param dqn_version_path:     the path to store the results (combines both the central directory and the DQN version)
    param temperature:          boltazmann's key parameter  
    param learning_rate:        learning rate chosen
    param experiment_label:     number that represents the combination of the above hyperparameters  
    param episode_lengths:      number for episode length
    param repetition:           number for repetition of the combination
    '''
    run_details_dict = dict()
    key = (('learning_rate', learning_rate), ('temperature', temperature))
    run_details_dict[key] = episode_lengths

    # store the dictionary to a pickle file
    name = dqn_version_path + '/combination_' + str(experiment_label) + '__repetition_' + str(repetition) + '.pkl'
    a_file = open(name, "wb")
    pickle.dump(run_details_dict, a_file, pickle.HIGHEST_PROTOCOL)
    a_file.close()


def store_results_to_file_v2(dqn_version_path, loss_init, neurons_layers, experiment_label, episode_lengths, repetition):
    '''
    Make a dictionary for the results obtained and store them to the file.
    param dqn_version_path:     the path to store the results (combines both the central directory and the DQN version)
    param loss_init:            list with loss function and initialization function used 
    param neurons_layers:       list with number of neurons in each hidden layer
    param experiment_label:     number that represents the combination of the above hyperparameters  
    param episode_lengths:      number for episode length
    param repetition:           number for repetition of the combination
    '''
    run_details_dict = dict()
    key = (str)((('loss_init',loss_init),('neurons_layers',neurons_layers)))
    run_details_dict[key] = episode_lengths

    # store the dictionary to a pickle file
    name = dqn_version_path + '/combination_' + str(experiment_label) + '__repetition_' + str(repetition) + '.pkl'
    a_file = open(name, "wb")
    pickle.dump(run_details_dict,a_file,pickle.HIGHEST_PROTOCOL)
    a_file.close()