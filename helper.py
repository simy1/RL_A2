import numpy as np
import sys
import os
import pickle

def ckeckCMD():
    '''
    Ckeck the command from the terminal for validity and print the corresponding message (if needed).
    '''
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


def printNotAcceptedCMD(error_msg):
    '''
    Print error message in case checkCMD() function returns error code.
    '''
    if error_msg == 'error_length':
        print('The command was not accepted due to the following reason: too many/few arguments')
    if error_msg == 'error_term':
        print('The command was not accepted due to the following reason: wrong argument(s) given')
    print('Please follow one of the examples listed below:')
    print('$ python dqn.py')
    print('$ python dqn.py --experience_replay')
    print('$ python dqn.py --target_network')
    print('$ python dqn.py --experience_replay --target_network')


def exponential_anneal(t, start, final, decay_constant):
    ''' Linear annealing scheduler
    t: current timestep
    start: initial value
    final: value after percentage*T steps'''
    return final + (start - final) * np.exp(-decay_constant*t)


def make_central_directory():
    '''
    Make the central directory for storing all the results.
    '''
    file_dir = os.path.dirname(__file__)
    central_path = file_dir + '/RL-as2-details4runs'
    try:
        os.mkdir(central_path)
    except OSError as error:
        print(error)
    return central_path


def make_DQN_directory(central_path, activate_TN, activate_ER):
    '''
    Make the directory for the corresponding DQN version you want to run.
    '''
    dqn_version_path = central_path + '/DQN'
    if activate_TN == True:
        dqn_version_path += '-TN'
    if activate_ER == True:
        dqn_version_path += '-ER'
    try:
        os.mkdir(dqn_version_path)
    except OSError as error:
        print(error)
    return dqn_version_path


def store_results_to_file(dqn_version_path,initial_exploration, final_exploration, decay_constant, learning_rate, experiment_label, episode_lengths, repetition):
    '''
    Make a dictionary for the results obtained and store them to the file.
    '''
    run_details_dict = dict()
    key = (('initial_exploration',initial_exploration),('final_exploration',final_exploration),('decay_constant',decay_constant),('learning_rate',learning_rate))
    run_details_dict[key] = episode_lengths

    # store the dictionary to a pickle file
    name = dqn_version_path + '/combination_' + str(experiment_label) + '__repetition_' + str(repetition) + '.pkl'
    a_file = open(name, "wb")
    pickle.dump(run_details_dict,a_file,pickle.HIGHEST_PROTOCOL)
    a_file.close()

    # load (when we want to observe the results and make plots)
    # a_file = open(name, "rb")
    # x_dict = pickle.load(a_file)