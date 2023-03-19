import numpy as np
import sys

def ckeckCMD():
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
    if error_msg == 'error_length':
        print('The command was not accepted due to the following reason: too many/few arguments')
    if error_msg == 'error_term':
        print('The command was not accepted due to the following reason: wrong argument(s) given')
    print('Please follow one of the examples listed below:')
    print('$ python dqn.py')
    print('$ python dqn.py --experience_replay')
    print('$ python dqn.py --target_network')
    print('$ python dqn.py --experience_replay --target_network')


def exponential_anneal(t,start,final,decay_constant):
    ''' Linear annealing scheduler
    t: current timestep
    start: initial value
    final: value after percentage*T steps'''
    return final + (start - final) * np.exp(-decay_constant*t)