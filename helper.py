import numpy as np

def exponential_anneal(t,start,final,decay_constant):
    ''' Linear annealing scheduler
    t: current timestep
    start: initial value
    final: value after percentage*T steps'''
    return final + (start - final) * np.exp(-decay_constant*t)