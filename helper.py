import numpy as np

def exponential_anneal(t,T,start,final,percentage,decay_constant):
    ''' Linear annealing scheduler
    t: current timestep
    T: total timesteps
    start: initial value
    final: value after percentage*T steps
    percentage: percentage of T after which annealing finishes
    '''
    final_from_T = int(percentage*T)
    if t > final_from_T:
        return final
    else:
        return final + (start - final) * np.exp(-decay_constant*t)