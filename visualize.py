import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pickle
import os

class LearningCurvePlot:
    '''
    A class for handling better a learning curve. This is because #curves = #repetitions for each combination.
    Also we apply smoothing to every learning curve. 
    This code is taken from the previous assignment after getting permission.  
    '''
    def __init__(self,title=None):
        '''
        Make the plot for the learning curve(s) and set the axis.
        param title:    title of the plot
        '''
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Reward')      
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self,y,label=None):
        ''' 
        Add the curve to the constructed plot.
        param y:        vector of average reward results
        param label:    string to appear as label in plot legend 
        '''
        if label is not None:
            self.ax.plot(y,label=label)
        else:
            self.ax.plot(y)
    
    def set_ylim(self,lower,upper):
        '''
        Set the limits for the y-axis.
        param lower:    lower limit
        param higher:   higher limit
        '''
        self.ax.set_ylim([lower,upper])

    def add_hline(self,height,label):
        '''
        In case of knowing the optimal value, we plot the horizontal line that represents that value.
        param height:   optimal value 
        param label:    label of the horizontal line
        '''
        self.ax.axhline(height,ls='--',c='k',label=label)

    def save(self,name='test.png'):
        ''' 
        Save the plot.
        param name:     string for filename of saved figure 
        '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)


def average_episode_length(episode_lengths):
    '''
    Obtain the average episode length.
    param episode_lengths:  the list (size of episodes) containing the episode_length for each episode
    '''
    episode_lengths = np.array(episode_lengths)
    return np.cumsum(episode_lengths) / np.arange(1, len(episode_lengths)+1)


def plot_episode_length(episode_lengths, experiment_label):
    '''
    Plot a given list of episode lengths for a combination of hyperparameters.
    param episode_lengths:  the list (size of episodes) containing the episode_length for each episode
    param experiment_label: combination of hyperparameters
    '''
    episode_lengths = np.array(episode_lengths)

    plt.figure(figsize=(15, 6))

    plt.scatter(range(len(episode_lengths)), episode_lengths, color='navy', label='episode length')

    average_ep_length = average_episode_length(episode_lengths)
    plt.plot(range(len(episode_lengths)), average_ep_length, color='chocolate', label='average episode length')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title(f'experiment {experiment_label}')
    plt.tight_layout()
    plt.savefig(f'fig {experiment_label} 2')
    # plt.show()


def smooth(y, window, poly=1):
    '''
    Smooth the learning curve with savgol_filter.
    param y:        vector to be smoothed 
    param window:   size of the smoothing window 
    '''
    return savgol_filter(y,window,poly)


if __name__ == '__main__':
    print('vizualizing results (using the code given in the assignment 1)')
    smoothing_window = 31
    reps = 3
    policy = 'annealing'

    for experiment_num in range(1,216+1):                       # GRAPHS: combinations
        Plot = LearningCurvePlot() 
        for dqn_version in ['DQN','DQN-TN','DQN-ER','DQN-TN-ER','DQN-TN-ER']:       # LINES IN EACH GRAPH: 4 DQN versions
            learning_curve_over_reps = list()
            for rep in range(1,reps+1):
                name = os.path.dirname(__file__) + '/hyperparams-tuning_' + policy + '/' + dqn_version + '/combination_' + str(experiment_num) + '__repetition_' + str(rep) + '.pkl'
                a_file = open(name, "rb")
                x_dict = pickle.load(a_file)
                title = x_dict.keys()
                Plot.ax.set_title(str(title),fontsize = 8)
                results = x_dict[list(x_dict.keys())[0]]
                a_file.close()
                learning_curve_over_reps.append(results)
            learning_curve = np.mean(learning_curve_over_reps,axis=0) # average over repetitions
            learning_curve = smooth(learning_curve,smoothing_window)
            Plot.add_curve(learning_curve,label=r'{}'.format(dqn_version))
            

        filename = policy + '-test' + str(experiment_num) +  '_smooth_win' + str(smoothing_window) + '.png'
        Plot.save(filename)

    #-------------------------------------------------------------------------------------

    for experiment_num in range(1,126+1):                       # GRAPHS: combinations
        Plot = LearningCurvePlot() 
        for dqn_version in ['DQN','DQN-TN','DQN-ER','DQN-TN-ER','DQN-TN-ER']:       # LINES IN EACH GRAPH: 4 DQN versions
            learning_curve_over_reps = list()
            for rep in range(1,reps+1):
                name = os.path.dirname(__file__) + '/hyperparams-tuning_architecture/' + dqn_version + '/combination_' + str(experiment_num) + '__repetition_' + str(rep) + '.pkl'
                a_file = open(name, "rb")
                x_dict = pickle.load(a_file)
                title = x_dict.keys()
                Plot.ax.set_title(str(title),fontsize = 8)
                results = x_dict[list(x_dict.keys())[0]]
                a_file.close()
                learning_curve_over_reps.append(results)
            learning_curve = np.mean(learning_curve_over_reps,axis=0) # average over repetitions
            learning_curve = smooth(learning_curve,smoothing_window)
            Plot.add_curve(learning_curve,label=r'{}'.format(dqn_version))
            

        filename = 'architecture-test' + str(experiment_num) +  '_smooth_win' + str(smoothing_window) + '.png'
        Plot.save(filename)