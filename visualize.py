import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pickle

class LearningCurvePlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Reward')      
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self,y,label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y,label=label)
        else:
            self.ax.plot(y)
    
    def set_ylim(self,lower,upper):
        self.ax.set_ylim([lower,upper])

    def add_hline(self,height,label):
        self.ax.axhline(height,ls='--',c='k',label=label)

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

def average_episode_length(episode_lengths):
    return np.cumsum(episode_lengths) / np.arange(1, len(episode_lengths)+1)


def plot_episode_length(episode_lengths, experiment_label):
    episode_lengths = np.array(episode_lengths)

    plt.figure(figsize=(15, 6))

    plt.scatter(range(len(episode_lengths)), episode_lengths, color='navy', label='episode length')

    average_ep_length = average_episode_length(episode_lengths)
    plt.plot(range(len(episode_lengths)), average_ep_length, color='chocolate', label='average episode length')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title(experiment_label)
    plt.tight_layout()
    plt.savefig(f'fig {experiment_label} 2')
    # plt.show()

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

if __name__ == '__main__':
    print('vizualizing results (using the code given in the assignment 1)')
    smoothing_window = 10
    reps = 3

    for experiment_num in range(1,9+1):
        Plot = LearningCurvePlot() 
        for dqn_version in ['DQN-TN-ER']: #for dqn_version in ['DQN','DQN-TN','DQN-ER','DQN-TN-ER']:
            learning_curve_over_reps = list()
            for rep in range(1,reps+1):
                # learning curve: actual data
                name = 'C:/Users/User/Documents/GitHub/RL_A2/RL-as2-details4runs/' + dqn_version + '/combination_' + str(experiment_num) + '__repetition_' + str(rep) + '.pkl'
                a_file = open(name, "rb")
                x_dict = pickle.load(a_file)
                # print('keys:',x_dict.keys())
                # print('values:',x_dict[list(x_dict.keys())[0]])
                title = x_dict.keys()
                Plot.ax.set_title(str(title),fontsize = 8)
                results = x_dict[list(x_dict.keys())[0]]
                a_file.close()
                learning_curve_over_reps.append(results)
            learning_curve = np.mean(learning_curve_over_reps,axis=0) # average over repetitions
            learning_curve = smooth(learning_curve,smoothing_window)
            Plot.add_curve(learning_curve,label=r'{}'.format(dqn_version))
            

        filename = 'test' + str(experiment_num) +  '_smooth_win' + str(smoothing_window) + '.png'
        Plot.save(filename)