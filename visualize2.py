import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pickle
import os

class LearningCurvePlot:

    def __init__(self,title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Reward')      
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self, y, label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y, label=label)
        else:
            self.ax.plot(y)
    
    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height,label):
        self.ax.axhline(height, ls='--', c='k', label=label)

    def save(self, name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name, dpi=300)

# def average_episode_length(episode_lengths):
#     return np.cumsum(episode_lengths) / np.arange(1, len(episode_lengths)+1)
#
#
# def plot_episode_length(episode_lengths, experiment_label):
#     episode_lengths = np.array(episode_lengths)
#
#     plt.figure(figsize=(15, 6))
#
#     plt.scatter(range(len(episode_lengths)), episode_lengths, color='navy', label='episode length')
#
#     average_ep_length = average_episode_length(episode_lengths)
#     plt.plot(range(len(episode_lengths)), average_ep_length, color='chocolate', label='average episode length')
#     plt.legend(bbox_to_anchor=(1, 1))
#     plt.title(experiment_label)
#     plt.tight_layout()
#     plt.savefig(f'fig {experiment_label} 2')
#     # plt.show()


def smooth(y, window=5, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y, window, poly)


def boltzmann_graphs():
    smoothing_window = 5
    num_repetitions = 3
    num_episodes = 500
    experiments = np.arange(1, 4)  # for Boltzmann
    dqn_versions = ['DQN', 'DQN-TN', 'DQN-ER', 'DQN-ER-TN']
    fname_versions = ['dqn', 'dqn_tn', 'dqn_er', 'dqn_er_tn']
    dir = os.getcwd() + r'\boltzmann_new'
    print(dir)

    for experiment in experiments:
        print(f'\n\t\tExperiment = {experiment}')
        fig = plt.figure(figsize=(12, 8))



        for dqn_version, fname_version in zip(dqn_versions, fname_versions):
            num_repetitions = 1 if dqn_version == 'DQN-ER-TN' else 3
            if experiment == 3 and dqn_version == 'DQN-ER':
                num_repetitions = 2

            episode_lengths = np.zeros((num_repetitions, num_episodes))

            for repetition in range(num_repetitions):
                path = dir + chr(92) + dqn_version + '-boltzmann' + chr(92) + f'combination_{experiment}__repetition_{repetition+1}.pkl'
                print('\n', experiment, dqn_version)
                print(path)

                file = open(path, "rb")
                data = pickle.load(file)

                a = list(data.keys())[0]
                print(a)
                learning_rate = a[0][1]
                exploration = a[1][1]
                temperature = a[2][1]
                # print(f'temperature = {temperature}')

                print(f'learning rate {learning_rate}, exploration {exploration}, temperature {temperature}')

                measured_ep_lengths = data[list(data.keys())[0]]
        #
        #
                if dqn_version != 'DQN-ER-TN':
                    episode_lengths[repetition] = smooth(measured_ep_lengths, window=smoothing_window)

                    average_ep_length = np.mean(episode_lengths, axis=0)
                    std = np.std(episode_lengths, axis=0)
                else:
                    average_ep_length = smooth(measured_ep_lengths, window=smoothing_window)
        #
            plt.plot(np.arange(num_episodes), average_ep_length, label=dqn_version)
            plt.fill_between(np.arange(num_episodes), average_ep_length-std, average_ep_length+std, alpha=0.2)

        plt.axis(xmin=0, xmax=num_episodes, ymin=0)
        plt.legend(loc='upper left')
        plt.grid()
        plt.title(f'Boltzmann Exploration with learning rate {learning_rate} and temperature {temperature}')
        plt.savefig(f'figure_boltzmann_new_experiment={experiment}_lr={learning_rate}_temp={temperature}.png')
        plt.show()

def first_tuning_graphs():
    smoothing_window = 5
    num_repetitions = 3
    num_episodes = 500
    experiments = [1, 2, 4, 5, 7, 8, 10, 11, 12]

    dqn_versions = ['DQN', 'DQN-TN', 'DQN-ER', 'DQN-TN-ER']
    # fname_versions = ['dqn', 'dqn_tn', 'dqn_er', 'dqn_er_tn']
    dir = os.getcwd() + r'\RL-as2-details4runs'


    for experiment in experiments:
        print(f'\n\t\tExperiment = {experiment}')
        fig = plt.figure(figsize=(12, 8))

        for dqn_version in dqn_versions:

            episode_lengths = np.zeros((num_repetitions, num_episodes))

            for repetition in range(num_repetitions):
                path = dir + chr(92) + dqn_version + chr(92) + f'combination_{experiment}__repetition_{repetition+1}.pkl'
                print('\n', experiment, dqn_version)
                print(path)

                file = open(path, "rb")
                data = pickle.load(file)

                a = list(data.keys())[0]
                print(a)
                initial_exploration = a[0][1]
                final_exploration = a[1][1]
                decay_constant = a[2][1]
                learning_rate = a[3][1]

                print(f'learning rate {learning_rate}, decay constant {decay_constant}, initial exploration {initial_exploration}, final exploration {final_exploration}')

                measured_ep_lengths = data[list(data.keys())[0]]

                episode_lengths[repetition] = smooth(measured_ep_lengths, window=smoothing_window)

            average_ep_length = np.mean(episode_lengths, axis=0)
            std = np.std(episode_lengths, axis=0)

            plt.plot(np.arange(num_episodes), average_ep_length, label=dqn_version)
            plt.fill_between(np.arange(num_episodes), average_ep_length-std, average_ep_length+std, alpha=0.2)

        plt.axis(xmin=0, xmax=num_episodes, ymin=0)
        plt.legend(title='Model', loc='upper left')
        plt.grid()
        plt.title(f'annealing e-greedy exploration with learning rate {learning_rate} and decay constant {decay_constant}')
        plt.xlabel('Episode')
        plt.ylabel('Episode length')
        plt.savefig(f'figure_hyperparam_tune_1--experiment={experiment}_lr={learning_rate}_decay={decay_constant}.png')
        # plt.show()


def check():
    smoothing_window = 5
    num_repetitions = 3
    num_episodes = 500
    # experiments = [1, 2, 4, 5, 7, 8, 10, 11, 12]
    experiments = np.arange(1, 10)  # for Boltzmann
    dqn_versions = ['DQN'] #, 'DQN-TN', 'DQN-ER', 'DQN-ER-TN']
    fname_versions = ['dqn', 'dqn_tn', 'dqn_er', 'dqn_er_tn']
    dir = os.getcwd() + r'\boltzmann'
    print(dir)

    # experiments = [8]
    # dqn_versions = ['DQN-ER-TN']
    # fname_versions = ['dqn_er_tn']

    for experiment in experiments:
        print(f'\n\t\tExperiment = {experiment}')
        # fig = plt.figure(figsize=(12, 8))

        for dqn_version, fname_version in zip(dqn_versions, fname_versions):

            # episode_lengths = np.zeros((num_repetitions, num_episodes))

            for repetition in range(num_repetitions):
                path = r'C:\Users\Gebruiker\PycharmProjects\RL_assignment2\RL-as2-details-boltzmann-newruns\DQN-boltzmann' + chr(92) + f'combination_{experiment}.pkl'
                # print('\n', experiment, dqn_version)
                print(path)

                file = open(path, "rb")
                data = pickle.load(file)

                a = list(data.keys())[0]
                print(a)
                # learning_rate = a[1][1]
                # exploration = a[2][1]
                # temperature = a[3][1]
                # print(f'temperature = {temperature}')
                #
                # print(f'learning rate {learning_rate}, exploration {exploration}, temperature {temperature}')


def architecture_tuning():
    smoothing_window = 5
    num_repetitions = 1  # TODO
    num_episodes = 300
    # experiments = [1, 2, 4, 5, 7, 8, 10, 11, 12]
    experiments = np.arange(1, 7)
    # dqn_versions = ['DQN', 'DQN-TN', 'DQN-ER', 'DQN-ER-TN']
    # fname_versions = ['dqn', 'dqn_tn', 'dqn_er', 'dqn_er_tn']
    dir = os.getcwd() + r'\Architecture'
    print(dir)




    for experiment in experiments:
        print(f'\n\t\tExperiment = {experiment}')
        loss = 'Huber' if experiment <= 3 else 'Mean Squared Error'
        initializer = 'HeUniform' if experiment <= 3 else 'Glorot Uniform'
        if experiment in [1, 4]:
            neurons_layers = [24, 12]
        elif experiment in [2, 5]:
            neurons_layers = [64, 128]
        elif experiment in [3, 6]:
            neurons_layers = [128, 128]
        else:
            print('ER GAAT IETS MIS MET NEURONS LAYERS')
            sys.exit()

        fig = plt.figure(figsize=(12, 8))

        # episode_lengths = np.zeros((num_repetitions, num_episodes))

        for repetition in range(num_repetitions):
            path = dir + chr(92) + f'combination_{experiment}__repetition_{repetition+1}.pkl'
            print('\n', experiment)
            print(path)

            file = open(path, "rb")
            data = pickle.load(file)

            a = list(data.keys())[0]
            print(a)
            print(f'loss {loss}, initializer {initializer}, neurons layers {neurons_layers}')
            # print(f'learning rate {learning_rate}, exploration {exploration}, temperature {temperature}')

            measured_ep_lengths = data[list(data.keys())[0]]

    #         episode_lengths[repetition] = smooth(measured_ep_lengths, window=smoothing_window)
            measured_ep_lengths = np.array(smooth(measured_ep_lengths))
    #
    #     average_ep_length = np.mean(episode_lengths, axis=0)
    #     std = np.std(episode_lengths, axis=0)
    #
        plt.plot(np.arange(num_episodes), measured_ep_lengths)
    #     plt.fill_between(np.arange(num_episodes), average_ep_length-std, average_ep_length+std, alpha=0.2)
    #
        plt.axis(xmin=0, xmax=num_episodes, ymin=0)
        plt.legend(loc='upper left')
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.grid()
        plt.title(f'Architecture tuning with neurons {neurons_layers}, initializer {initializer}, loss {loss}')
        plt.savefig(f'figure_architecture_experiment={experiment}-initializer={initializer}_loss={loss}_neurons={neurons_layers}.png')
        plt.show()



def architecture_tuning2():
    smoothing_window = 5
    num_repetitions = 3  # TODO
    num_episodes = 300
    experiments = np.arange(1, 7)
    dir = os.getcwd() + r'\Architecture'
    print(dir)
    colours = ['coral', 'green', 'mediumslateblue']

    for exp, plot in zip([[1, 2, 3], [4, 5, 6]], ['A', 'B']):
        fig = plt.figure(figsize=(12, 8))

        if plot == 'A':
            loss = 'Huber'
            initializer = 'HeUniform'
        elif plot == 'B':
            loss = 'Means Squared Error'
            initializer = 'Glorot Uniform'

        for experiment in exp:
            print(f'\n\t\tExperiment = {experiment}')
            if experiment in [1, 4]:
                neurons_layers = [24, 12]
            elif experiment in [2, 5]:
                neurons_layers = [64, 128]
            elif experiment in [3, 6]:
                neurons_layers = [128, 128]
            else:
                print('ER GAAT IETS MIS MET NEURONS LAYERS')
                sys.exit()


            episode_lengths = np.zeros((num_repetitions, num_episodes))

            for repetition in range(num_repetitions):
                path = dir + chr(92) + f'combination_{experiment}__repetition_{repetition+1}.pkl'
                print('\n', experiment)
                print(path)

                file = open(path, "rb")
                data = pickle.load(file)

                a = list(data.keys())[0]
                print(a)
                print(f'loss {loss}, initializer {initializer}, neurons layers {neurons_layers}')

                measured_ep_lengths = data[list(data.keys())[0]]

                episode_lengths[repetition%3] = smooth(measured_ep_lengths, window=smoothing_window)
                print(episode_lengths[:, 0])
                # measured_ep_lengths = np.array(smooth(measured_ep_lengths))

            average_ep_length = np.mean(episode_lengths, axis=0)
            std = np.std(episode_lengths, axis=0)

            plt.plot(np.arange(num_episodes), average_ep_length, color=colours[experiment%3], label=f'{neurons_layers[0]}, {neurons_layers[1]}')
            plt.fill_between(np.arange(num_episodes), average_ep_length-std, average_ep_length+std, alpha=0.15, color=colours[experiment%3])

        plt.axis(xmin=0, xmax=num_episodes, ymin=0)
        plt.legend(title='number of neurons per layer', loc='upper left')
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.grid()
        plt.title(f'Architecture tuning with initializer {initializer}, loss {loss}')
        plt.savefig(f'figure_architecture_plot{plot}_averaged-initializer={initializer}_loss={loss}.png')
        plt.show()


if __name__ == '__main__':

    # first_tuning_graphs()

    # check()

    # architecture_tuning2()

    boltzmann_graphs()






