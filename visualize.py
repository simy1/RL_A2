import numpy as np
import matplotlib.pyplot as plt

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
