# Reinforcement Learning - DQN


## Brief Description
This project applies the DQN algorithm to solve the CartPole Environment (OpenAI GYM). Apart from the DQN baseline algorithm, we implement three more DQN versions: a DQN version that maintains an Experience Replay Buffer, another one that uses a Target Network and the standard DQN version (which combines these two techniques). Ragrding exploration strategies, we use epsilon annealing and boltzmann policies. We focus on how these versions of DQN, and especially the most powerful one, deal with the problem when hyperparameters change / are adjusted such as the learning rate, eploration factor, exploration strategy and architecture of the neural network. In addition, we investigate how the algorithm responds to a different environment with distinct philosophy. Therefore, we test it in the Acrobot Environment (OpenAI GYM).


## Files
- dqn.py: This file hosts the implementation of DQN. Therefore, we can make a single run consisting of the best hyperparameters (by default) or other desired ones for one out of the 4 following DQN versions: DQN, DQN-ER, DQN-TN, DQN-ER-TN. The program runs from the command line (see section 'How to run') and when it finishes it prints out some information like the total time, the numnber of episodes of the single run and the results.

- hyperparam_tune.py: The certain file is used when we want to test our model with different hyperparameter combinations. This enables us to conduct experiments regarding both the exploration strategy and the architecture. After the experiments are finished with all the repetitions, data is stored in two directories (one for the architecture and one for the chosen policy).

- helper.py: This file consists of functions that consider to be helpful to either run a DQN version or perform hyperparameter tuning. Thus, it is the only python file which is not a stand-alone program. Some of the functions provided have to do with getting and controlling the input command line from terminal, others for saving the data for future use (such as plotting figures). Functions that implement exponential annealing and boltzmann policy can also be found in this file.

- vizualize.py: The specific python file is run whenever we want to plot our stored results and, thus, it is used after the hyperparam_tune.py. For that reason, there is a class for handling better our learning curves. In general, a good practise is to obtain the mean over a number of repetitions of the results and smooth the learning curve. This code is taken from the previous assignment after getting permission. 

- acrobot.py: This file contains the code that solves the Acrobot Environment (OpenAI GYM) with the DQN algorithm. When the code ends, it returns some information like the total time, the numnber of episodes and a figure which depicts the performance of the algorithm for this problem.

- requirements.txt: It saves a list of the modules and packages required for our project. Note: the whole project was implemented in a new virtual environment and therefore it contains only the required modules and packages for the purpose of the project with NO additional useless packages.  


## How to run 
- Get the modules for the project:    pip install -r requirements.txt

- Single run of the DQN algorithm:    python dqn.py
                                    python dqn.py --experience_replay
                                    python dqn.py --target_network
                                    python dqn.py --experience_replay --target_network

- Tune hyperparams:                   python hyperparam_tune.py

- Make plots:                         python [fullpath/]visualize.py
This code access already stored files. Thus in case you make a new virtual environment to run the code, you have to specify the fullpath of the python file to be able to actually retrieve the files (for safety reasons). 

- Run for the acrobot environment:    python acrobot.py
                                    python acrobot.py --experience_replay
                                    python acrobot.py --target_network
                                    python acrobot.py --experience_replay --target_network


## License
MIT License