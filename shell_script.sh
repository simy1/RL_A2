#!/bin/bash
# pyenv shell 3.8.0
for i in {1..9}
do
    echo ">>> Shell script here <<<"
    echo ">>> starting a new experiment: $i <<<"
    /mnt/c/Users/User/.pyenv/pyenv-win/versions/3.8.0/python.exe c:/Users/User/Documents/GitHub/RL_A2/hyperparam_tune.py $i
    echo ">>> completed the experiment: $i <<<"
done
