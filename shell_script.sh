#!/bin/bash
# pyenv shell 3.8.0
for j in 1 # {1..3}
do 
	for i in {10..12} #{1..9}
	do 
		echo ">>> Shell script here <<<"
		echo ">>> Starting a new experiment: $i - repetition $j <<<"
		/mnt/c/Users/User/.pyenv/pyenv-win/versions/3.8.0/python.exe c:/Users/User/Documents/Github/RL_A2/hyperparam_tune.py $i $j
		echo ">>> Completed the experiment: $i - repetition $j <<<"
	done
done
