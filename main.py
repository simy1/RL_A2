import gymnasium as gym
import numpy as np
import pygame
import tensorflow as tf

env = gym.make('CartPole-v1', render_mode='human')
observation,info = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

