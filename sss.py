import time
import gym
import numpy as np

frame_time = 1.0 / 15 # seconds
n_episodes = 500
env=gym.make('MsPacman-v0',render_mode='human')

scores = []
for i_episode in range(n_episodes):
    t=0
    score=0
    then = 0
    done = False
    env.reset()
    while not done:
        now = time.time()
        if frame_time < now - then:
            action = env.action_space.sample()
            observation, reward, done,_, info = env.step(action)
            print(reward)
            score += reward
            env.render()
            then = now
            t=t+1
    scores.append(score)
