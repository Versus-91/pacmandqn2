import gym
env = gym.make("MsPacman-v4",render_mode='human')
env.reset()
while True:
    env.render()
    new_state, reward, is_done,_, info =env.step(env.action_space.sample()) # take a random action
    if is_done:
        print("finished reward ", reward)
        break