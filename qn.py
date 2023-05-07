
import gym
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import random
import datetime
from constants import *
import torchvision.transforms as transforms

from game import GameWrapper
reward_number = 0.37
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PacmanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(in_features=32 * 8 * 8, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = PacmanModel()


model = model.to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


class DQNAgent:
    def __init__(self, action_size=4):
        self.state_size = 4
        self.action_size = action_size
        self.memory_n = deque(maxlen=2000)
        self.memory_p = deque(maxlen=2000)
        self.gamma = 1.0    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.5
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        if reward == 0:
            self.memory_p.append((state, action, reward, next_state, done))
        else:
            self.memory_n.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice([UP, DOWN, LEFT, RIGHT])
        screen_tensor = torch.from_numpy(
            state.transpose((2, 0, 1))).float() / 255.0
        # Add a batch dimension to the tensor
        screen_tensor = screen_tensor.unsqueeze(0)
        # Create a conv2d module with input size of (batch_size, num_channels, height, width)
        act_values = self.model(screen_tensor.to(device)
                                ).cpu().detach().numpy()
        if len(act_values) == 0 or np.any(act_values < 0):
            # handle empty or negative arrays
            return random.choice([UP, DOWN, LEFT, RIGHT])
        result = np.argmax(act_values[0])
        return result  # returns action

    def replay(self, batch_size):
        if len(agent.memory_n) > batch_size / 2:
            print("Negative batch ready:")
            minibatch_n = random.sample(self.memory_n, 5)
            minibatch_p = random.sample(self.memory_p, 59)
            minibatch = random.sample((minibatch_p+minibatch_n), batch_size)
        else:
            minibatch = random.sample(self.memory_p, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # reward = 0.001 if reward == 0 else 0.001
            ns_model = self.model(torch.from_numpy(
                next_state).float()).cpu().detach().numpy()
            if reward == 0:
                reward = 1.0001
                # print("Reward:", reward)
                target = reward * np.amax(ns_model[0])
                # print("target: ", target)
                target_f = ns_model
                # print("target_f: ", target_f)
                # print('Argmax: ', np.argmax(ns_model[0]))
                target_f[0][np.argmax(ns_model[0])] = target
                # print("target_f[0][np.argmax(ns_model[0])]: ", target_f)
            else:
                reward = reward_number
                # print("Reward:", reward)
                target = reward * np.amin(ns_model[0])
                # print("target: ", target)
                target_max = 0.0001 * np.amax(ns_model[0])
                # print("target_max: ", target_max)
                target_f = ns_model
                # print("target_f: ", target_f)
                target_f[0][action] = target
                target_f[0][random.choice(
                    [i for i in range(0, 4) if i not in [action]])] = target_max
                # print("target_f[0][several actions]: ", target_f)
            self.train(next_state, target_f, epochs=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, input, target, epochs=1):
        input = torch.from_numpy(input).float().cuda()
        target = torch.from_numpy(target).float().cuda()
        y_pred = 0
        for t in range(1):
            y_pred = model(input)
            loss = - criterion(y_pred, target)
            # print(t, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def load_all(self, name):
        loaded = torch.load(name)
        self.memory_n = loaded['memory_n']
        self.memory_p = loaded['memory_p']
        self.model.load_state_dict(loaded['state'])

    def save_all(self, name):
        torch.save({'state': self.model.state_dict(),
                    'memory_n': self.memory_n,
                    'memory_p': self.memory_p
                    }, name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


env = GameWrapper()
action_size = 4
agent = DQNAgent()

done = False
batch_size = 32
EPISODES = 5
for e in range(EPISODES):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vw = cv2.VideoWriter('gdrive/My Drive/Colab Notebooks/Pacman/' + "Reward_number_" + str(
        reward_number) + "_" + str(e) + str(datetime.datetime.now()) + '.avi', fourcc, 4, (160, 210))
    state = env.reset()
    s = state[0]
    for time in range(1000000000):
        print(time)
        action = agent.act(s)
        next_state, reward, done, _ = env.step(action)
        vw.write(next_state)
        reward = reward if not done else 10
        reward = reward if reward == 0 else 10
        if reward != 0:
            print("ATTENTION NEGATIVE REWARD", reward)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            vw.release()
            agent.save('./results/' + "Reward_number_" +
                       str(reward_number) + '.pt')
            print("saved")
        if (len(agent.memory_p) > batch_size) & (len(agent.memory_n) > batch_size/2):
            agent.replay(batch_size)
