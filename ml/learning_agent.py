from math import log
import math
import os
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import time
import cv2
import numpy as np
from constants import *
from dqn import DQN
from game import GameWrapper
import random
import matplotlib
from time import sleep
matplotlib.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPLAY_MEMORY_SIZE = 65000


class LearningAget:
    def __init__(self) -> None:
        self.env = GameWrapper()
        self.target_net = DQN().to(device)
        self.policy_net = DQN().to(device)
        self.episodes = 2000
        self.steps_done = 0
        self.optimizer = optim.SGD(
            self.policy_net.parameters(), lr=0.01, momentum=0.9)
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.tau = 0.005
        self.lr = 2.5e-4
        self.momentum = 0.95
        self.discount_rate = 0.99

    def training(self):
        for episode in range(self.episodes):
            self.env.reset()
            state = self.env.get_state()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                self.memory.push((state, action, reward, next_state, done))
                state = next_state
                self.optimize_model()
            self.target_net.load_state_dict(self.policy_net.state_dict())
            if episode % 10 == 0:
                self.save_model()
