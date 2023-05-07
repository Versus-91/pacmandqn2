from collections import namedtuple
import random
from sysconfig import is_python_build
import time
import cv2
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
from game import GameWrapper
from utils import display

color = np.array([210, 164, 74]).mean()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, output_dim):
        super(DQN, self).__init__()

        self.device = device
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        # (1, 84, 84) -> (16, 40, 40) -> (32, 18, 18) -> (64, 7, 7)

        self.fc1 = nn.Linear(7*7*64, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        return self.fc1(x)


def preprocess_observation(obs):

    # Crop and resize the image
    res = cv2.resize(obs, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
    crop_img = res[15:190, 0:200]
    # Convert the image to greyscale
    crop_img = crop_img.mean(axis=2)
    # Improve image contrast
    crop_img[crop_img == color] = 0
    res = cv2.resize(crop_img, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('image', res)
    return res
    # Next we normalize the image from -1 to +1
    # cv2.imshow('image', image)


model = DQN(5).to(device)
game = GameWrapper()
state = game.reset()
image = preprocess_observation(state[0])
tensor = torch.tensor(image, dtype=torch.float32,
                      device=device).unsqueeze(0).unsqueeze(0)
# # Define the Conv2d layer
# conv_layer = torch.nn.Conv2d(
#     in_channels=1, out_channels=16, kernel_size=5, stride=2)
# conv_layer.to(device)
# # Pass the tensor through the Conv2d layer
# output = conv_layer(tensor).to(device)
output = model(tensor)
print(torch.argmax(output, dim=1).item())
