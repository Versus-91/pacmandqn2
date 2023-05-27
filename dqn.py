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
from game import GameWrapper
import random
import matplotlib
from time import sleep
matplotlib.use('Agg')


K_FRAME = 2
def optimization(it, r): return it % K_FRAME == 0 and r


REVERSED = {0: 1, 1: 0, 2: 3, 3: 2}
is_reversed = (
    lambda last_action, action: "default" if REVERSED[action] -
    last_action else "reverse"
)

ACTIONS = {
    1: [1, 4, 6, 5],
    2: [5, 7, 3, 2],
    3: [6, 8, 3, 2],
    4: [1, 4, 8, 7],
    5: [1, 4, 3, 2],
    6: [1, 4, 3, 2],
    7: [1, 4, 3, 2],
    8: [1, 4, 3, 2],
}

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 2.5e-4
MOMENTUM = 0.95

# Reinforcement learning constants
DISCOUNT_RATE = 0.99
EPS_MAX = 1.0
EPS_MIN = 0.1
TARGET_UPDATE = 60  # here
REPLAY_MEMORY_SIZE = 3 * 6000
steps_done = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTIONS = 4


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def optimize_model(policy_DQN, target_DQN, memory, optimizer, device):
    if len(memory) < BATCH_SIZE:
        return
    experiences = memory.sample(BATCH_SIZE)
    batch = Experience(*zip(*experiences))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    new_state_batch = torch.cat(batch.new_state)
    reward_batch = torch.cat(batch.reward)
    indices = random.sample(range(len(experiences)), k=BATCH_SIZE)
    def extract(list_): return [list_[i] for i in indices]
    done_array = [s for s in batch.done]
    dones = torch.from_numpy(
        np.vstack(extract(done_array)).astype(np.uint8)).to(device)
    predicted_targets = policy_DQN(state_batch).gather(1, action_batch)
    target_values = target_DQN(new_state_batch).detach().max(1)[0]
    labels = reward_batch + DISCOUNT_RATE * \
        (1 - dones.squeeze(1)) * target_values

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(predicted_targets,
                     labels.detach().unsqueeze(1)).to(device)
    # display.data.losses.append(loss.item())
    # print("loss", loss.item())
    optimizer.zero_grad()
    loss.backward()
    for param in policy_DQN.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    # # Softmax update
    # for target_param, local_param in zip(target_DQN.parameters(), policy_DQN.parameters()):
    #     target_param.data.copy_(TAU * local_param.data + (1 - TAU) * target_param.data)


policy_DQN = DQN(22 * 18, N_ACTIONS).to(device)
target_DQN = DQN(22 * 18, N_ACTIONS).to(device)
optimizer = optim.SGD(
    policy_DQN.parameters(), lr=LR, momentum=MOMENTUM, nesterov=True
)
steps_done = 0
episodes = 0
old_action = 0


def transform_reward(reward):
    return log(reward, 1000) if reward > 0 else reward


def select_action(state, policy_DQN):
    global steps_done
    global old_action
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # display.data.q_values.append(q_values.max(1)[0].item())
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_DQN(state)
        # Optimal action
        vals = q_values.max(1)[1]
        return vals.view(1, 1)
    else:
        # Random action
        action = random.randrange(N_ACTIONS)
        while action == REVERSED[old_action]:
            action = random.randrange(N_ACTIONS)
        return torch.tensor([[action]], device=device, dtype=torch.long)


plt.ion()
Experience = namedtuple('Experience', field_names=[
                        'state', 'action', 'reward', 'done', 'new_state'])


class ExperienceReplay:
    def __init__(self, capacity) -> None:
        self.buffer = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, done, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


memory = ExperienceReplay(18000)
game = GameWrapper()
episode_rewards = []
REWARDS = {
    "default": -0.2,
    200: 20,
    50: 15,
    10: 10,
    0: 0,
    "lose": -log(20, 1000),
    "win": 10,
    "reverse": -2,
}


def plot_images(images, rows, cols):
    """
    Display a list of images in a grid using Matplotlib.

    Args:
        images (list): List of images to be displayed.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

    for i, ax in enumerate(axes.flat):
        # Check if there are more images than available grid cells
        if i < len(images):
            ax.imshow(images[i])
            ax.axis('off')
        else:
            # Hide empty subplots
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_rewards(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig('plot.png')


obs = game.start()

while True:
    action_interval = 0.02  # 1 second
    start_time = time.time()
    frames = []
    episodes += 1
    lives = 3
    jump_dead_step = False
    obs, reward, done, info = game.step(2)
    obs = obs[0].flatten().astype(dtype=np.float32)
    state = torch.from_numpy(obs).unsqueeze(0).to(device)
    got_reward = False
    reward_sum = 0
    last_score = 0
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= action_interval:
            action = select_action(state, policy_DQN)
            action_t = action.item()
            obs, reward, done, remaining_lives = game.step(action_t)
            reward_ = reward - last_score
            if reward_ >= 200:
                reward_ = 20
            if last_score < reward:
                reward_sum += reward - last_score
            old_action = action_t
            last_score = reward
            if remaining_lives < lives:
                lives -= 1
                reward_ = -10
            if reward_ == last_score:
                reward_ = -0.2
            observation = obs[0].flatten().astype(dtype=np.float32)
            next_state = torch.from_numpy(observation).unsqueeze(0).to(device)
            action_tensor = torch.tensor(
                [[action_t]], device=device, dtype=torch.long)
            memory.append(state, action_tensor,
                          torch.tensor([reward_], device=device), next_state, done)

            state = next_state
            if steps_done % 2 == 0:
                optimize_model(
                    policy_DQN,
                    target_DQN,
                    memory,
                    optimizer,
                    device
                )
            if steps_done % TARGET_UPDATE == 0:
                target_DQN.load_state_dict(policy_DQN.state_dict())
            start_time = time.time()
        if done:
            assert reward_sum == reward
            episode_rewards.append(reward_sum)
            plot_rewards()
            game.restart()
            time.sleep(3)
            reward_sum = 0
            torch.cuda.empty_cache()
            break
        if episodes % 500 == 0:
            torch.save(policy_DQN.state_dict(), os.path.join(
                os.getcwd() + "\\results", f"policy-model-{episodes}.pt"))
            torch.save(target_DQN.state_dict(), os.path.join(
                os.getcwd() + "\\results", f"target-model-{episodes}.pt"))
