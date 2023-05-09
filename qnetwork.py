# create a dqn eperience replay buffer
from collections import deque, namedtuple
import random
import cv2
import numpy as np
from constants import *
from game import GameWrapper
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
steps_done = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learn_counter = 0
N_ACTIONS = 4
TARGET_UPDATE = 8_000  # here


def preprocess_observation(obs):
    color = np.array([210, 164, 74]).mean()
    # Crop and resize the image
    res = cv2.resize(obs, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
    crop_img = res[15:190, 0:200]
    # Convert the image to greyscale
    crop_img = crop_img.mean(axis=2)
    # Improve image contrast
    crop_img[crop_img == color] = 0
    res = cv2.resize(crop_img, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
    return res


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


def optimize_model(policy_DQN, target_DQN, memory, optimizer, learn_counter, device):
    if len(memory) < BATCH_SIZE:
        return learn_counter
    learn_counter += 1
    states, actions, rewards, next_states, dones = memory.sample()

    predicted_targets = policy_DQN(states).gather(1, actions)

    target_values = target_DQN(next_states).detach().max(1)[0]
    labels = rewards + GAMMA * (1 - dones.squeeze(1)) * target_values

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(predicted_targets,
                     labels.detach().unsqueeze(1)).to(device)
    # display.data.losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    for param in policy_DQN.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    # # Softmax update
    # for target_param, local_param in zip(target_DQN.parameters(), policy_DQN.parameters()):
    #     target_param.data.copy_(TAU * local_param.data + (1 - TAU) * target_param.data)

    return learn_counter


policy_DQN = DQN(5).to(device)
target_DQN = DQN(5).to(device)
optimizer = optim.SGD(
    policy_DQN.parameters(), lr=LR, momentum=GAMMA, nesterov=True
)
steps_done = 0


def select_action(state, policy_DQN, learn_counter):
    global steps_done
    sample = random.random()
    eps_threshold = max(EPS_START, EPS_END - (EPS_END - EPS_START)
                        * learn_counter / EPS_DECAY)
    steps_done += 1
    tensor = torch.tensor(state, dtype=torch.float32,
                          device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_DQN(tensor)
    # display.data.q_values.append(q_values.max(1)[0].item())
    if sample > eps_threshold:
        # Optimal action
        return q_values.max(1)[1].view(1, 1)
    else:
        # Random action
        action = random.randrange(N_ACTIONS)
        # while action == REVERSED[self.old_action]:
        #     action = random.randrange(N_ACTIONS)
        return torch.tensor([[action]], device=device, dtype=torch.long)


plt.ion()
Experience = namedtuple('Experience', field_names=[
                        'state', 'action', 'reward', 'done', 'new_state'])


class ExperienceReplay:
    def __init__(self, capacity) -> None:
        self.buffer = deque(maxlen=capacity)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)


memory = ExperienceReplay(10000)
game = GameWrapper()
# Main loop
while True:
    # if dmaker.steps_done > MAX_FRAMES:
    #     break
    # episodes += 1

    obs = game.reset()
    lives = 3
    jump_dead_step = False
    old_action = 0
    obs, reward, done, info = game.step(3)
    state = preprocess_observation(obs)
    got_reward = False
    old_action = 3

    while True:
        # if dmaker.steps_done > MAX_FRAMES:
        #     break
        # epsilon greedy decision maker
        action = select_action(
            state, policy_DQN, learn_counter)
        # action_ = ACTIONS[old_action][action.item()]
        if action.item() == 0:
            action = UP
        elif action.item() == 1:
            action = DOWN
        elif action.item() == 2:
            action = LEFT
        elif action.item() == 3:
            action = RIGHT
        obs, reward_, done, remaining_lives = game.step(action)
        # display.obs = obs.copy()
        update_all = False
        if remaining_lives < lives:
            lives -= 1
            jump_dead_step = True
            got_reward = False
            reward += -100
            # dmaker.old_action = 3
            update_all = True

        if done and lives > 0:
            reward += +100

        got_reward = got_reward or reward != 0
        # display.data.rewards.append(reward)
        reward = torch.tensor([reward], device=device)

        old_action = action
        # if reward != 0:
        #     dmaker.old_action = action.item()

        next_state = preprocess_observation(obs)

        if got_reward:
            memory.push(state, action, reward, next_state, done)

        state = next_state
        learn_counter = optimize_model(
            policy_DQN,
            target_DQN,
            memory,
            optimizer,
            learn_counter,
            device,
        )

        if steps_done % TARGET_UPDATE == 0:
            target_DQN.load_state_dict(policy_DQN.state_dict())

        # display.stream(update_all)
        # if done:
        #     display.data.successes += remaining_lives > 0
        #     torch.cuda.empty_cache()
        #     break
        # if jump_dead_step:
        #     for i_dead in range(DEAD_STEPS):
        #         obs, reward, done, info = game.step(0)
        #     jump_dead_step = False
        torch.cuda.empty_cache()

    # if episodes % SAVE_MODEL == 0:
    #     torch.save(policy_DQN.state_dict(), PATH_MODELS /
    #                f"policy-model-{episodes}.pt")
    #     torch.save(target_DQN.state_dict(), PATH_MODELS /
    #                f"target-model-{episodes}.pt")
    #     display.save()

    # display.data.round()
    # torch.save(policy_DQN.state_dict(), PATH_MODELS / f"policy-model-final.pt")
    # torch.save(target_DQN.state_dict(), PATH_MODELS / f"target-model-final.pt")
    # print("Complete")
