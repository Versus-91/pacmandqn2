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
BATCH_SIZE = 10
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
    print(res.shape)
    crop_img = res[15:190, 0:200]
    print(crop_img.shape)

    # Convert the image to greyscale
    # crop_img = crop_img.mean(axis=2)
    # Improve image contrast
    # crop_img[crop_img == color] = 0
    res = cv2.resize(crop_img, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
    print(res.shape)
    # Convert image to PyTorch tensor and normalize
    img = np.transpose(res, (2, 0, 1)).astype(np.float32) / 255.0
    img = torch.from_numpy(img).to(device)
    return img.unsqueeze(0)
    # Add batch dimension
    img = img.unsqueeze(0)
    print(img.shape)
    screen = torch.from_numpy(res).to(device)
    return img.to(device)
    # return screen
    return torch.from_numpy(res).unsqueeze(0).unsqueeze(0).to(device)


class DQN(nn.Module):
    def __init__(self, output_dim):
        super(DQN, self).__init__()

        self.device = device
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
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
    experiences = memory.sample(BATCH_SIZE)
    batch = Experience(*zip(*experiences))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    predicted_targets = policy_DQN(state_batch).gather(1, action_batch)
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


policy_DQN = DQN(4).to(device)
target_DQN = DQN(4).to(device)
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
    with torch.no_grad():
        q_values = policy_DQN(state)
    # display.data.q_values.append(q_values.max(1)[0].item())
    if sample > eps_threshold:
        # Optimal action
        print("optimal action")
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

    def append(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, done, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, 10)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        batch = Experience(*zip(*transitions))
        return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.buffer)


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
        # actions = [0, 1, 2, 3]
        # predicted_targets = policy_DQN(state).gather(1, torch.tensor([actions], dtype=torch.long).to(device))
        # target_values = target_DQN(state).detach().max(1)[0]
        action = select_action(
            state, policy_DQN, learn_counter)
        # action_ = ACTIONS[old_action][action.item()],
        action_t = action.item()
        if action_t == 0:
            action = UP
        elif action_t == 1:
            action = DOWN
        elif action_t == 2:
            action = LEFT
        elif action_t == 3:
            action = RIGHT
        else:
            print("ERROR")
        obs, reward_, done, remaining_lives = game.step(action)
        # display.obs = obs.copy()
        update_all = False
        if remaining_lives < lives:
            lives -= 1
            jump_dead_step = True
            got_reward = False
            reward_ += -100
            # dmaker.old_action = 3
            update_all = True

        if done and lives > 0:
            reward += +100

        got_reward = reward_ != 0
        # display.data.rewards.append(reward)
        # reward = torch.tensor([reward], device=device)

        old_action = action
        # if reward != 0:
        #     dmaker.old_action = action.item()
        next_state = preprocess_observation(obs)
        if got_reward:
            action_tensor = torch.tensor([action_t], device=device, dtype=torch.long)
            memory.append(state,action_tensor,
                          reward_, next_state, done)

        state = next_state
        learn_counter = optimize_model(
            policy_DQN,
            target_DQN,
            memory,
            optimizer,
            learn_counter,
            device,
        )
            # predicted_targets = policy_DQN(
            #     state_batch[0])

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
