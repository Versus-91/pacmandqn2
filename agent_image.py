# create a dqn eperience replay buffer
from math import log
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
# matplotlib.use('Agg')
K_FRAME = 2
def optimization(it, r): return it % K_FRAME == 0 and r


REVERSED = {0: 1, 1: 0, 2: 3, 3: 2}
isreversed = (
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
BATCH_SIZE = 128
DISCOUNT_RATE = 0.99
EPS_MAX = 1.0
EPS_MIN = 0.1
EPS_DECAY = 1_000_000
TARGET_UPDATE = 8_000  # here
REPLAY_MEMORY_SIZE = 3 * 6000
steps_done = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learn_counter = 0
N_ACTIONS = 4
TARGET_UPDATE = 8_000  # here


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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def optimize_model(policy_DQN, target_DQN, memory, optimizer, learn_counter, device):
    if len(memory) < BATCH_SIZE:
        return learn_counter
    learn_counter += 1
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

    return learn_counter


policy_DQN = DQN(588, N_ACTIONS).to(device)
target_DQN = DQN(588, N_ACTIONS).to(device)
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
    eps_threshold = max(EPS_MIN, EPS_MAX - (EPS_MAX - EPS_MIN) / EPS_DECAY)
    steps_done += 1
    with torch.no_grad():
        q_values = policy_DQN(state)
    # display.data.q_values.append(q_values.max(1)[0].item())
    if sample > eps_threshold:
        # Optimal action
        return q_values.max(1)[1].view(1, 1)
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
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        batch = Experience(*zip(*transitions))
        return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.buffer)


memory = ExperienceReplay(18000)
game = GameWrapper()
episode_durations = []
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


def plot_durations():
    plt.figure()
    plt.plot(np.arange(len(episode_durations)), episode_durations)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards over Time')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig('plot.png')


recod_score = 0
obs = game.start()
# Main loop
frame = 0


def process_image(observation):
    smaller_img = observation[70:380:2, 95:450:2]
    gray_img = cv2.cvtColor(smaller_img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray_img, (44, 40), interpolation=cv2.INTER_AREA)
    return resized.transpose(1, 0)


def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


while True:
    # Set the desired frame rate
    # Set the desired action interval
    action_interval = 0.016  # 1 second
    start_time = time.time()
    frames = []
    episodes += 1
    lives = 3
    jump_dead_step = False
    obs, reward, done, info = game.step(2)
    state = torch.from_numpy(obs[0].astype(dtype=np.float32)).to(device)
    smaller_img = obs[70:380:2, 95:450:2]
    frames.append(smaller_img.transpose(1, 0, 2))
    got_reward = False
    skipped_frames = 0
    old_action = 3
    reward_sum = 0
    last_score = 0
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= action_interval:
            obs, reward, done, info = game.step(
                random.choice([0,1,2,3]))
            if last_score < reward:
                reward_sum += reward - last_score
            last_score = reward
            image = process_image(obs)
            frames.append(image)
            if len(frames) % 16 == 0:
                plot_images(frames, rows=4, cols=4)
            start_time = time.time()
            print("action", skipped_frames)
            skipped_frames = 0
        else:
            skipped_frames += 1
            print("no action", skipped_frames)
        if done:
            assert reward_sum == reward
            game.restart()
            time.sleep(3)
            reward_sum = 0
        # obs, reward, done, info = game.step(random.choice([UP, DOWN, LEFT, RIGHT]))
        # frames.append(obs.transpose(1, 0, 2))
        # if len(frames) % 16 == 0:
        #     plot_images(frames, rows=4, cols=4)
        # current_time = time.time()
        # if not current_time - start_time >= 0.08:
        #     continue
        # start_time = time.time()
        # action = select_action(state, policy_DQN)
        # action_t = action.item()
        # if action_t == 0:
        #     action = UP
        # elif action_t == 1:
        #     action = DOWN
        # elif action_t == 2:
        #     action = LEFT
        # elif action_t == 3:
        #     action = RIGHT
        # else:
        #     print("ERROR")
        # obs, reward_, done, remaining_lives = game.step(action)
        # reward = transform_reward(reward_)
        # update_all = False
        # if remaining_lives < lives:
        #     lives -= 1
        #     jump_dead_step = True
        #     got_reward = False
        #     reward += REWARDS["lose"]
        #     update_all = True

        # if done and lives > 0:
        #     reward += REWARDS["win"]

        # got_reward = got_reward or reward != 0

        # old_action = action_t
        # next_state = torch.from_numpy(
        #     obs[0].astype(dtype=np.float32)).to(device)
        # next_state = next_state.view(1, -1)
        # if got_reward:
        #     reward_sum += reward_
        #     reward = torch.tensor([reward], device=device)
        #     action_tensor = torch.tensor(
        #         [[action_t]], device=device, dtype=torch.long)
        #     memory.append(state, action_tensor,
        #                   reward, next_state, done)

        # state = next_state
        # if got_reward and steps_done % 2 == 0:
        #     optimize_model(
        #         policy_DQN,
        #         target_DQN,
        #         memory,
        #         optimizer,
        #         learn_counter,
        #         device
        #     )

        # if steps_done % TARGET_UPDATE == 0:
        #     target_DQN.load_state_dict(policy_DQN.state_dict())
        # if done:
        #     if reward_sum > recod_score:
        #         recod_score = reward_sum
        #         print("recod_score", recod_score)
        #     episode_durations.append(reward_sum)
        #     torch.cuda.empty_cache()
        #     plot_durations()
        #     game.restart()
        #     time.sleep(3)
        #     break
        # if jump_dead_step:
        #     time.sleep(1)
        #     jump_dead_step = False
