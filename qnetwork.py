# create a dqn eperience replay buffer
from collections import deque, namedtuple
import random
import time
import cv2
import numpy as np
from constants import *
from game import GameWrapper
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import log
K_FRAME = 2
def optimization(it, r): return it % K_FRAME == 0 and r


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
    # Crop and resize the image
    obs = np.flipud(obs).transpose((1, 0, 2))
    res = cv2.resize(obs, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
    crop_img = res[15:190, 0:200]
    # Convert the image to greyscale
    # crop_img = crop_img.mean(axis=2)
    # Improve image contrast
    # crop_img[crop_img == color] = 0
    res = cv2.resize(crop_img, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res = cv2.merge((res))
    res = np.transpose(res, (2, 0, 1))
    # plt.imshow(res)
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
    labels = reward_batch + GAMMA * (1 - dones.squeeze(1)) * target_values

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(predicted_targets,
                     labels.detach().unsqueeze(1)).to(device)
    # display.data.losses.append(loss.item())
    #print("loss", loss.item())
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
episodes = 0


def transform_reward(reward):
    return log(reward, 1000) if reward > 0 else reward


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
        return random.sample(self.buffer, batch_size)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        batch = Experience(*zip(*transitions))
        return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.buffer)


memory = ExperienceReplay(10000)
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


def plot_durations(show_result=False):
    plt.plot(np.arange(len(episode_durations)), episode_durations)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards over Time')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


obs = game.starz()
# Main loop
while True:
    # if dmaker.steps_done > MAX_FRAMES:
    #     break

    episodes += 1
    lives = 3
    jump_dead_step = False
    old_action = 0
    obs, reward, done, info = game.step(random.choice([UP, DOWN, LEFT, RIGHT]))
    state = preprocess_observation(obs)
    got_reward = False
    old_action = 3
    reward_sum = 0
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
        reward = transform_reward(reward_)
        update_all = False
        if remaining_lives < lives:
            lives -= 1
            jump_dead_step = True
            got_reward = False
            reward += REWARDS["lose"]
            # dmaker.old_action = 3
            update_all = True

        if done and lives > 0:
            reward += REWARDS["win"]

        got_reward = got_reward or reward != 0
        # display.data.rewards.append(reward)
        # display.data.rewards.append(reward)
        # reward = torch.tensor([reward], device=device)

        old_action = action
        # if reward != 0:
        #     dmaker.old_action = action.item()
        next_state = preprocess_observation(obs)
        if got_reward:
            reward_sum += reward
            reward = torch.tensor([reward], device=device)
            action_tensor = torch.tensor(
                [[action_t]], device=device, dtype=torch.long)
            memory.append(state, action_tensor,
                          reward, next_state, done)

        state = next_state
        if got_reward:
            optimize_model(
                policy_DQN,
                target_DQN,
                memory,
                optimizer,
                learn_counter,
                device
            )
        # predicted_targets = policy_DQN(
        #     state_batch[0])

        if steps_done % TARGET_UPDATE == 0:
            target_DQN.load_state_dict(policy_DQN.state_dict())
        # display.stream(update_all)
        if done:
            # display.data.successes += remaining_lives > 0
            print("done", reward_sum)
            torch.cuda.empty_cache()
            game.restart()
            time.sleep(3)
            # plot_durations()
            break
        if jump_dead_step:
            time.sleep(1)
            jump_dead_step = False
        torch.cuda.empty_cache()

    # if episodes % 1000 == 0:
    #     # torch.save(policy_DQN.state_dict(), PATH_MODELS /
    #     #            f"policy-model-{episodes}.pt")
    #     # torch.save(target_DQN.state_dict(), PATH_MODELS /
    #     #            f"target-model-{episodes}.pt")
    #     display.save()
    # display.data.round()
    # torch.save(policy_DQN.state_dict(), PATH_MODELS / f"policy-model-final.pt")
    # torch.save(target_DQN.state_dict(), PATH_MODELS / f"target-model-final.pt")
    # print("Complete")
