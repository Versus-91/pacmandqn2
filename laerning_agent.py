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
from cnn import NeuralNetwork
from constants import *
from game import GameWrapper
import random
import matplotlib
from time import sleep
matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTIONS = 4
BATCH_SIZE = 128
TARGET_UPDATE = 60  # here
K_FRAME = 2
SAVE_EPISODE_FREQ = 100
def optimization(it, r): return it % K_FRAME == 0 and r


Experience = namedtuple('Experience', field_names=[
                        'state', 'action', 'reward', 'done', 'new_state'])

REVERSED = {0: 1, 1: 0, 2: 3, 3: 2}
is_reversed = (
    lambda last_action, action: "default" if REVERSED[action] -
    last_action else "reverse"
)


class ExperienceReplay:
    def __init__(self, capacity) -> None:
        self.buffer = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, done, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


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


class LearningAgent:
    def __init__(self):
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.gamma = 0.99
        self.momentum = 0.95
        self.replay_size = 80000
        self.learning_rate = 0.0003
        self.steps = 0
        self.target = NeuralNetwork().to(device)
        self.policy = NeuralNetwork().to(device)
        # self.load_model()
        self.memory = ExperienceReplay(self.replay_size)
        self.game = GameWrapper()
        self.last_action = 0
        self.last_reward = -1
        self.rewards = []
        state_buffer = deque(maxlen=4)
        self.episode = 0
        self.optimizer = optim.SGD(
            self.policy.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=True
        )

    def calculate_distance(pos1, pos2):
        # pos1 and pos2 are tuples representing positions (x, y)
        x1, y1 = pos1
        x2, y2 = pos2
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance

    def calculate_reward(self, done, lives, eat_pellet, eat_powerup, hit_wall, hit_ghost, ate_ghost):

        # Initialize reward
        reward = 0

        # Check if the game is won or lost
        if done:
            if lives > 0:
                reward = 10  # Game won
            else:
                reward = -10  # Game lost
            return reward

        # Calculate the distance to the nearest food pellet
        # current_pos = state.get_pacman_position()
        # nearest_pellet = state.get_nearest_pellet_position()
        # distance_to_pellet = calculate_distance(current_pos, nearest_pellet)

        # Check if Pacman ate a pellet
        if eat_pellet:
            reward += 1  # Pacman ate a pellet
        if eat_powerup:
            reward += 3  # Pacman ate a power pellet

        # Encourage Pacman to move towards the nearest pellet
        # reward -= distance_to_pellet

        # Penalize Pacman for hitting walls or ghosts
        if hit_wall:
            print("hit the wall")
            reward -= 1  # Pacman hit a wall
        if hit_ghost:
            reward -= 5  # Pacman hit a ghost
        if ate_ghost:
            reward += 3
        self.last_reward = reward
        if self.last_reward == 0 and reward == 0:
            reward = -0.5
        return reward

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        experiences = self.memory.sample(BATCH_SIZE)
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
        predicted_targets = self.policy(state_batch).gather(1, action_batch)
        target_values = self.target(new_state_batch).detach().max(1)[0]
        labels = reward_batch + self.gamma * \
            (1 - dones.squeeze(1)) * target_values

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(predicted_targets,
                         labels.detach().unsqueeze(1)).to(device)
        # display.data.losses.append(loss.item())
        # print("loss", loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.policy.state_dict())
        # # Softmax update
        # for target_param, local_param in zip(target_DQN.parameters(), policy_DQN.parameters()):
        #     target_param.data.copy_(TAU * local_param.data + (1 - TAU) * target_param.data)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps / self.eps_decay)
        self.steps += 1
        # display.data.q_values.append(q_values.max(1)[0].item())
        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy(state)
            # Optimal action
            vals = q_values.max(1)[1]
            return vals.view(1, 1)
        else:
            # Random action
            action = random.randrange(N_ACTIONS)
            while action == REVERSED[self.last_action]:
                action = random.randrange(N_ACTIONS)
            return torch.tensor([[action]], device=device, dtype=torch.long)

    def plot_rewards(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.rewards, dtype=torch.float)
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        plt.savefig('plot.png')

    def process_state(self, states):
        pallets_tensor = torch.from_numpy(states[0]).float().to(device)
        pacman_tensor = torch.from_numpy(states[1]).float().to(device)
        pellets_tensor = torch.from_numpy(states[2]).float().to(device)
        powerpellets_tensor = torch.from_numpy(states[3]).float().to(device)
        ghosts_tensor = torch.from_numpy(states[4]).float().to(device)
        frightened_ghosts_tensor = torch.from_numpy(
            states[5]).float().to(device)
        channel_matrix = torch.stack([pallets_tensor, pacman_tensor, pellets_tensor,
                                     powerpellets_tensor, ghosts_tensor, frightened_ghosts_tensor], dim=0)
        channel_matrix = channel_matrix.unsqueeze(0)
        return channel_matrix

    def save_model(self):
        if self.episode % SAVE_EPISODE_FREQ == 0:
            torch.save(self.policy.state_dict(), os.path.join(
                os.getcwd() + "\\results", f"policy-model-{self.episode}-{self.steps}.pt"))
            torch.save(self.target.state_dict(), os.path.join(
                os.getcwd() + "\\results", f"target-model-{self.episode}-{self.steps}.pt"))

    def load_model(self, name="200-44483"):
        self.steps = 44483
        path = os.path.join(
            os.getcwd() + "\\results", f"target-model-{name}.pt")
        self.target.load_state_dict(torch.load(path))
        self.target.train()
        path = os.path.join(
            os.getcwd() + "\\results", f"policy-model-{name}.pt")
        self.policy.load_state_dict(torch.load(path))
        self.policy.train()

    def train(self):
        self.save_model()
        obs = self.game.start()
        action_interval = 0.02
        start_time = time.time()
        self.episode += 1
        lives = 3
        obs, reward, done, info, invalid_move = self.game.step(2)
        # obs = obs[0].flatten().astype(dtype=np.float32)
        # state = torch.from_numpy(obs).unsqueeze(0).to(device)
        state = self.process_state(obs)
        reward_sum = 0
        last_score = 0
        pellet_eaten = 0
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= action_interval:
                action = self.select_action(state)
                action_t = action.item()
                obs, reward, done, remaining_lives, invalid_move = self.game.step(
                    action_t)
                hit_ghost = False
                if lives != remaining_lives:
                    hit_ghost = True
                    lives -= 1
                next_state = self.process_state(obs)
                reward_ = self.calculate_reward(
                    done, lives, reward - last_score == 10, reward - last_score == 50, invalid_move, hit_ghost, reward - last_score >= 200)
                if last_score < reward:
                    reward_sum += reward - last_score
                last_score = reward
                action_tensor = torch.tensor(
                    [[action_t]], device=device, dtype=torch.long)
                self.memory.append(state, action_tensor,
                                   torch.tensor([reward_], device=device), next_state, done)
                state = next_state
                if self.steps % 2 == 0:
                    self.optimize_model()

                start_time = time.time()
            elif elapsed_time < action_interval:
                self.game.update()
            if done:
                assert reward_sum == reward
                self.rewards.append(reward_sum)
                self.plot_rewards()
                time.sleep(1)
                self.game.restart()
                reward_sum = 0
                torch.cuda.empty_cache()
                break


if __name__ == '__main__':
    agetnt = LearningAgent()
    while True:
        agetnt.train()
