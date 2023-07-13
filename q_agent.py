from math import log
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
import numpy as np
from cnn import *
from constants import *
from game import GameWrapper
import random
import matplotlib
from state import GameState
from tensorboardX import SummaryWriter

matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTIONS = 4
BATCH_SIZE = 128
SAVE_EPISODE_FREQ = 100
GAMMA = 0.99
MOMENTUM = 0.95
MEMORY_SIZE = 30000
LEARNING_RATE = 0.0003

Experience = namedtuple('Experience', field_names=[
                        'state', 'action', 'reward', 'done', 'new_state'])

REVERSED = {0: 1, 1: 0, 2: 3, 3: 2}
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 450000
MAX_STEPS = 900000


class ExperienceReplay:
    def __init__(self, capacity) -> None:
        self.exps = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        self.exps.append(Experience(state, action, reward, done, next_state))

    def sample(self, batch_size):
        return random.sample(self.exps, batch_size)

    def __len__(self):
        return len(self.exps)


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

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(32 * 21 * 28, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


class PacmanAgent:
    def __init__(self):
        self.steps = 0
        self.score = 0
        self.target = QNetwork().to(device)
        self.policy = QNetwork().to(device)
        # self.load_model()
        self.memory = ExperienceReplay(MEMORY_SIZE)
        self.game = GameWrapper()
        self.last_action = 0
        self.buffer = deque(maxlen=4)
        self.last_reward = -1
        self.rewards = []
        self.loop_action_counter = 0
        self.counter = 0
        self.score = 0
        self.episode = 0
        self.writer = SummaryWriter('oldlogs')
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=LEARNING_RATE
        )
        self.prev_info =GameState()

    def get_reward(self, done, lives, hit_ghost, action, prev_score,info:GameState):
        reward = 0
        if done:
            if lives > 0:
                print("won")
                reward = 30
            else:
                reward = -30
            return reward
        progress =  int((info.collected_pellets / info.total_pellets) * 7)
        if self.score - prev_score == 10:
            reward += 10
        if self.score - prev_score == 50:
            reward += 13
            if info.ghost_distance != -1 and info.ghost_distance < 10:
                reward += 3
        if reward > 0:
            reward += progress
            return reward
        if self.score - prev_score >= 200:
            return 16 + (self.score - prev_score // 200) * 2
        if hit_ghost:
            reward -= 20
        if (info.ghost_distance >=1 and info.ghost_distance < 5):
            if  self.prev_info.ghost_distance >= info.ghost_distance:
                reward += 3
            elif self.prev_info.ghost_distance < info.ghost_distance:
                reward -= 3
            return reward
        if self.prev_info.food_distance >= info.food_distance and info.food_distance != -1:
                reward += 3
        elif self.prev_info.food_distance < info.food_distance and info.food_distance != -1:
            reward -= 2
        if info.scared_ghost_distance < 8 and self.prev_info.scared_ghost_distance >= info.scared_ghost_distance and info.scared_ghost_distance != -1:
            reward += 3
        if not (info.ghost_distance >=1 and info.ghost_distance < 5):
            if action == REVERSED[self.last_action] and not info.invalid_move:
                reward -= 2
        if info.invalid_move:
            reward -= 8     
            print("invalid act",reward)
        reward -= 1            
        return reward
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        self.counter += 1
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        new_state_batch = torch.cat(batch.new_state)
        reward_batch = torch.cat(batch.reward)
        dones = torch.tensor(batch.done, dtype=torch.float32).to(device)
        predicted_targets = self.policy(state_batch).gather(1, action_batch)
        target_values = self.target(new_state_batch).detach().max(1)[0]
        labels = reward_batch + GAMMA * \
            (1 - dones) * target_values

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(predicted_targets,
                         labels.detach().unsqueeze(1)).to(device)
        self.writer.add_scalar('loss', loss.item(), global_step=self.steps)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.steps % 10 == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def select_action(self, state, eval=False):
        if eval:
            with torch.no_grad():
                q_values = self.policy(state)
            vals = q_values.max(1)[1]
            return vals.view(1, 1)
        rand = random.random()
        epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END)
                      * self.counter / EPS_DECAY)
        self.steps += 1
        if rand > epsilon:
            with torch.no_grad():
                q_values = self.policy(state)
            vals = q_values.max(1)[1]
            return vals.view(1, 1)
        else:
            # Random action
            action = random.randrange(N_ACTIONS)
            while action == REVERSED[self.last_action] or self.game.get_invalid_action(action):
                action = random.randrange(N_ACTIONS)
            return torch.tensor([[action]], device=device, dtype=torch.long)

    def plot_rewards(self, name="plot.png", avg=100):
        plt.figure(1)
        durations_t = torch.tensor(self.rewards, dtype=torch.float)
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= avg:
            means = durations_t.unfold(0, avg, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(avg - 1), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        plt.savefig(name)

    def process_state(self, states):

        tensor = [torch.from_numpy(arr).float().to(device) for arr in states]

        # frightened_ghosts_tensor = torch.from_numpy(
        #     states[3]).float().to(device)
        channel_matrix = torch.stack(tensor, dim=0)
        channel_matrix = channel_matrix.unsqueeze(0)
        return channel_matrix

    def save_model(self):
        if self.episode % SAVE_EPISODE_FREQ == 0 and self.episode != 0:
            torch.save(self.policy.state_dict(), os.path.join(
                os.getcwd() + "\\results", f"policy-model-{self.episode}-{self.steps}.pt"))
            torch.save(self.target.state_dict(), os.path.join(
                os.getcwd() + "\\results", f"target-model-{self.episode}-{self.steps}.pt"))

    def load_model(self, name, eval=False):
        name_parts = name.split("-")
        self.episode = int(name_parts[0])
        self.steps = int(name_parts[1])
        self.counter = int(self.steps / 2)
        path = os.path.join(
            os.getcwd() + "\\results", f"target-model-{name}.pt")
        self.target.load_state_dict(torch.load(path))
        path = os.path.join(
            os.getcwd() + "\\results", f"policy-model-{name}.pt")
        self.policy.load_state_dict(torch.load(path))
        if eval:
            self.target.eval()
            self.policy.eval()
        else:
            self.target.train()
            self.policy.train()
    def pacman_pos(self,state):
        index = np.where(state != 0)
        if len(index[0]) != 0:
            x = index[0][0]
            y = index[1][0]
            return (x,y)
        return None
    def train(self):
        if self.steps >= MAX_STEPS:
            return
        self.save_model()
        obs = self.game.start()
        self.episode += 1
        random_action = random.choice([0, 1, 2, 3])
        obs, self.score, done, info = self.game.step(
            random_action)
        state = self.process_state(obs)
        last_score = 0
        lives = 3
        pacman_pos = self.pacman_pos(obs[2])
        while True:
            action = self.select_action(state)
            action_t = action.item()
            while True:
                if not done:
                    obs, self.score, done, info = self.game.step(
                        action_t)
                    pacman_pos_new = self.pacman_pos(obs[2])
                    if pacman_pos_new != pacman_pos or  lives != info.lives or info.invalid_move:
                        pacman_pos = pacman_pos_new
                        break
                else:
                    break
            hit_ghost = False
            if lives != info.lives:
                hit_ghost = True
                lives -= 1
            next_state = self.process_state(obs)
            reward_ = self.get_reward(done, lives, hit_ghost, action_t, last_score, info)
            self.prev_info = info
            last_score = self.score
            self.memory.append(state, action,torch.tensor([reward_], device=device), next_state, done)
            state = next_state
            if self.steps % 2 == 0:
                self.optimize_model()
            self.last_action = action_t
            if done:

                epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END)* self.counter / EPS_DECAY)
                print("epsilon: ",round(epsilon,2),"reward: ",self.score,"steps: ",self.steps,
                      "completion: ",round((info.collected_pellets / info.total_pellets)*100,2)
                      ,"spisode",self.episode)
                self.writer.add_scalar('episode reward', self.score, global_step=self.episode)
                # assert reward_sum == reward
                self.rewards.append(self.score)
                self.plot_rewards(avg=50)
                time.sleep(1)
                self.game.restart()
                torch.cuda.empty_cache()
                break

    def test(self, episodes=10):
        if self.episode < episodes:
            obs = self.game.start()
            self.episode += 1
            random_action = random.choice([0, 1, 2, 3])
            obs, reward, done, _ = self.game.step(
                random_action)
            state = self.process_state(obs)
            pacman_pos = self.pacman_pos(obs[2])
            lives = 3
            while True:
                action = self.select_action(state, eval=True)
                action_t = action.item()
                while True:
                    if not done:
                        obs, self.score, done, info = self.game.step(
                            action_t)
                        pacman_pos_new = self.pacman_pos(obs[2])
                        if pacman_pos_new != pacman_pos or  lives != info.lives or info.invalid_move:
                            pacman_pos = pacman_pos_new
                            break
                    else:
                        break
                lives != info.lives
                state = self.process_state(obs)
                if done:
                    self.rewards.append(reward)
                    self.plot_rewards(name="test.png", avg=2)
                    time.sleep(1)
                    self.game.restart()
                    torch.cuda.empty_cache()
                    break
        else:
            self.game.stop()


if __name__ == '__main__':
    agent = PacmanAgent()
    #agent.load_model(name="1900-599094", eval=False)
    agent.rewards = []
    while True:
        agent.train()
        #agent.test()
