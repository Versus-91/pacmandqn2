import os
import numpy as np
import torch.optim as optim
import torch
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import time
from cnn import Conv2dNetwork
from game import GameWrapper
import random
import matplotlib
import torch.optim.lr_scheduler as lr_scheduler

from run import GameState

matplotlib.use("Agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTIONS = 4
BATCH_SIZE = 128
SAVE_EPISODE_FREQ = 100
GAMMA = 0.99
MOMENTUM = 0.95
MEMORY_SIZE = 20000
LEARNING_RATE = 0.0005

Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)

REVERSED = {0: 1, 1: 0, 2: 3, 3: 2}
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 200000
MAX_STEPS = 400000


class ExperienceReplay:
    def __init__(self, capacity) -> None:
        self.exps = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        self.exps.append(Experience(state, action, reward, done, next_state))

    def sample(self, batch_size):
        return random.sample(self.exps, batch_size)

    def __len__(self):
        return len(self.exps)


class PacmanAgent:
    def __init__(self):
        self.steps = 0
        self.score = 0
        self.target = Conv2dNetwork().to(device)
        self.policy = Conv2dNetwork().to(device)
        self.memory = ExperienceReplay(MEMORY_SIZE)
        self.game = GameWrapper()
        self.last_action = 0
        self.buffer = deque(maxlen=6)
        self.last_reward = -1
        self.rewards = []
        self.loop_action_counter = 0
        self.score = 0
        self.episode = 0
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)

    def calculate_reward(
        self, done, lives, hit_ghost, action, prev_score, info: GameState
    ):
        reward = 0
        if done:
            if lives > 0:
                print("won")
                reward = 30
            else:
                reward = -30
            return reward
        if self.score - prev_score == 10:
            reward += 10
        if self.score - prev_score == 50:
            print("power up")
            reward += 14
        if reward > 0:
            progress = (info.collected_pellets / info.total_pellets) * 7
            reward += progress
            return reward
        if self.score - prev_score >= 200:
            return 16
        if info.invalid_move:
            reward -= 6
        if hit_ghost:
            reward -= 20
            return reward
        return reward

    def write_matrix(self, matrix):
        with open('outfile.txt', 'wb') as f:
            for line in matrix:
                np.savetxt(f, line, fmt='%.2f')

    def evaluate(self):
        if len(self.memory) < BATCH_SIZE:
            return
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        new_state_batch = torch.cat(batch.new_state)
        reward_batch = torch.cat(batch.reward)
        dones = torch.tensor(batch.done, dtype=torch.float32).to(device)
        predicted_targets = self.policy(state_batch).gather(1, action_batch)
        target_values = self.target(new_state_batch).detach().max(1)[0]
        labels = reward_batch + GAMMA * (1 - dones) * target_values

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(predicted_targets,
                         labels.detach().unsqueeze(1)).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.steps % 100 == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def act(self, state, eval=False):
        if eval:
            with torch.no_grad():
                q_values = self.policy(state)
            vals = q_values.max(1)[1]
            return vals.view(1, 1)
        rand = random.random()
        epsilon = max(
            EPS_END, EPS_START - (EPS_START - EPS_END) *
            (self.steps / 2) / EPS_DECAY
        )
        self.steps += 1
        if rand > epsilon:
            with torch.no_grad():
                outputs = self.policy(state)
            return outputs.max(1)[1].view(1, 1)
        else:
            # Random action
            action = random.randrange(N_ACTIONS)
            while action == REVERSED[self.last_action]:
                action = random.randrange(N_ACTIONS)
            return torch.tensor([[action]], device=device, dtype=torch.long)

    def plot_rewards(self, name="plot.png", avg=100):
        plt.figure(1)
        durations_t = torch.tensor(self.rewards, dtype=torch.float)
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
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

    def save_model(self, force=False):
        if (self.episode % SAVE_EPISODE_FREQ == 0 and self.episode != 0) or force:
            torch.save(
                self.policy.state_dict(),
                os.path.join(
                    os.getcwd() + "\\results",
                    f"policy-model-{self.episode}-{self.steps}.pt",
                ),
            )
            torch.save(
                self.target.state_dict(),
                os.path.join(
                    os.getcwd() + "\\results",
                    f"target-model-{self.episode}-{self.steps}.pt",
                ),
            )

    def load_model(self, name, eval=False):
        path = os.path.join(os.getcwd() + "\\results",
                            f"target-model-{name}.pt")
        self.target.load_state_dict(torch.load(path))
        path = os.path.join(os.getcwd() + "\\results",
                            f"policy-model-{name}.pt")
        self.policy.load_state_dict(torch.load(path))
        if eval:
            self.target.eval()
            self.policy.eval()
        else:
            name_parts = name.split("-")
            self.episode = int(name_parts[0])
            self.steps = int(name_parts[1])
            self.target.train()
            self.policy.train()

    def train(self):
        if self.steps >= MAX_STEPS:
            self.save_model(force=True)
            exit()
        self.save_model()
        obs = self.game.start()
        self.episode += 1
        random_action = random.choice([0, 1, 2, 3])
        # obs, self.score, done, info = self.game.step(random_action)
        # state = self.process_state(obs)
        # state = torch.tensor(obs).float().to(device)
        for i in range(6):
            obs, self.score, done, info = self.game.step(random_action)
            self.buffer.append(obs)
        state = self.process_state(self.buffer)
        last_score = 0
        lives = 3
        while True:
            action = self.act(state)
            action_t = action.item()
            for i in range(3):
                if not done:
                    obs, self.score, done, info = self.game.step(action_t)
                    if lives != info.lives or self.score - last_score != 0:
                        break
                else:
                    break
            self.buffer.append(obs)
            hit_ghost = False
            if lives != info.lives:
                # self.write_matrix(self.buffer)
                hit_ghost = True
                lives -= 1
            # next_state = torch.tensor(obs).float().to(device)
            next_state = self.process_state(self.buffer)

            reward_ = self.calculate_reward(
                done, lives, hit_ghost, action_t, last_score, info
            )
            last_score = self.score
            action_tensor = torch.tensor(
                [[action_t]], device=device, dtype=torch.long)
            self.memory.append(
                state, action_tensor, torch.tensor(
                    [reward_], device=device), next_state, done
            )
            state = next_state
            self.evaluate()
            self.last_action = action_t
            if self.steps % 100000 == 0:
                self.scheduler.step()
            if done:
                current_lr = self.optimizer.param_groups[0]["lr"]
                epsilon = max(
                    EPS_END,
                    EPS_START - (EPS_START - EPS_END) *
                    (self.steps / 2) / EPS_DECAY,
                )
                print(
                    "epsilon",
                    round(epsilon, 3),
                    "reward",
                    self.score,
                    "learning rate",
                    current_lr,
                    "episode",
                    self.episode,
                    "steps",
                    self.steps
                )
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
            for i in range(6):
                obs, self.score, done, info = self.game.step(random_action)
                self.buffer.append(obs)
            state = self.process_state(self.buffer)
            while True:
                action = self.act(state, eval=True)
                action_t = action.item()
                for i in range(3):
                    if not done:
                        obs, reward, done, _ = self.game.step(action_t)
                    else:
                        break
                self.buffer.append(obs)
                state = self.process_state(self.buffer)
                if done:
                    self.rewards.append(reward)
                    self.plot_rewards(name="test.png", avg=2)
                    time.sleep(1)
                    self.game.restart()
                    torch.cuda.empty_cache()
                    break
        else:
            exit()


if __name__ == "__main__":
    agent = PacmanAgent()
    agent.load_model(name="700-314695", eval=True)
    agent.rewards = []
    while True:
        # agent.train()
        agent.test()
