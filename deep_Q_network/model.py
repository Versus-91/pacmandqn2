from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T
import torch

from .parameters import BATCH_SIZE, DISCOUNT_RATE, device


def my_rely(x):
    return torch.maximum(x, torch.zeros_like(x))


class DQN(nn.Module):

    def __init__(self, outputs):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(in_features=32 * 8 * 8, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def optimize_model(policy_DQN, target_DQN, memory, optimizer, display, learn_counter, device):
    if len(memory) < BATCH_SIZE:
        return learn_counter
    learn_counter += 1
    states, actions, rewards, next_states, dones = memory.sample()

    predicted_targets = policy_DQN(states).gather(1, actions)

    target_values = target_DQN(next_states).detach().max(1)[0]
    labels = rewards + DISCOUNT_RATE * (1 - dones.squeeze(1)) * target_values

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(predicted_targets,
                     labels.detach().unsqueeze(1)).to(device)
    display.data.losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    for param in policy_DQN.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    # # Softmax update
    # for target_param, local_param in zip(target_DQN.parameters(), policy_DQN.parameters()):
    #     target_param.data.copy_(TAU * local_param.data + (1 - TAU) * target_param.data)

    return learn_counter
