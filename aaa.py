import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
class NeuralNetwork(nn.Module):
    def __init__(self, n_actions):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(588, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Creating an instance of the neural network

# Creating a random input tensor with shape (1, 1, 21, 28)
item = np.zeros((21,28))
item = item.flatten()
print(item.shape)
input_tensor = torch.tensor(item, dtype=torch.float32)
print(input_tensor.shape)
model = NeuralNetwork(4)
# Forward pass through the network
output = model(input_tensor)

# Printing the output shape
print(output.shape)
