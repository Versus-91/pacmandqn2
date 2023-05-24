import torch
import torch.nn as nn

# Define the NN model architecture


class ThreeMatrixNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ThreeMatrixNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


# Example usage
# Replace with the actual sizes of your input matrices
input_size = 22 * 18 * 3
input_shape = 22 * 18
hidden_size = 64  # Define the number of units in the hidden layer
num_classes = 4  # Replace with the actual number of output classes in your Pac-Man game

model = ThreeMatrixNN(input_size, hidden_size, num_classes)

# Use the model for forward pass
# Replace with your first input matrix
input_matrix1 = torch.randn(1, input_shape)
# Replace with your second input matrix
input_matrix2 = torch.randn(1, input_shape)
# Replace with your third input matrix
input_matrix3 = torch.randn(1, input_shape)
data = torch.cat((input_matrix1.view(input_matrix1.size(0), -1),
               input_matrix2.view(input_matrix2.size(0), -1), input_matrix3.view(input_matrix3.size(0), -1)), dim=1)
output = model(data)
print(output)
