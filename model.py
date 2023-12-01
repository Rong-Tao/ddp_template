import torch
import torch.nn as nn
import torch.nn.functional as F

### Define your model here ###
class Model_Class(nn.Module):
    def __init__(self):
        super(Model_Class, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(32 * 32 * 3, 512) # Assuming input size of CIFAR-10 images (32x32x3)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10) # Assuming 10 classes for output

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, 32 * 32 * 3)
        # Apply layers with ReLU activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # Output layer with no activation (or LogSoftmax if needed)
        x = self.fc4(x)
        return x
