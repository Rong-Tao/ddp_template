import torch
import torch.nn as nn
import torch.nn.functional as F

### Define your model here ###
class Model_Class(nn.Module):
    def __init__(self):
        super(Model_Class, self).__init__(img_size = 512)
        # Define the layers
        self.cnn = nn.Conv2d(3,1,3,padding='same')

    def forward(self, x):
        return self.cnn(x)
