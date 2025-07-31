"""
This is th emain AI model of the program. It will use a ANN to determine the microplastic content of samples using ED data
"""

## Import the Libraries

import torch
import torch.nn as nn # import the neural network package
import torch.optim as optim # import the optimizer function
import torch.nn.functional as F # import this helper function
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



class Brain(nn.Module):
    """
        This is the code to create the linear neural network model using pytorch.
        This code is largely taken from the pytorch website. 
        https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
    """

    def __init__(self, input_size: int = 73, hidden_size: int = 100, output_size: int = 4):
        super(Brain, self).__init__() # call the parent init fucntion
        input_size = 73
        hidden_size = 200
        output_size = 5

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=3)
        self.normalize = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)



    def forward(self, x): 
        """
            This function will act as a helper to pass information forward to the further layers
            :param x: this is the data that will pass through the neural netword
        """
        # print(f"Shape of x: {x.shape}")
        x = x.unsqueeze(1)
        # print(f"unsqueezed shape of x: {x.shape}")
        x = F.relu(self.conv1(x))
        x = self.normalize(x) # normalize the results of the convolutional layer
        x = torch.mean(x, dim = 2) # this flattens the tensor so it can be given to the fully connected layers
        x = F.relu(self.fc1(x)) # push through the first fully connected layer
        x = F.relu(self.fc2(x)) # push through the second fully connected layer

        return x
