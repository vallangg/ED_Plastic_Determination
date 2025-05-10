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

    def __init__(self):
        super().__init__() # call the parent init fucntion
        input_size = 73
        hidden_size = 100
        output_size = 5

        self.layer1 = nn.Linear(input_size, hidden_size) # first layer goes from 73 inputs to 100 hidden

        self.layer2 = nn.Linear(hidden_size, output_size) # this goes from the 100 hidden to 4 output


    def forward(self, x): 
        """
            This function will act as a helper to pass information forward to the further layers
            :param x: this is the data that will pass through the neural netword
        """
        x = F.relu(self.layer1(x)) # apply the relu activation function to the hidden layer

        x = self.layer2(x) # outputs the last layer after the actiuvation layer

        return x


