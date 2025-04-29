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
        self.flatten = nn.Flatten()
        input_size = 73
        hidden_size = 100
        output_size = 4


        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),   # hidden layer with 73 iputs--the number of observations  for the ED angle sweep
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),   #Hidden layer
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), # second hidden layer
            nn.ReLU(), 
            nn.Linear(hidden_size, output_size),  #Hidden Layers and 4 directions to move in
        )

        def forward(self, x): 
            """
                This function will act as a helper to pass information forward to the further layers
                :param x: this is the data that will pass through the neural netword
            """
            logits = self.linear_relu_stack(x)
            return logits
