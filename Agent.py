"""
     This script will house the agent that will learn the plastics
"""


import torch as T # import pytorch
import numpy as np # import numpy
import random # import random
from ED_Data_Pull import DataPuller, DatasetSplitter # import the data puller and datasplitter classes
import torch.nn as nn # import the neural netowrk package
import torch.optim as optim # import optimizer function
import torch.nn.functional as F # import more helper functions
from Model import Brain # bring in the brain so that we can work with it 


class Agent:
    def __init__(self, model, train_loader, test_loader, lr=0.001, momentum=0.9, epochs=10, device=None):
        """
        Initializes the training agent.

        Args:
            model (nn.Module): Neural network model (Brain).
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for test data.
            lr (float): Learning rate for optimizer.
            momentum (float): Momentum for SGD optimizer.
            epochs (int): Number of training epochs.
            device (str or torch.device): Device to run training on ("cpu" or "cuda").
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device or T.device("cuda" if T.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

    def train(self):
        """
        Trains the model for the specified number of epochs.
        """
        for epoch in range(self.epochs): # loops over the number of epochs
            self.model.train() # trains for each one
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

    def evaluate(self):
        """
        Evaluates the model on the test set.
        """
        self.model.eval()
        correct = 0
        total = 0

        with T.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = T.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

# define variables
model = Brain()
dataset = DataPuller('/Users/vandergeorgeff/Library/CloudStorage/OneDrive-UniversityofDenver/Bok Group/Microplastics/Initial Proof-of-Concept -- 13MAR2025/AS_Data')
splitter = DatasetSplitter(dataset)
train_loader, test_loader = splitter.get_loaders()



agent = Agent(model, train_loader, test_loader, epochs=20)
agent.train()
agent.evaluate()