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
import matplotlib.pyplot as plt # import matplot lib to make graphs
from sklearn.metrics import roc_curve, auc # import sklean to help do the data analysis
import pandas as pd


class Agent:
     """
          Create the agent class. This is what will be used to run the model
     """
     def __init__(self, model, train_loader, test_loader, lr=0.001, momentum=0.9, epochs=10, device=None, num_classes: int = 5):
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
               num_classes (int): the number of outputs that the model has, 5 at the moment
          """
          self.model = model
          self.train_loader = train_loader
          self.test_loader = test_loader
          self.epochs = epochs
          self.device = device or T.device("cuda" if T.cuda.is_available() else "cpu")
          self.num_classes = num_classes

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
               print(f"Train Set Size: {len(self.train_loader)}")
               print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

     def evaluate(self):
          """
          Evaluates the model on the test set.
          """
          self.model.eval()
          correct = 0
          total = 0

          # these will be sued to determine the AUC ROC curve
          all_outputs = [] 
          all_labels = []

          with T.no_grad():
               for inputs, labels in self.test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    _, predicted = T.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_outputs.append(outputs)
                    all_labels.append(labels)

          accuracy = 100 * correct / total
          print(f"Test Accuracy: {accuracy:.2f}%, Train Test Size: {len(train_loader)}")
          return T.cat(all_outputs), T.cat(all_labels) # return the label values



     def plot_auc_roc(self, save_path="roc_curve.png"):
          
          result = self.evaluate() # load the results

          
          outputs, labels = result
          probs = F.softmax(outputs, dim=1).numpy()
          onehot = F.one_hot(labels, num_classes=self.num_classes).numpy()

          plt.figure()
          for i in range(self.num_classes):
               fpr, tpr, _ = roc_curve(onehot[:, i], probs[:, i])
               roc_auc = auc(fpr, tpr)
               plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

          plt.plot([0, 1], [0, 1], 'k--', label='Chance')
          plt.xlabel('False Positive Rate')
          plt.ylabel('True Positive Rate')
          plt.title('AUC-ROC Curve')
          plt.legend()
          plt.grid(True)
          plt.savefig(save_path)
          plt.close()


     def export_roc_data(self, export_path="roc_data.csv"):
          outputs, labels = self.evaluate()
          probs = F.softmax(outputs, dim=1).numpy()
          onehot = F.one_hot(labels, num_classes=self.num_classes).numpy()

          rows = []
          for i in range(self.num_classes):
               fpr, tpr, thresholds = roc_curve(onehot[:, i], probs[:, i])
               for f, t, th in zip(fpr, tpr, thresholds):
                    rows.append({'class': i, 'fpr': f, 'tpr': t, 'threshold': th})
          pd.DataFrame(rows).to_csv(export_path, index=False)

     def save_model(self, save_path="model.pth"):
          T.save(self.model.state_dict(), save_path)

# define variables
model = Brain()
dataset = DataPuller('C:/Users/georg/OneDrive - University of Denver/Bok Group/Microplastics/AS_Data')
splitter = DatasetSplitter(dataset, train_ratio=0.1)
train_loader, test_loader = splitter.get_loaders()



agent = Agent(model, train_loader, test_loader, epochs=200)
agent.train()
agent.evaluate()
