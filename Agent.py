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
from sklearn.metrics import roc_curve, auc, confusion_matrix # import sklean to help do the data analysis
import pandas as pd
import seaborn as sb


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
          self.optimizer = T.optim.Adam(self.model.parameters(), lr=lr)

     def train(self):
          """
          Trains the model for the specified number of epochs.
          """
          for epoch in range(self.epochs): # loops over the number of epochs
               self.model.train() # set the model to training mode
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
          print(f"Test Accuracy: {accuracy:.2f}%")
          return T.cat(all_outputs), T.cat(all_labels) # return the label values



     def plot_auc_roc(self, save_path="roc_curve.png"):
          result = self.evaluate()
          if result is None:
               print("AUC plot skipped: No data returned from evaluate().")
               return

          outputs, labels = result

          # Move to CPU, detach, convert to numpy
          probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
          labels = labels.detach().cpu()
          onehot = F.one_hot(labels, num_classes=self.num_classes).cpu().numpy()

          plt.figure()

          for i in range(self.num_classes):
               y_true = onehot[:, i]
               y_score = probs[:, i]

               # Skip class if it has no positive samples in ground truth
               if np.sum(y_true) == 0:
                    print(f"Skipping class {i}: no positive samples in y_true.")
                    continue

               fpr, tpr, _ = roc_curve(y_true, y_score)
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

def plot_confusion_matrix(agent, data_loader, class_names):
     """
          # code to make a confusion matrix 
          # https://jillanisofttech.medium.com/building-an-ann-with-pytorch-a-deep-dive-into-neural-network-training-a7fdaa047d81
     """
     agent.model.eval()
     y_pred = []
     y_true = []

     with T.no_grad():
          for inputs, labels in data_loader:
               inputs = inputs.to(agent.device)
               labels = labels.to(agent.device)

               outputs = agent.model(inputs)
               _, predicted = T.max(outputs, 1)

               y_pred.extend(predicted.cpu().numpy())
               y_true.extend(labels.cpu().numpy())

     # Create normalized confusion matrix
     cf_matrix = confusion_matrix(y_true, y_pred)
     row_sums = np.sum(cf_matrix, axis=1, keepdims=True)
     row_sums[row_sums == 0] = 1  # Avoid division by zero

     df_cm = pd.DataFrame(cf_matrix / row_sums, index=class_names, columns=class_names)

     plt.figure(figsize=(10, 7))
     sb.heatmap(df_cm, annot=True, cmap="Blues", fmt=".2f")
     plt.title("Normalized Confusion Matrix")
     plt.xlabel("Predicted Label")
     plt.ylabel("True Label")
     plt.show()




# define variables
model = Brain()
dataset = DataPuller('/Users/vandergeorgeff/Library/CloudStorage/OneDrive-UniversityofDenver/Bok Group/Microplastics/AS_Data_Chiral')
# dataset = DataPuller('C:/Users/georg/OneDrive - University of Denver/Bok Group/Microplastics/AS_Data')
splitter = DatasetSplitter(dataset, train_ratio=0.8)
train_loader, test_loader = splitter.get_loaders()

all_loaders = splitter.get_all_loaders()

agent = Agent(model, test_loader, train_loader, epochs=300)
agent.train()
agent.evaluate()

class_names = ("PP", "PS")  # Make sure this matches your 5 classes
plot_confusion_matrix(agent, test_loader, class_names)
