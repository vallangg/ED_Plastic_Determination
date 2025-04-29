import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Define a custom PyTorch Dataset for reading headerless .csv files
class DataPuller(Dataset):
     def __init__(self, folder_path):
          """
          Initializes the dataset by reading all .csv files in the specified folder.
          Each file is treated as one sample. The 'average' column is extracted
          and used as the input tensor.
          
          Args:
               folder_path (str): Path to the folder containing .csv files.
          """
          self.data = []         # List of input tensors (each from a file)
          self.labels = []       # Numeric label for each sample
          self.label_map = {}    # Map from file name to numeric label
          self.file_names = []   # Optional: store names for reference

          self._load_data(folder_path)

     def _load_data(self, folder_path):
          """
          Loads and processes all .csv files. Manually assigns column names since the files have no headers.
          """
          for idx, filename in enumerate(sorted(os.listdir(folder_path))):
               if filename.endswith('.csv'):
                    filepath = os.path.join(folder_path, filename)
                    label_name = os.path.splitext(filename)[0].split("_")[0]
                    # print(os.path.splitext(filename)[0].split("_")[0])

                    # Assign a numeric label to this file
                    self.label_map[label_name] = idx

                    # Read CSV with no header and assign column names
                    df = pd.read_csv(filepath, header=None, names=[
                         'angle', 'rep1', 'rep2', 'rep3', 'rep4', 'rep5', 'average', 'stderr'
                    ])

                    # Extract 'average' column and convert to a PyTorch tensor
                    avg_tensor = torch.tensor(df['average'].values, dtype=torch.float32)

                    self.data.append(avg_tensor)
                    self.labels.append(idx)
                    self.file_names.append(label_name)

          self.labels = torch.tensor(self.labels, dtype=torch.long)

     def __len__(self):
          """Returns the number of samples (files)."""
          return len(self.data)

     def __getitem__(self, index):
          """Returns the (average tensor, label) pair for the given index."""
          return self.data[index], self.labels[index]


# Example usage
if __name__ == '__main__':
     folder_path = 'C:/Users/georg/OneDrive - University of Denver/Bok Group/Microplastics/Initial Proof-of-Concept -- 13MAR2025/AS_Data'  # Replace with actual path

     dataset = DataPuller(folder_path)
     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

     for batch_data, batch_labels in dataloader:
          print("Batch data:", batch_data)
          print("Batch labels:", batch_labels)
          print(dataset.label_map)
          break
