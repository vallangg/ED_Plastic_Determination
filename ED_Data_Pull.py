import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
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

                    # Read CSV with no headers, columns 1-5 are indivudal reads, column 6 is the average and coluumn 7 is the standard error
                    df = pd.read_csv(filepath, header=None)

                    # for the purpose of increasing the number of training data, all reads and aveages will be used


                    # cycle through all values and add them to the data
                    for ii_reads in range(1,6):
                         avg_tensor = torch.tensor(df[ii_reads].values, dtype=torch.float32)

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



class DatasetSplitter:
     def __init__(self, dataset, train_ratio: float=0.8, batch_size: int=32, shuffle: bool=True):
          """
          Takes the data set and a ratio to create a training data set and a testing data set. 
          
          :param:
               dataset (Dataset): this comes from the datapuller class
               train_ratio (float): Proportion of data to use for training. standard=0.7
               test_ratio (float): Proportion of data to use for testing. standard=0.2
               batch_size (int): Number of samples per batch in the DataLoader.
               shuffle (bool): Whether to shuffle the dataset before splitting.
          """

          self.dataset = dataset
          self.train_ratio = train_ratio
          self.batch_size = batch_size
          self.shuffle = shuffle

          self.train_loader, self.test_loader = self._split_dataset()

     def _split_dataset(self):
          """
          Splits the dataset into training and testing DataLoaders.
          """
          total_len = len(self.dataset)
          train_len = int(total_len * self.train_ratio)
          test_len = total_len - train_len 

          if self.shuffle:
               generator = torch.Generator().manual_seed(42)
               train_data, test_data = random_split(self.dataset, [train_len, test_len], generator=generator) # creates the randomly split data sets
          else:
               train_data, test_data = random_split(self.dataset, [train_len, test_len])

          train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
          test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

          return train_loader, test_loader

     def get_loaders(self):
          """
          Returns the training and testing DataLoaders.
          """
          return self.train_loader, self.test_loader


# Example usage
if __name__ == '__main__':
     # folder_path = 'C:/Users/georg/OneDrive - University of Denver/Bok Group/Microplastics/Initial Proof-of-Concept -- 13MAR2025/AS_Data'  # Replace with actual path

     folder_path = '/Users/vandergeorgeff/Library/CloudStorage/OneDrive-UniversityofDenver/Bok Group/Microplastics/Initial Proof-of-Concept -- 13MAR2025/AS_Data'

     dataset = DataPuller(folder_path)
     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

     for batch_data, batch_labels in dataloader:
          print("Batch data:", batch_data)
          print("Batch labels:", batch_labels)
          print(dataset.label_map)
          break




