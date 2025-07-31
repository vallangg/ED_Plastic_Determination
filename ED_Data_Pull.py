import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

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
          self.allowed_classes = {'PE': 0, 'PET':1, 'PP':2, 'PS':3} # these are the classes that are allowed
          self.class_lookup = {0: 'PE', 1:"PET", 2:"PP", 3:"PS"}

          self._load_data(folder_path)

     def _load_data(self, folder_path):
          """
          Loads and processes all .csv files. Manually assigns column names since the files have no headers.
          """
          class_counter = 0  # Counter for assigning numeric labels to unique classes

          for idx, filename in enumerate(os.listdir(folder_path)):
               if filename.endswith('.csv'):
                    filepath = os.path.join(folder_path, filename)
                    label_name = os.path.splitext(filename)[0].split("-")[0].upper()
                    #print(f"idx: {idx} file name: {os.path.splitext(filename)[0].split("-")[0]}")

                    index_label = self.allowed_classes[label_name]

                    #print(f"index: {index_label} file name: {os.path.splitext(filename)[0].split("-")[0]}")

                    # Assign a consistent label for each material type
                    if label_name not in self.label_map:
                         self.label_map[label_name] = class_counter
                         class_counter += 1

                    # Assign a numeric label to this file
                    self.label_map[label_name] = index_label

                    # Read CSV with no headers, columns 1-5 are indivudal reads, column 6 is the average and coluumn 7 is the standard error
                    df = pd.read_csv(filepath, header=None)

                    # for the purpose of increasing the number of training data, all reads and aveages will be used


                    # cycle through all values and add them to the data
                    for ii_reads in range(1,6):
                         avg_tensor = torch.tensor(df[ii_reads].values, dtype=torch.float32)

                         self.data.append(avg_tensor)
                         self.labels.append(index_label)
                         self.file_names.append(label_name)

     def __len__(self):
          """Returns the number of samples (files)."""
          return len(self.data)

     def __getitem__(self, index):
          """Returns the (average tensor, label) pair for the given index."""
          return self.data[index], self.labels[index]
     
     def __str__(self):
          """
          what to print when the function is called
          """
          return f"data: {self.data}, labels: {self.labels}"



class DatasetSplitter:
     def __init__(self, dataset, train_ratio: float=0.8, batch_size: int=32, shuffle: bool=True, shuffler: bool = True):
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

          if shuffler:
               print("Stratified")
               self.train_loader, self.test_loader = self._stratified_split()
          else:
               print("Random")
               self.train_loader, self.test_loader = self._split_dataset()

     def _split_dataset(self):
          """
          Splits the dataset into training and testing DataLoaders.
          each set should have at least one of each type of data in it 
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
     
     def _stratified_split(self):
          # Extract labels from dataset
          labels = self.dataset.labels.numpy() if isinstance(self.dataset.labels, torch.Tensor) else self.dataset.labels

          splitter = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.train_ratio, random_state=42)
          train_idx, test_idx = next(splitter.split(X=range(len(labels)), y=labels))

          train_subset = Subset(self.dataset, train_idx)
          test_subset = Subset(self.dataset, test_idx)

          train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=self.shuffle)
          test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

          return train_loader, test_loader

     def get_loaders(self):
          """
          Returns the training and testing DataLoaders.
          """
          print(f"Train Loader Labels:")
          for batch in self.train_loader:
               _, labels = batch
               print(labels)
          print(f"Test Loader Labels:")
          for batch in self.test_loader:
               _, labels = batch
               print(labels)
          return self.train_loader, self.test_loader
     
     def get_all_loaders(self):
          return DataLoader(self.dataset, shuffle=False)
     
     def __str__(self):
          """
          This will return the data set when the print function is called on the splitter type
          """
          print_labels = []
          for label in self.dataset.labels:
               # print(label)
               # print(self.dataset.class_lookup[label])
               print_labels.append(self.dataset.class_lookup[label])
               # print_labels += self.dataset.class_lookup[label]
          return f"{print_labels}"
          return f"{self.dataset.labels} {self.dataset.label_map}"
     
     def __len__(self):
          """
          Returns the length of the splitter
          """
          return len(self.dataset)



# folder_path = "/Users/vandergeorgeff/Library/CloudStorage/OneDrive-UniversityofDenver/Bok Group/Microplastics/AS_Data"

# dataset_1 = DataPuller(folder_path)
# # dataloader = DataLoader(dataset_1, batch_size=4, shuffle=True)
# splitter = DatasetSplitter(dataset_1, train_ratio=0.8)
# # splitter.print_subset_info()

# # for batch_data, batch_labels in dataloader:
# #      # print("Batch data:", batch_data)
# #      print("Batch labels:", batch_labels)
# #      print(f"Label Map:",dataset.label_map)

# #      break




