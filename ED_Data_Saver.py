import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class ASDataSaver(Dataset):
    def __init__(self, folder_path):
        self.data = []         # List of input tensors
        self.labels = []       # Numeric label for each sample
        self.label_map = {}    # Map from class name to numeric label
        self.file_names = []   # List of file label names
        self.read_ids = []     # List of read column IDs
        self.sample_id = [] # this is for the is of which read it is
        self.allowed_classes = {'PE': 3, 'PET': 1, 'PP': 0, 'PS': 1}
        self.class_lookup = {3: 'PE', 1: "PET", 0: "PP", 1: "PS"}

        self._load_data(folder_path)

    def _load_data(self, folder_path):
        class_counter = 0

        for idx, filename in enumerate(os.listdir(folder_path)):
            if filename.endswith('.csv'):
                filepath = os.path.join(folder_path, filename)
                label_name = os.path.splitext(filename)[0].split("-")[0].upper()
                sample_name = os.path.splitext(filename)[0].split("-")[1].upper()

                if label_name not in self.allowed_classes:
                    continue  # skip any unknown labels

                index_label = self.allowed_classes[label_name]

                if label_name not in self.label_map:
                    self.label_map[label_name] = class_counter
                    class_counter += 1

                self.label_map[label_name] = index_label

                df = pd.read_csv(filepath, header=None)

                for ii_reads in range(1, 6):
                    avg_tensor = torch.tensor(df[ii_reads].values, dtype=torch.float32)

                    self.data.append(avg_tensor)
                    self.labels.append(index_label)
                    self.file_names.append(filename)  # use full filename for clarity
                    self.sample_id.append(sample_name)
                    self.read_ids.append(ii_reads)    # track which read column it came from

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __str__(self):
        return f"data: {self.data}, labels: {self.labels}"

    def save_to_csv(self, output_path='compiled_reads.csv'):
        """
        Save all read data into a single CSV file with columns:
        file_name, read_id, value_index, value, label
        """
        records = []

        for i, tensor in enumerate(self.data):
            Sample_id = self.sample_id[i]
            read_id = self.read_ids[i]
            Plastic = self.class_lookup[self.labels[i]]
            values = tensor.tolist()

            for j, value in enumerate(values):
                records.append({
                    'file_name': Sample_id,
                    'read_id': read_id,
                    'value_index': j,
                    'value': value,
                    'Plastic': Plastic
                })

        df_out = pd.DataFrame(records)
        df_out.to_csv(output_path, index=False)


folder_path = r"C:\Users\georg\OneDrive - University of Denver\Bok Group\Microplastics\AS_Data"

print(os.path.exists(folder_path))

dataset = ASDataSaver(folder_path)
dataset.save_to_csv('output_data.csv')
