import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

class KampDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.inputs.iloc[idx].values, dtype=torch.float32)
        y = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return x, y


def evaluate_model(model, dataloader):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader):
            x_batch = x_batch.unsqueeze(1)
            y_pred = model(x_batch)

            _, predicted_class = torch.max(y_pred, dim=1)

            all_predictions.extend(predicted_class.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return np.array(all_labels), np.array(all_predictions)

def check_fail_rate(data):
    num_pass = len(data[data['passorfail'] == 0])
    num_fail = len(data[data['passorfail'] == 1])

    print(f"합격 데이터 수 : {num_pass}")
    print(f"불합격 데이터 수 : {num_fail}")
    print(f"불합격 데이터 비율 : {round(num_fail / (num_pass + num_fail) * 100, 2)} %")

def check_nan(data, percent=False):
    if percent:
        print(data.isna().sum() / len(data))
    else:
        print(data.isna().sum())