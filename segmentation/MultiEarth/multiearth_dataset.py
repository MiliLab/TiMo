import os
import pickle

import numpy as np
import torch


def normalize(img):
    img_max = np.max(img)
    img_min = np.min(img)
    return (img - img_min) / (img_max - img_min + 1e-13)


class MultiEarthDeforest(torch.utils.data.Dataset):
    """MultiEarth deforestation dataset exported as per-location pkl files."""

    def __init__(self, pkl_path: str, split: str):
        super().__init__()
        self.pkl_path = pkl_path
        self.loc = os.listdir(pkl_path)

        if split == 'train':
            self.loc = self.loc[:78]
        elif split == 'val':
            self.loc = self.loc[78:156]
        elif split == 'test':
            self.loc = self.loc[156:]
        else:
            raise ValueError("Please specify split as train, val, or test.")

    def __getitem__(self, index):
        loc = self.loc[index]
        data_list = []
        date_list = []
        target_list = []
        loc_path = os.path.join(self.pkl_path, loc)

        for ts in os.listdir(loc_path):
            ts_path = os.path.join(loc_path, ts)
            with open(ts_path, 'rb') as file:
                data = pickle.load(file)

            data_list.append(normalize(data['img']))
            target_list.append(data['target'][0])
            date_list.append([int(ts[:4]), int(ts[5:7]), int(ts[8:10])])

        data = np.stack(data_list)[:3]
        dates = np.stack(date_list)[:3]
        target = np.stack(target_list)[:3]

        return data.astype(np.float32), dates, target

    def __len__(self):
        return len(self.loc)


def MultiEarth_pad_collate_same(batch, pad_value=0):
    sizes = [e[0].shape[0] for e in batch]
    m = min(sizes)
    data = [torch.from_numpy(i[0]) for i in batch]
    dates = [torch.from_numpy(i[1]) for i in batch]
    labels = [torch.from_numpy(i[2]) for i in batch]

    padded_data, padded_dates, padded_labels = [], [], []
    if not all(s == m for s in sizes):
        for sample, date, label in zip(data, dates, labels):
            padded_data.append(sample[:m])
            padded_dates.append(date[:m])
            padded_labels.append(label[:m])
    else:
        padded_data = data
        padded_dates = dates
        padded_labels = labels

    return torch.stack(padded_data), torch.stack(padded_dates), torch.stack(padded_labels)
