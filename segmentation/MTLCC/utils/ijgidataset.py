import os
import random

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

try:
    from .progressbar import ProgressBar
except ImportError:
    from progressbar import ProgressBar


LABEL_FILENAME = "y.tif"


def read(file):
    with rasterio.open(file) as src:
        return src.read(), src.profile


class ijgiDataset(torch.utils.data.Dataset):
    """MTLCC crop-type segmentation dataset used in TiMo fine-tuning."""

    def __init__(self, root_dir, seqlength=30, tileids=None):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "data")
        self.seqlength = seqlength

        stats = dict(rejected_nopath=0, rejected_length=0, total_samples=0)
        self.samples = []
        self.ndates = []

        if tileids is None:
            files = os.listdir(self.data_dir)
        else:
            with open(os.path.join(root_dir, tileids), "r") as f:
                files = [el.strip() for el in f.readlines()]
            if tileids == "tileids/train_fold0.tileids":
                random.seed(6)
                random.shuffle(files)
            if tileids == "tileids/test_fold0.tileids":
                random.seed(10)
                random.shuffle(files)

        self.classids, self.classes = self.read_classes(os.path.join(self.root_dir, "classes.txt"))

        progress = ProgressBar(len(files), fmt=ProgressBar.FULL)
        for f in files:
            progress.current += 1
            progress()

            path = os.path.join(self.data_dir, f)
            if not os.path.exists(path):
                stats["rejected_nopath"] += 1
                continue

            ndates = len(get_dates(path))
            if ndates < self.seqlength:
                stats["rejected_length"] += 1
                continue

            stats["total_samples"] += 1
            self.samples.append(f)
            self.ndates.append(ndates)

        self.len = len(self.samples)
        progress.done()
        print_stats(stats)

    def read_classes(self, csv):
        with open(csv, "r") as f:
            classes = f.readlines()

        ids = []
        names = []
        for row in classes:
            row = row.strip()
            if "|" in row:
                class_id, class_name = row.split("|")
                ids.append(int(class_id))
                names.append(class_name)

        return ids, names

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.samples[idx])
        label, profile = read(os.path.join(path, LABEL_FILENAME))
        profile["name"] = self.samples[idx]

        dates = get_dates(path, n=self.seqlength)

        x10 = []
        x20 = []
        x60 = []
        for date in dates:
            x10.append(read(os.path.join(path, date + "_10m.tif"))[0])
            x20.append(read(os.path.join(path, date + "_20m.tif"))[0])
            x60.append(read(os.path.join(path, date + "_60m.tif"))[0])

        x10 = np.array(x10) * 1e-4
        x20 = np.array(x20) * 1e-4
        x60 = np.array(x60) * 1e-4

        label = label[0]
        new_label = np.zeros(label.shape, np.int32)
        for class_id, idx in zip(self.classids, range(len(self.classids))):
            new_label[label == class_id] = idx

        label = torch.from_numpy(new_label)
        x10 = torch.from_numpy(x10)
        x20 = torch.from_numpy(x20)
        x60 = torch.from_numpy(x60)

        x20 = F.interpolate(x20, size=x10.shape[2:4])
        x60 = F.interpolate(x60, size=x10.shape[2:4])
        x = torch.cat((x10, x20, x60), 1)

        return x.float(), torch.tensor(range(x.shape[0])), label.long()


def get_dates(path, n=None):
    files = os.listdir(path)
    dates = []
    for f in files:
        f = f.split("_")[0]
        if len(f) == 8:
            dates.append(f)

    unique_dates = list(set(dates))
    unique_dates.sort(key=dates.index)
    dates = unique_dates

    if n is not None:
        dates = dates[:n]

    dates = list(dates)
    dates.sort()
    return dates


def print_stats(stats):
    print(", ".join(["{}:{}".format(k, v) for k, v in stats.items()]))
