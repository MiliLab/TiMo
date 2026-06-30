import os
from pathlib import Path

import numpy as np
import rasterio
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
# from pl_bolts.models.self_supervised.moco.transforms import GaussianBlur, imagenet_normalization
import cv2
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
RGB_BANDS = ['B04', 'B03', 'B02']
SAR_BANDS=['corrected_VH']
class SeasonalContrastBase(Dataset):

    def __init__(self, root, bands=None, transform=None):
        super().__init__()
        self.root = Path(root)
        self.bands = bands if bands is not None else RGB_BANDS
        self.transform = transform

        self._samples = None

    @property
    def samples(self):
        if self._samples is None:
            self._samples = self.get_samples()
        return self._samples

    def get_samples(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

def read_image(path, bands, quantiles=None):
    channels = []
    for b in bands:
        try:
            ch = rasterio.open(path+'_'+f'{b}.tif').read(1)
        except Exception:
            ch = np.zeros((224, 224), dtype=np.float32)
        ch = cv2.resize(ch, dsize=(224, 224), interpolation=cv2.INTER_LINEAR_EXACT)
        # ch = normalize(ch)
        ch=ch*1e-4
        channels.append(ch)
    img = np.dstack(channels)
    img=img.transpose(2,0,1)
    return img

def normalize(img):
    min_q=np.min(img)
    max_q=np.max(img)
    img = (img - min_q) / (max_q - min_q+1e-13)

    return img

class Sen12Flood(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str):
        super().__init__()
        self.root = root
        self.samples = []
        self.list_path = os.path.join(self.root, 'S2list.json')
        with open(self.list_path, 'r') as f:
            self.metadata = json.load(f)
            for l in self.metadata:
                self.samples.append(str(self.metadata[l]['folder']))
        if split=='train':
            self.samples=self.samples[:30]#:201
        elif split=='val':
            self.samples=self.samples[30:]##201:268
        elif split=='test':
            self.samples=self.samples[30:]

    def __getitem__(self, index):

        folder = self.samples[index]
        filelist=[]
        datelist=[]
        labellist=[]
        label = self.metadata
        for c in range(1, label[folder]['count'] + 1):
            filelist.append(label[folder][str(c)]['filename'])
            datelist.append(self.parse_timestamp(label[folder][str(c)]['date']))
            labellist.append(label[folder][str(c)]['FLOODING'])

        images=[]
        for file in filelist:
            img = read_image(os.path.join(self.root,folder,file), RGB_BANDS)
            # if self.transform is not None:
            #     img = self.transform(img)
            images.append(img)
        images=np.array(images)
        dates=np.array(datelist)
        labels=np.array(labellist)
        if images.shape[0]>=3:
            np.random.seed(0)
            seasons = np.random.choice(range(images.shape[0]), 3, replace=False)
        else:
            np.random.seed(0)
            seasons = np.random.choice(range(images.shape[0]), 3, replace=True)
        images=np.stack([images[season] for season in seasons], axis=0)
        dates=np.stack([dates[season] for season in seasons], axis=0)
        labels = np.stack([labels[season] for season in seasons], axis=0)
        labels[labels.astype(str)=='False']=0
        labels[labels.astype(str)=='True']=1

        return images.astype(np.float32),dates,labels.astype(np.int64)#.split('/')[-1]

    def parse_timestamp(self, timestamp):

        year = int(timestamp[:4])
        month = int(timestamp[5:7])
        day = int(timestamp[8:10])
        return np.array([year - 2018, month - 1, day])

    def __len__(self):
        return len(self.samples)
