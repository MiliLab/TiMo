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

ALL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
RGB_BANDS = ['B4', 'B3', 'B2']

QUANTILES = {
    'min_q': {
        'B2': 3.0,
        'B3': 2.0,
        'B4': 0.0
    },
    'max_q': {
        'B2': 88.0,
        'B3': 103.0,
        'B4': 129.0
    }
}


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


def normalize(img, min_q, max_q):
    img = (img - min_q) / (max_q - min_q)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def read_image(path, bands, quantiles=None):
    channels = []
    for b in bands:
        ch = rasterio.open(path / f'{b}.tif').read(1)
        ch = cv2.resize(ch, dsize=(264, 264), interpolation=cv2.INTER_LINEAR_EXACT)
        if quantiles is not None:
            ch = normalize(ch, min_q=quantiles['min_q'][b], max_q=quantiles['max_q'][b])
        channels.append(ch)
    img = np.dstack(channels)
    # img = Image.fromarray(img)
    return img


class SeasonalContrastBasic(SeasonalContrastBase):

    def get_samples(self):
        # return [path for path in self.root.glob('*/*') if path.is_dir()]
        samples = []
        for entry in os.scandir(self.root):
            for subentry in os.scandir(entry.path):
                if subentry.is_dir():
                    samples.append(Path(subentry.path))
        return samples

    def __getitem__(self, index):
        path = self.samples[index]
        img = read_image(path, self.bands, QUANTILES)
        if self.transform is not None:
            img = self.transform(img)
        return img


class SeasonalContrastTemporal(SeasonalContrastBase):

    def get_samples(self):
        return [path for path in self.root.glob('*') if path.is_dir()]

    def __getitem__(self, index):
        root = self.samples[index]
        paths = np.random.choice([path for path in root.glob('*') if path.is_dir()], 2)
        images = []
        for path in paths:
            img = read_image(path, self.bands, QUANTILES)
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
        return images[0], images[1]



class SeasonalContrastTemporal_v3(SeasonalContrastBase):#跟v2相比返回了paths

    def get_samples(self):
        return [path for path in self.root.glob('*') if path.is_dir()]

    def __getitem__(self, index):

        root = self.samples[index]
        paths = [path for path in root.glob('*') if path.is_dir()]

        images = []
        for path in paths:
            img = read_image(path, RGB_BANDS)
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
        try:
            images=np.array(images).transpose(0,3,1,2)
        except:
            print(str(root).split('/')[-1])
        return images,str(root).split('/')[-1],paths

class _RepeatSampler(object):
    """
    Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class InfiniteDataLoader(DataLoader):
    """
    Dataloader that reuses workers.
    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class TPCO_npy_ts(torch.utils.data.Dataset):

    def __init__(self, root,csv_path,num_ts=3,mode='rgb',min_year=2002,date_format='ymd',transform=None):
        self.root = root

        # self.ids = os.listdir(os.path.join(self.root, self.mode[0]))

        self.ids=pd.read_csv(csv_path)
        # self.ids= self.ids.drop([97801,42494])
        # self.ids = self.ids.reset_index(drop=True)
        self.length = len(self.ids)
        self.num_ts=num_ts
        self.mode=mode
        self.transform=transform
        self.date_format = date_format
        self.min_year = min_year
        self.totensor = transforms.ToTensor()
        self.scale = transforms.Resize(224)
    def __getitem__(self, index):
        # try:
        ts =  np.load(os.path.join(self.root,str(self.ids['loc'][index]).zfill(6)+'.npy'))#.transpose(0,3,1,2) # [4,13,264,264] int16 or uint8

        self.transform=transforms.Compose([self.totensor,self.scale])
        ts=torch.stack([self.transform((t/255).transpose(1,2,0).astype(np.float32)) for t in ts])
        dates = np.array([date[:8] for date in self.ids.loc[index][['t1','t2','t3','t4','t5','t6','t7','t8','t9','t10']]])
        seasons = np.random.choice(range(ts.shape[0]), self.num_ts, replace=False)
        ts = np.stack([ts[season] for season in seasons], axis=0)
        dates = np.stack([dates[season] for season in seasons], axis=0)

        if self.date_format=='ymd':
            dates=np.array([self.parse_timestamp(date) for date in dates])


        return ts,dates,ts

    def parse_timestamp(self, timestamp):
        # print('timestamp',timestamp)
        year = int(timestamp[:4])
        month = int(timestamp[4:6])
        day = int(timestamp[6:8])
        return np.array([year - self.min_year, month - 1, day])

    def __len__(self):
        return self.length
