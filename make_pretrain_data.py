import os
from pathlib import Path

import numpy as np
import rasterio
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
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



class SeasonalContrastTemporal_v3(SeasonalContrastBase):

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

        self.ids=pd.read_csv(csv_path)
        
        self.length = len(self.ids)
        self.num_ts=num_ts
        self.mode=mode
        self.transform=transform
        self.date_format = date_format
        self.min_year = min_year
        self.totensor = transforms.ToTensor()
        self.scale = transforms.Resize(224)
    def __getitem__(self, index):
        
        ts =  np.load(os.path.join(self.root,str(self.ids['loc'][index]).zfill(6)+'.npy'))

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
       
        year = int(timestamp[:4])
        month = int(timestamp[4:6])
        day = int(timestamp[6:8])
        return np.array([year - self.min_year, month - 1, day])

    def __len__(self):
        return self.length



if __name__ == '__main__':
    import argparse
    import shutil
    from tqdm import tqdm
    from torch.utils.data import ConcatDataset
    import multiprocessing as mp
    import csv

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='/data/xiaolei.qin/Dataset/temporal_contrast_1m')
   
    parser.add_argument('--save_path',
                        default='/data/xiaolei.qin/Dataset/TPCONPY_RGB',
                        type=str)
    parser.add_argument('--make_npy_file', action='store_true', default=False)
    parser.add_argument('--make_ts_file', action='store_true', default=False)
    parser.add_argument('--frac', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--mode', nargs='*', type=str, default=['s1', 's2c'])
    parser.add_argument('--dtype', type=str, default='uint8')
    parser.add_argument('--sid', type=int, default=0)
    parser.add_argument('--eid', type=int, default=12)
    args = parser.parse_args()

    dataset = SeasonalContrastTemporal_v3(args.root)

    if args.make_npy_file:
        loader = InfiniteDataLoader(dataset, num_workers=args.num_workers, collate_fn=lambda x: x[0])
        for index, (image, name,_) in tqdm(enumerate(loader), total=len(dataset), desc='Creating NPY'):
            saveimg = np.array(image)
            if not os.path.exists(args.save_path+'/' + name + '.npy'):
                np.save(args.save_path+'/' + name + '.npy', saveimg)

    if args.make_ts_file:
        dataset = SeasonalContrastTemporal_v3(args.root)
        dataset = torch.utils.data.Subset(dataset, range(10))
        with open("/data/xiaolei.qin/Dataset/tmp/TPCONPY_ts.csv", "w", newline="") as file:#please specify the csv file name to save
            data_root_patch = args.root
            for _,patch_id,patch_seasons in tqdm(dataset):
                writer = csv.writer(file,delimiter=';')
                writer.writerow([str(patch_id)] + [[str(patch_seasons[i]).split('/')[-1] for i in range(len(patch_seasons))]])
