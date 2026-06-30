import numpy as np
import numpy.ma as ma
import os
from datetime import datetime, timedelta
import json
# from pyproj import Proj, transform
# import fiona
# import warnings
from collections import OrderedDict
from torch.utils.data.dataset import Dataset
import torch
import rasterio
from torch.nn import functional as F
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from itertools import product
from pathlib import Path
from torchvision.transforms import functional as TF


def normalize(img, min_q, max_q):
    img = (img - min_q) / (max_q - min_q)
    # img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

#copy from oscd
QUANTILES = {
    'min_q': {
        'B02': 885.0,
        'B03': 667.0,
        'B04': 426.0
    },
    'max_q': {
        'B02': 2620.0,
        'B03': 2969.0,
        'B04': 3698.0
    }
}

def read_image(path, img, normalize=True):
    ch = rasterio.open(path / img).read()
    ch=ch[:3]
    ch=ch.transpose(1,2,0)
    img = Image.fromarray(ch)
    return img

def read_label(path, img, normalize=True):
    ch = rasterio.open(path / img).read(1)
    img = Image.fromarray(ch)
    return img

class KuroSiwo_dataset(Dataset):
    def __init__(self,root:str,split:str):
        super().__init__()
        self.root=root
        self.samples=[]
        for subfolder1 in os.listdir(root):
            if os.path.isdir(root + '/' + subfolder1):
                for subfolder2 in os.listdir(root + '/' + subfolder1):
                    if os.path.isdir(root + '/' + subfolder1 + '/' + subfolder2):
                        for subfolder3 in os.listdir(root + '/' + subfolder1 + '/' + subfolder2):
                            if os.path.isdir(root + '/' + subfolder1 + '/' + subfolder2+ '/' + subfolder3):
                                self.samples.append(root + '/' + subfolder1 + '/' + subfolder2+ '/' + subfolder3)
        if split=='train':
            self.samples = self.samples[:(len(self.samples)//5)*3]

        elif split=='valid':
            self.samples = self.samples[(len(self.samples)//5)*3:(len(self.samples)//5)*4]
        elif split=='test':
            self.samples = self.samples[(len(self.samples)//5)*4:]
        else:
            raise Exception("Please specify a split!!")
    def __getitem__(self, index):
        t1_vv,t2_vv,t3_vv,t1_vh,t2_vh,t3_vh=[],[],[],[],[],[]
        datelist=[]
        for file in os.listdir(self.samples[index]):
            if file.startswith('MK0_MLU'):
                label=rasterio.open(self.samples[index]+'/'+file).read(1)
            if file.startswith('SL1_IVV'):
                t1_vv=rasterio.open(self.samples[index]+'/'+file).read(1)
                t1_vv=(t1_vv-t1_vv.min())/(t1_vv.max()-t1_vv.min()+1e-13)
                datelist.append([int(file.split('_')[-1][:4]),int(file.split('_')[-1][4:6]),int(file.split('_')[-1][6:8])])
            if file.startswith('SL1_IVH'):
                t1_vh=rasterio.open(self.samples[index]+'/'+file).read(1)
                t1_vh = (t1_vh - t1_vh.min()) / (t1_vh.max() - t1_vh.min()+1e-13)
            if file.startswith('MS1_IVV'):
                t2_vv=rasterio.open(self.samples[index]+'/'+file).read(1)
                t2_vv = (t2_vv - t2_vv.min()) / (t2_vv.max() - t2_vv.min()+1e-13)
                datelist.append([int(file.split('_')[-1][:4]),int(file.split('_')[-1][4:6]),int(file.split('_')[-1][6:8])])
            if file.startswith('MS1_IVH'):
                t2_vh=rasterio.open(self.samples[index]+'/'+file).read(1)
                t2_vh = (t2_vh - t2_vh.min()) / (t2_vh.max() - t2_vh.min()+1e-13)
            if file.startswith('SL2_IVV'):
                t3_vv=rasterio.open(self.samples[index]+'/'+file).read(1)
                t3_vv = (t3_vv - t3_vv.min()) / (t3_vv.max() - t3_vv.min()+1e-13)
                datelist.append([int(file.split('_')[-1][:4]),int(file.split('_')[-1][4:6]),int(file.split('_')[-1][6:8])])
            if file.startswith('SL2_IVH'):
                t3_vh=rasterio.open(self.samples[index]+'/'+file).read(1)
                t3_vh = (t3_vh - t3_vh.min()) / (t3_vh.max() - t3_vh.min()+1e-13)

        t1=torch.concat([torch.tensor(t1_vv).unsqueeze(0),torch.tensor(t1_vh).unsqueeze(0)])
        t2 = torch.concat([torch.tensor(t2_vv).unsqueeze(0), torch.tensor(t2_vh).unsqueeze(0)])
        t3 = torch.concat([torch.tensor(t3_vv).unsqueeze(0), torch.tensor(t3_vh).unsqueeze(0)])

        ts=torch.concat([t3.unsqueeze(0),t1.unsqueeze(0),t2.unsqueeze(0)])
        ts[ts!=ts]=-1
        date=torch.tensor(datelist)

        return ts,date,label


    def __len__(self):

            return len(self.samples)


def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)

def SN7_pad_collate(batch, pad_value=0):
    sizes = [e[0].shape[0] for e in batch]
    m = min(sizes)
    _sen = [i[0] for i in batch]
    _dates = [i[1] for i in batch]
    _label = [i[2] for i in batch]

    padded_data, padded_dates , padded_label= [], [],[]
    if not all(s == m for s in sizes):
        for data, date,label in zip(_sen, _dates,_label):
            np.random.seed(0)
            idlist = range(data.shape[0])
            id_choice = np.random.choice(idlist, m)
            padded_data.append(data[id_choice])
            padded_dates.append(date[id_choice])
            padded_label.append(label[id_choice])
    else:
        padded_data=_sen
        padded_dates=_dates
        padded_label=_label

    return torch.stack(padded_data),torch.stack(padded_dates),torch.stack(padded_label)
