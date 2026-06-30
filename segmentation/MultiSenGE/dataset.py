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

import warnings
warnings.filterwarnings("ignore")
def normalize(img, min_q, max_q):
    img = (img - min_q) / (max_q - min_q)
    # img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

class MultiSenGE_dataset(Dataset):
    def __init__(self,json_path:str,split:str):
        super().__init__()
        self.json_path=json_path
        #总共有14个tiles，选择8个为train，3个valid，3个test
        if split=='train':
            tiles = np.unique(np.array([json_name[:5] for json_name in os.listdir(json_path)]))[:8]
        elif split=='valid':
            tiles= np.unique(np.array([json_name[:5] for json_name in os.listdir(json_path)]))[8:11]
        elif split=='test':
            tiles= np.unique(np.array([json_name[:5] for json_name in os.listdir(json_path)]))[11:14]
        else:
            raise Exception("Please specify a split!!")
        self.jsons=[]
        for tile in tiles:
            for js in os.listdir(json_path):
                if js.startswith(tile):
                    self.jsons.append(js)

    def __getitem__(self, index):
        data=os.path.join(self.json_path,self.jsons[index])
        with open(data) as f:
            data = json.load(f)
            s2_names = data['corresponding_s2'].split(';')
            label_name = s2_names[0][:6] + 'GR' + s2_names[0][17:]
            s2_list=[]
            date_list=[]
            for s2_name in s2_names:
                img_path=os.path.join(os.path.dirname(self.json_path),'s2',s2_name)
                with rasterio.open(img_path) as data:
                    img = data.read()
                    s2_list.append(img)
                    date_list.append([int(s2_name[6:10]),int(s2_name[10:12]),int(s2_name[12:14])])
            s2_ts=np.stack(s2_list)*1e-4
            dates=np.stack(date_list)
            label_path=os.path.join(os.path.dirname(self.json_path),'ground_reference',label_name)
            with rasterio.open(label_path) as data:
                label = data.read()
        return s2_ts.astype(np.float32),dates ,(label-1)[0]
    def __len__(self):

        return len(self.jsons)
def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)

def SenGE_pad_collate(batch, pad_value=0):
    sizes = [e[0].shape[0] for e in batch]
    m = max(sizes)
    _sen = [torch.from_numpy(i[0]) for i in batch]
    _dates = [torch.from_numpy(i[1]) for i in batch]
    _label = [torch.from_numpy(i[2]) for i in batch]
    padded_data, padded_dates = [], []
    if not all(s == m for s in sizes):
        for data, date in zip(_sen, _dates):
            padded_data.append(pad_tensor(data, m, pad_value=pad_value))
            padded_dates.append(pad_tensor(date, m, pad_value=pad_value))
    else:
        padded_data=_sen
        padded_dates=_dates

    return torch.stack(padded_data),torch.stack(padded_dates),torch.stack(_label)

def SenGE_pad_collate_same(batch, pad_value=0):
    sizes = [e[0].shape[0] for e in batch]
    m = 12 #12 months
    _sen = [torch.from_numpy(i[0]) for i in batch]
    _dates = [torch.from_numpy(i[1]) for i in batch]
    _label = [torch.from_numpy(i[2]) for i in batch]
    padded_data, padded_dates = [], []
    if not all(s == m for s in sizes):
        for data, date in zip(_sen, _dates):
            padded_data.append(pad_tensor(data, m, pad_value=pad_value))
            padded_dates.append(pad_tensor(date, m, pad_value=pad_value))
    else:
        padded_data=_sen
        padded_dates=_dates

    return torch.stack(padded_data), torch.stack(padded_dates), torch.stack(_label)


def SenGE_pad_collate_3(batch):

    sizes = [e[0].shape[0] for e in batch]

    _sen = [torch.from_numpy(i[0]) for i in batch]
    _dates = [torch.from_numpy(i[1]) for i in batch]
    _label = [torch.from_numpy(i[2]) for i in batch]

    padded_data, padded_dates = [], []
    if not all(s == 3 for s in sizes):
        for data, date in zip(_sen, _dates):
            np.random.seed(0)
            idlist = range(data.shape[0])
            id_choice = np.random.choice(idlist, 3)
            padded_data.append(data[id_choice])
            padded_dates.append(date[id_choice])
    else:
        padded_data = _sen
        padded_dates = _dates

    return torch.stack(padded_data), torch.stack(padded_dates), torch.stack(_label)
