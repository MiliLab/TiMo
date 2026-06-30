#!/usr/bin/env python
# coding: utf-8
import os
import argparse
import numpy as np
import os, random, time
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import misc as misc
import logging
from sync_batchnorm import patch_replication_callback
from pos_embed import interpolate_pos_embed
from pathlib import Path
from MultiSenGE.dataset import MultiSenGE_dataset, SenGE_pad_collate_same, SenGE_pad_collate_3

from torch.utils.data import DataLoader
from MultiEarth.multiearth_dataset import MultiEarthDeforest
from KuroSiwo.kurosiwo_dataset import KuroSiwo_dataset
from MTLCC.utils.ijgidataset import ijgiDataset
from SegFramework import SemsegFramework

import subprocess
from engine_finetune import train_one_epoch,evaluate
from misc import NativeScalerWithGradNormCount as NativeScaler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
parser.add_argument('--backbone', type=str, default='TiMo_base_local2200_tspos',
                    help='backbone name')
parser.add_argument('--decoder', type=str, default=None, choices=['unet', 'unetpp', 'upernet', 'mask2former'],
                    help='decoder name')
parser.add_argument('--model', type=str, default='TiMo', choices=['TiMo'],
                    help='model name')
# epoch
parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to train')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')

# batch size
parser.add_argument('--batch_size', type=int, default=2, help='input batch size for training')
parser.add_argument('--workers', type=int, default=4, help='workers num')

#dataset
parser.add_argument('--dataset', type=str, default='MultiSenGE',
                    choices=['MultiSenGE', 'MultiEarthDeforest', 'kurosiwo', 'MTLCC'], help='dataset')
parser.add_argument('--multisenge_path', type=str, default='', help='Root path for MultiSenGE labels directory')
parser.add_argument('--multiearth_path', type=str, default='', help='Root path for MultiEarth deforestation pkl data')
parser.add_argument('--kurosiwo_path', type=str, default='', help='Root path for KuroSiwo split directory')
parser.add_argument('--mtlcc_path', type=str, default='', help='MTLCC root containing data/, tileids/, and classes.txt')

# distributed
parser.add_argument('--distributed', type=str, default='False', choices=['True', 'False'], help='distributed training')
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--local_rank', type=int)#, default=0)
parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')

# ft: continue training
parser.add_argument('--ft', type=str, default='', help='finetune model')
parser.add_argument('--amp', type=str, default='True', help='whether use amp')
parser.add_argument('--resume', type=str, default='', help='resume name')
parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')

# save
parser.add_argument('--output_dir', type=str, default='./outputs/segmentation', help='path of saving model')
parser.add_argument('--log_dir', default='./logs/segmentation',
                        help='path where to tensorboard log')
# ignored
parser.add_argument('--ignore_label', type=int, default=255, help='ignore index of loss')
parser.add_argument('--nb_classes', type=int, default=14, help='number of classes')
parser.add_argument('--subset', action='store_true',
                        help='Use subset of the data')
# interval
parser.add_argument('--interval', default=5, type=int, help='valid interval')

# init_backbone

parser.add_argument('--init_backbone', type=str, default=None,
                    choices=['none', 'imp', 'rsp', 'beit', 'mae', 'samrs-mae-expand'], help='init model')

# optim
parser.add_argument('--optim', type=str, default=None, choices=['adamw', 'sgd'], help='optim')
parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay')
# input img size
parser.add_argument('--img_size', type=int, default=256, help='image size')
parser.add_argument('--in_chans', type=int, default=10, help='image input channels')
parser.add_argument('--ts_len', type=str, default='3',choices=['3','12','minimax','38'], help='length of temporal dimension')
parser.add_argument('--tubelet_size', type=int, default=1, help='tubelet_size')
parser.add_argument('--patch_size', type=int, default=16, help='patch size')
parser.add_argument("--config_file",default=None, help="Path to the .yml config file")

# port
parser.add_argument('--port', type=str, default='40003', help='master ports')

args = parser.parse_args()
args.rank = 0
LOCAL_RANK = args.local_rank if args.local_rank is not None else 0

if args.output_dir:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

if args.log_dir:
    os.makedirs(args.log_dir, exist_ok=True)

logger_name = "main-logger"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(args.log_dir, 'log.txt'), mode='a')
log_format = '%(asctime)s %(message)s'
fh.setFormatter(logging.Formatter(log_format))
logger.addHandler(fh)

handler = logging.StreamHandler()
fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
handler.setFormatter(logging.Formatter(fmt))
logger.addHandler(handler)


def main_process(args):
    return not args.distributed == 'True' or (args.distributed == 'True' and args.rank % args.world_size == 0)


def set_seeds(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seeds()

################################################### setting ###################################################

if args.distributed == 'True':

    # if 'MASTER_ADDR' not in os.environ:
    if 'SLURM_NTASKS' in os.environ.keys():
        logger.info('#################### srun for DDP! ############################')
    
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.rank = int(os.environ['SLURM_PROCID'])  
        LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
        torch.cuda.set_device(LOCAL_RANK)  
        # os.environ['MASTER_PORT'] = args.port
        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
        dist_url = 'tcp://%s:%s' % (addr, args.port)
        dist.init_process_group(backend='nccl', init_method=dist_url, world_size=args.world_size,
                                rank=args.rank)  

    else:
        logger.info('#################### Launch for DDP! ############################')
        if 'RANK' not in os.environ:
            raise RuntimeError('Distributed mode requires torchrun/srun environment variables. Use --distributed False for single-process runs.')
        args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        args.rank = int(os.environ["RANK"])
        LOCAL_RANK = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(LOCAL_RANK)  
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size,
                                rank=args.rank)  

    assert torch.distributed.is_initialized()

if main_process(args):
    logger.info('<<<<<<<<<<<<<<<<< args <<<<<<<<<<<<<<<<<')
    logger.info(args)

##################################################### augmentation #######################################################

def set_seeds(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()


########################################################## dataset ####################################################
def require_path(path, arg_name):
    if not path:
        raise ValueError(f'Please set {arg_name} for dataset {args.dataset}.')
    return path


if args.dataset=='MultiSenGE':
    data_root = require_path(args.multisenge_path, '--multisenge_path')
    dataset_train = MultiSenGE_dataset(data_root, 'train')
    dataset_val = MultiSenGE_dataset(data_root, 'valid')
    dataset_test = MultiSenGE_dataset(data_root, 'test')
    config=None

elif args.dataset=='MultiEarthDeforest':
    data_root = require_path(args.multiearth_path, '--multiearth_path')
    dataset_train = MultiEarthDeforest(data_root, 'train')
    dataset_val = MultiEarthDeforest(data_root, 'val')
    dataset_test = MultiEarthDeforest(data_root, 'test')
    if args.subset==True:
        dataset_train = torch.utils.data.Subset(dataset_train, range(0, len(dataset_train), 3))
    config = None

elif args.dataset=='kurosiwo':
    data_root = require_path(args.kurosiwo_path, '--kurosiwo_path')
    dataset_train = KuroSiwo_dataset(data_root, 'train')
    dataset_val = KuroSiwo_dataset(data_root, 'valid')
    dataset_test = KuroSiwo_dataset(data_root, 'test')
    config = None

elif args.dataset == 'MTLCC':
    data_root = require_path(args.mtlcc_path, '--mtlcc_path')
    dataset_train = ijgiDataset(data_root, tileids='tileids/train_fold0.tileids')
    dataset_val = ijgiDataset(data_root, tileids='tileids/test_fold0.tileids')
    dataset_test = ijgiDataset(data_root, tileids='tileids/eval.tileids')
    config = None
################################## sampler

if args.distributed == 'True':
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank,
            shuffle=False)  # shuffle=True to reduce monitor bias
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
else:
    num_tasks = 1
    global_rank = 0
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

################################### batch
if args.dataset=='MultiSenGE' and args.ts_len=='3':
    collate_fn=SenGE_pad_collate_3
elif args.dataset=='MultiSenGE' and args.ts_len=='12':
    collate_fn=SenGE_pad_collate_same
elif args.dataset=='MultiEarthDeforest':
    collate_fn=None
elif args.dataset=='kurosiwo':
    collate_fn=None
elif args.dataset == 'MTLCC':
    collate_fn=None


train_loader = DataLoader(
    dataset=dataset_train,
    batch_size=args.batch_size,
    sampler=sampler_train,
    num_workers=args.workers,
    drop_last=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    dataset=dataset_val,
    batch_size=args.batch_size,
    sampler=sampler_val,
    num_workers=args.workers,
    drop_last=False,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    dataset=dataset_test,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    drop_last=False,
    collate_fn=collate_fn
)

################################## loss
criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
################################## model
if args.model in ['TiMo']:
    model = SemsegFramework(args, classes=args.nb_classes)

model_without_ddp = model
#################### load checkpoint (backbone or whole model)
if args.ft and not args.eval:
    checkpoint = torch.load(args.ft, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.ft)

    checkpoint_model = checkpoint['model']
    state_dict = model.encoder.state_dict()
    print('state_dict',state_dict.keys())

  
    # TODO: Do something smarter?
    for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    if args.model=='TiMo':
        for k in ['patch_embed1.conv.0.weight']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]


    # interpolate position embedding
    interpolate_pos_embed(model.encoder, checkpoint_model)

    # load pre-trained model
    msg = model.encoder.load_state_dict(checkpoint_model, strict=False)
    # print(msg)
    print('msg',set(msg.missing_keys))

model.to(DEVICE)

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

eff_batch_size = args.batch_size * 1 * misc.get_world_size()

if main_process(args):
    logger.info('number of finetune params (M): %.2f' % (n_parameters / 1.e6))


################################## distributed
if args.distributed == 'True':
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK], find_unused_parameters=True)
    model_without_ddp = model.module

    if main_process(args):
        logger.info("Implementing distributed training!")
    seed = 2023 + LOCAL_RANK
    set_seeds(seed)
else:
    model = torch.nn.DataParallel(model)  # 普通的单机多卡
    patch_replication_callback(model)
    if main_process(args):
        logger.info("Implementing parallel training!")
################################## optimizer and scheduler

optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr * eff_batch_size / 256,
                              weight_decay=args.weight_decay)
loss_scaler = NativeScaler()

################################# ft & continue train

losses = []


if os.path.isfile(args.resume):
    if main_process(args):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch']
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        print('_')
    model_without_ddp.load_state_dict(checkpoint['model'],strict=False)
    
    if main_process(args):
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

if args.eval:
    import time
    time1=time.time()
    test_stats = evaluate(test_loader, model, DEVICE,args,config)
    time2=time.time()
    print(f"Evaluation of {args.model} on {len(dataset_test)} test images- allAcc: {test_stats['allAcc']:.4f},mF1:{test_stats['mF1']},mIoU:{test_stats['mIoU']}")
    print('time',time2-time1)
    exit(0)
##################################################### training #####################################################
max_accuracy = 0.0
for epoch in range(args.start_epoch, args.epochs):

    if args.distributed == 'True':
        train_loader.sampler.set_epoch(epoch)

    train_stats = train_one_epoch(
        model, criterion, train_loader,
        optimizer, DEVICE, epoch, loss_scaler,
        args.clip_grad, None,
        log_writer=None,
        args=args,logger=logger
    )

    test_stats = evaluate(val_loader, model, DEVICE,args,config)

    if args.output_dir:
        if test_stats["allAcc"] > max_accuracy:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        max_accuracy = max(max_accuracy, test_stats["allAcc"])
        print(f"Accuracy of the network {args.model} resume on {args.resume} on the {len(dataset_val)} test images: {test_stats['allAcc']:.4f}")
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

