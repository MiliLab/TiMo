# https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/eb6d04f118f711e50613633bfae9b6220fc2969e/MAEPretrain_SceneClassification/main_pretrain.py
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn


import misc
from misc import NativeScalerWithGradNormCount as NativeScaler

from engine_pretrain import train_one_epoch


from tpco_dataset import TPCO_npy_ts
from torchvision import transforms

import TiMo_mae

import logging
import subprocess
import torch.distributed as dist


def get_args_parser():
    parser = argparse.ArgumentParser('FLIP pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='TiMo_base_tspos', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_type', default='TiMo', type=str, choices=['TiMo'],
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='Directory containing per-location .npy time-series files')
    parser.add_argument('--csv_path', default='', type=str,
                        help='CSV with loc and timestamp columns t1...t10')
    parser.add_argument('--num_frames', default=3, type=int,
                        help='Number of temporal frames sampled per location')
    parser.add_argument('--subset', default='False', type=str)
    parser.add_argument('--output_dir', default='./outputs/pretrain',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./logs/pretrain',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', type=str, default='False', choices=['True', 'False'],
                        help='distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--port', type=str, default='40003', help='master ports')
    parser.add_argument('--amp', type=str, default='True',choices=['True', 'False'], help='whether to use amp')

    # dataset
    parser.add_argument('--dataset', default='millionst', type=str, choices=['millionst'], help='type of dataset')

    #other setting
    parser.add_argument("--in_channels", default=3, type=int, help='input channels; current release supports RGB pre-training')

    return parser


def main(args):
    args.rank = 0
    LOCAL_RANK = getattr(args, 'local_rank', 0)
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

    ################################################### setting ###################################################

    if args.distributed == 'True':

        # if 'MASTER_ADDR' not in os.environ:
        if 'SLURM_NTASKS' in os.environ.keys():
            # logger.info('#################### srun for DDP! ############################')

            args.world_size = int(os.environ['SLURM_NTASKS'])
            args.rank = int(os.environ['SLURM_PROCID'])  # if 'RANK' in os.environ else 0
            LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
            torch.cuda.set_device(LOCAL_RANK)  # 设置节点等级为GPU数
            node_list = os.environ['SLURM_NODELIST']
            addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
            dist_url = 'tcp://%s:%s' % (addr, args.port)
            dist.init_process_group(backend='nccl', init_method=dist_url, world_size=args.world_size,
                                    rank=args.rank)  # 分布式TCP初始化

        else:
            # logger.info('#################### Launch for DDP! ############################')
            if 'RANK' not in os.environ:
                raise RuntimeError('Distributed mode requires torchrun/srun environment variables. Use --distributed False for single-process runs.')
            args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
            args.rank = int(os.environ["RANK"])
            LOCAL_RANK = args.rank % torch.cuda.device_count()

            torch.cuda.set_device(LOCAL_RANK)  # 设置节点等级为GPU数
            dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size,
                                    rank=args.rank)  # 分布式TCP初始化

    if main_process(args):
        logger.info('<<<<<<<<<<<<<<<<< args <<<<<<<<<<<<<<<<<')
        logger.info(args)

    # misc.init_distributed_mode(args) #ignoreddd

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True


    if not args.data_path or not args.csv_path:
        raise ValueError('Please set --data_path and --csv_path for MillionST/TPCO pre-training data.')

    dataset_train = TPCO_npy_ts(root=args.data_path, csv_path=args.csv_path, num_ts=args.num_frames)

    if args.subset=='True':
        dataset_train=torch.utils.data.Subset(dataset_train,range(0,100))
    # output folder
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, args.dataset + '_' + str(args.input_size),
                                       str(args.epochs) + '_' + str(args.mask_ratio) + '_' + str(args.blr) + '_' + str(
                                           args.weight_decay) + '_' + str(args.batch_size)+ '_'+str(args.in_channels))
        os.makedirs(args.output_dir, exist_ok=True)

    # print(dataset_train)

    if args.distributed == 'True':
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)


    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )


    model = TiMo_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)


    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256  # 累积iter, lr会增加

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed == 'True':
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed == 'True':
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,logger=logger
        )

        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
