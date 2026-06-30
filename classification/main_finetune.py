# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
# import wandb
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter

import timm

# assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import misc as misc
from pos_embed import interpolate_pos_embed
from misc import NativeScalerWithGradNormCount as NativeScaler
import logging
import subprocess
import torch.distributed as dist
from sen12flood_dataset import Sen12Flood
import TiMo_cls


from engine_finetune import (train_one_epoch, train_one_epoch_temporal,
                             evaluate, evaluate_temporal)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model_type', default='TiMo', choices=['TiMo'],
                        help='Model family to fine-tune')
    parser.add_argument('--model', default='TiMo_base', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='images input size')
    parser.add_argument('--in_chans', type=int, default=3, help='image input channels')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.005, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.1)')


    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='Root directory of SEN12-FLOOD with S2list.json')
    
    parser.add_argument('--dataset', default='Sen12Flood', choices=['Sen12Flood'])

    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./outputs/classification',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./logs/classification',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--save_every', type=int, default=1, help='How frequently (in epochs) to save ckpt')
    parser.add_argument('--wandb', type=str, default=None,
                        help="Wandb project name, eg: sentinel_finetune")

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
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
    parser.add_argument('--amp', type=str, default='True', choices=['True', 'False'], help='whether to use amp')
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


    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    ################################################### setting ###################################################
    def main_process(args):
        return not args.distributed == 'True' or (args.distributed == 'True' and args.rank % args.world_size == 0)

    if args.distributed == 'True':

        # if 'MASTER_ADDR' not in os.environ:
        if 'SLURM_NTASKS' in os.environ.keys():
            # logger.info('#################### srun for DDP! ############################')
            # torch.multiprocessing.set_start_method('spawn')
            args.world_size = int(os.environ['SLURM_NTASKS'])
            # args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
            args.rank = int(os.environ['SLURM_PROCID'])  # if 'RANK' in os.environ else 0
            # args.rank = int(os.environ["RANK"])
            # args.rank = dist.get_rank()
            LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
            # LOCAL_RANK = int(os.environ['LOCAL_RANK'])
            # LOCAL_RANK = args.rank % torch.cuda.device_count()
            # IP = os.environ['SLURM_STEP_NODELIST']
            # DIST_URL = 'tcp://' + IP + ':' + str(port)
            torch.cuda.set_device(LOCAL_RANK)  # 设置节点等级为GPU数
            # os.environ['MASTER_PORT'] = args.port
            node_list = os.environ['SLURM_NODELIST']
            addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
            # os.environ['MASTER_ADDR'] = addr
            dist_url = 'tcp://%s:%s' % (addr, args.port)
            dist.init_process_group(backend='nccl', init_method=dist_url, world_size=args.world_size,
                                    rank=args.rank)  # 分布式TCP初始化

        else:
            # logger.info('#################### Launch for DDP! ############################')
            #     args.world_size = int(os.environ['SLURM_NTASKS'])
            if 'RANK' not in os.environ:
                raise RuntimeError('Distributed mode requires torchrun/srun environment variables. Use --distributed False for single-process runs.')
            args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
            #     #args.rank = int(os.environ['SLURM_PROCID']) #if 'RANK' in os.environ else 0
            args.rank = int(os.environ["RANK"])
            #     #args.rank = dist.get_rank()
            #     #LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
            #     #LOCAL_RANK = int(os.environ['LOCAL_RANK'])
            LOCAL_RANK = args.rank % torch.cuda.device_count()
            #     #IP = os.environ['SLURM_STEP_NODELIST']
            #     #DIST_URL = 'tcp://' + IP + ':' + str(port)
            torch.cuda.set_device(LOCAL_RANK)  # 设置节点等级为GPU数
            dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size,
                                    rank=args.rank)  # 分布式TCP初始化

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.dataset=='Sen12Flood':
        if not args.data_path:
            raise ValueError('Please set --data_path to the SEN12-FLOOD root directory.')
        dataset_train = Sen12Flood(args.data_path, 'train')
        dataset_val = Sen12Flood(args.data_path, 'val')
        
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
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = None#SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


    if args.dataset=='Sen12Flood':
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,  # args.num_workers
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        data_loader_test = data_loader_val

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # Define the model
    if args.model_type == 'TiMo':
        model = TiMo_cls.__dict__[args.model](
            in_chans=args.in_chans,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
        )


    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        # TODO: Do something smarter?
        for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        if args.model_type=='TiMo':
            for k in ['patch_embed1.conv.0.weight']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        for name, parameters in model.named_parameters():
            if name.startswith('SPE.blocks.0.attn.proj.weight') or name.startswith(
                    'tblocks.layers.0.self_attn.in_proj_weight'):
                print('load once')
                print(name, ':', parameters)

        # # TODO: change assert msg based on patch_embed
        # if args.global_pool:
        print(set(msg.missing_keys))


    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)


    if args.distributed == 'True':
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK],find_unused_parameters=True)
        model_without_ddp = model.module

    # build optimizer
    param_groups = model_without_ddp.parameters()

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate_temporal(data_loader_test, model, device, args)

        print(f"Evaluation on {len(data_loader_val)} test images- acc1: {test_stats['acc1']:.2f}%, "
              f"acc5: {test_stats['acc5']:.2f}%")
        exit(0)

    print('model_type', args.model_type)
    print('dataset_type', args.dataset_type)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed == 'True':
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_temporal(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=None,
            args=args,logger=logger
        )

        test_stats = evaluate_temporal(data_loader_val, model, device,args)

        if args.output_dir and  epoch + 1 == args.epochs:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")


        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

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
