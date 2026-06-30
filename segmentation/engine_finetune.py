import math
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist

import misc as misc
from utils import AverageMeter, intersectionAndUnionGPU


def _flatten_multiearth_targets(targets, args):
    if args.dataset == 'MultiEarthDeforest':
        return targets.view(-1, *targets.shape[2:])
    return targets


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn=None, log_writer=None, args=None, logger=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    for data_iter_step, (samples, dates, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header, logger)):
        samples = samples.to(device, non_blocking=True)
        dates = dates.to(device, non_blocking=True)
        targets = targets.long().to(device, non_blocking=True)
        targets = _flatten_multiearth_targets(targets, args)

        if args.amp == 'True':
            with torch.cuda.amp.autocast():
                outputs = model(samples, dates)
                loss = criterion(outputs, targets)
        else:
            outputs = model(samples, dates)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            if logger is not None:
                logger.info(model.state_dict())
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=True)
        optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        max_lr = max(group["lr"] for group in optimizer.param_groups)
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, config):
    print('Begin evaluation')
    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    predict_meter = AverageMeter()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    y_true = []
    y_pred = []
    last_loss = 0.0

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0].to(device, non_blocking=True)
        dates = batch[1].to(device, non_blocking=True)
        target = batch[-1].long().to(device, non_blocking=True)
        target = _flatten_multiearth_targets(target, args)

        if args.amp == 'True':
            with torch.cuda.amp.autocast():
                output = model(images, dates)
                loss = criterion(output, target)
        else:
            output = model(images, dates)
            loss = criterion(output, target)

        last_loss = loss.item()
        output = output.max(1)[1]

        if args.dataset in ['MultiSenGE', 'kurosiwo']:
            y_true.append(target.cpu().detach().numpy())
            y_pred.append(output.cpu().detach().numpy())

        intersection, union, target_area, predict = intersectionAndUnionGPU(
            output, target, args.nb_classes, args.ignore_label)

        if misc.is_dist_avail_and_initialized():
            dist.all_reduce(intersection)
            dist.all_reduce(union)
            dist.all_reduce(target_area)
            dist.all_reduce(predict)

        intersection = intersection.cpu().numpy()
        union = union.cpu().numpy()
        target_area = target_area.cpu().numpy()
        predict = predict.cpu().numpy()

        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target_area)
        predict_meter.update(predict)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    precise_class = intersection_meter.sum / (predict_meter.sum + 1e-10)
    f1_class = 2 * (precise_class * accuracy_class) / (precise_class + accuracy_class + 1e-10)

    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    mF1 = np.mean(f1_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    metric_logger.update(loss=last_loss)
    metric_logger.meters['mIoU'].update(mIoU)
    metric_logger.meters['mAcc'].update(mAcc)
    metric_logger.meters['mF1'].update(mF1)
    metric_logger.meters['allAcc'].update(allAcc)

    if args.dataset in ['MultiSenGE', 'kurosiwo'] and y_true:
        from sklearn.metrics import cohen_kappa_score
        y_true = np.concatenate(y_true, 0).flatten()
        y_pred = np.concatenate(y_pred, 0).flatten()
        kappa = cohen_kappa_score(y_true[y_true != args.ignore_label], y_pred[y_true != args.ignore_label])
        metric_logger.meters['kappa'].update(kappa)
        print('Kappa:', kappa)

    metric_logger.synchronize_between_processes()
    print('mIou:', metric_logger.mIoU, 'mAcc:', metric_logger.mAcc,
          'mF1:', metric_logger.mF1, 'allAcc:', metric_logger.allAcc,
          'IoU:', iou_class, 'F1:', f1_class)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
