from collections import defaultdict, deque
import datetime
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
import torch
import errno
import numpy as np
import os

from .dice_coefficient_loss import multiclass_dice_coeff, build_target


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class AUCCalculator(object):
    def __init__(self, num_classes, ignore_index=None):
        """
        num_classes: Number of classes for classification.
        ignore_index: Label value to ignore, default is None.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index  # New ignore_index parameter
        self.all_labels = []
        self.all_preds = []

    def update(self, a, b):
        """
        Update the labels and predictions.

        a: Ground truth tensor (1D)
        b: Predictions (logits tensor, typically of shape [batch_size, num_classes])
        """
        with torch.no_grad():
            # Extract positive class probabilities (suitable for binary classification)
            pred_probs = F.softmax(b, dim=1)[:, 1, :, :]  # Extract positive class probability
            pred_probs = pred_probs.flatten()
            labels = a.flatten()

            # If ignore_index is provided, filter out the corresponding pixels
            if self.ignore_index is not None:
                valid_mask = labels != self.ignore_index  # Valid pixel mask
                labels = labels[valid_mask]
                pred_probs = pred_probs[valid_mask]
            
            self.all_labels.append(labels.cpu().numpy())  # Collect valid labels
            self.all_preds.append(pred_probs.cpu().numpy())  # Collect valid prediction probabilities

    def reset(self):
        """Reset all collected labels and prediction values."""
        self.all_labels = []
        self.all_preds = []

    def compute(self):
        """
        Compute the AUC-ROC score.
        """
        # Concatenate all labels and prediction probabilities into one array
        all_labels = np.concatenate(self.all_labels, axis=0)
        all_preds = np.concatenate(self.all_preds, axis=0)

        # Compute AUC-ROC using sklearn
        auc = roc_auc_score(all_labels, all_preds)
        return auc

    def reduce_from_all_processes(self):
        """
        Synchronize results across distributed processes.
        Merge the 'all_labels' and 'all_preds' data from all processes.
        """
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return

        # Convert to tensor
        all_labels_tensor = torch.tensor(np.concatenate(self.all_labels, axis=0), dtype=torch.float32, device='cuda')
        all_preds_tensor = torch.tensor(np.concatenate(self.all_preds, axis=0), dtype=torch.float32, device='cuda')

        # Use distributed all_reduce to synchronize data
        torch.distributed.all_reduce(all_labels_tensor)
        torch.distributed.all_reduce(all_preds_tensor)

        # Update data
        self.all_labels = [all_labels_tensor.cpu().numpy()]
        self.all_preds = [all_preds_tensor.cpu().numpy()]

    def __str__(self):
        """Return a string representation of the AUC-ROC score."""
        auc = self.compute()
        return str(auc)


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # Create confusion matrix
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # Identify valid pixel indices in ground truth
            k = (a >= 0) & (a < n)
            # Count number of pixels where true label a[k] is predicted as label b[k]
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # Compute global accuracy
        acc_global = torch.diag(h).sum() / h.sum()
        # Compute Intersection over Union (IoU)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, iu.mean()

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, mean_iou = self.compute()
        return (
            'Global Accuracy: {:.6f}\n'
            'Mean IoU: {:.6f}'
        ).format(
            acc_global.item() * 100,
            mean_iou.item() * 100
        )


class DiceCoefficient(object):
    def __init__(self, num_classes: int = 2, ignore_index: int = -100):
        self.cumulative_dice = None
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.count = None

    def update(self, pred, target):
        if self.cumulative_dice is None:
            self.cumulative_dice = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        if self.count is None:
            self.count = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        # Compute the Dice score, ignoring background
        pred = F.one_hot(pred.argmax(dim=1), self.num_classes).permute(0, 3, 1, 2).float()
        dice_target = build_target(target, self.num_classes, self.ignore_index)
        self.cumulative_dice += multiclass_dice_coeff(pred[:, 1:], dice_target[:, 1:], ignore_index=self.ignore_index)
        self.count += 1

    @property
    def value(self):
        if self.count == 0:
            return 0
        else:
            return self.cumulative_dice / self.count

    def reset(self):
        if self.cumulative_dice is not None:
            self.cumulative_dice.zero_()

        if self.count is not None:
            self.count.zeros_()

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.cumulative_dice)
        torch.distributed.all_reduce(self.count)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in the master process.
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_proce
