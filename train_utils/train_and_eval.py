import torch
from torch import nn
import numpy as np
import os
from PIL import Image
from cb_loss import cbdice_loss 
import torch.nn.functional as F
import train_utils.distributed_utils as utils
from src import topofeature, topoUnet
from train_utils.clDice import clDice
from clLoss.cldice import soft_cldice
from train_utils.betti import calculate_betti
from src import topolayer
from PHLoss import topoloss
from .dice_coefficient_loss import dice_loss, build_target


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = False, ignore_index: int = -100):
    """
    Calculate the loss for segmentation using cross-entropy (and optionally Dice loss).
    
    Args:
        inputs (dict): Dictionary containing network outputs.
        target (torch.Tensor): Ground truth labels.
        loss_weight: Class weights for the cross-entropy loss.
        num_classes (int): Number of classes.
        dice (bool): Whether to include Dice loss.
        ignore_index (int): Label value to ignore.
    
    Returns:
        torch.Tensor: Combined loss value.
    """
    losses = {}
    for name, x in inputs.items():
        # Ignore pixels with target value equal to ignore_index (these are boundary or padding pixels)
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def topo_module(lh, gt):
    topoloss_sum = []
    # Print the shape of the input if needed
    lh_bu = lh
    gt_bu = gt
    # Process each image in the batch
    for i in range(0, len(lh)):
        print(i)
        lh_re = lh_bu[i, :, :, :].detach().cpu()
        lh_np = np.argmax(lh_re.numpy(), axis=0)
        lh = lh_np
        lh = torch.from_numpy(lh)

        gt_re = gt_bu[i, :, :, :].detach().cpu()
        gt_np = np.argmax(gt_re.numpy(), axis=0)
        gt = gt_np
        gt = torch.from_numpy(gt)

        loss_topo = topoloss_pytourch.getTopoLoss(lh, gt)
        topoloss_sum.append(loss_topo)

    loss_topo_out = 0
    for i in topoloss_sum:
        loss_topo_out = loss_topo_out + i
    print("Average topological loss:", loss_topo_out / len(topoloss_sum))
    return loss_topo_out / len(topoloss_sum)


def ce_topoloss(inputs, target, num_class_topo=int, ignore_index: int = -100):
    list_topo_loss = list()
    lh_bu = inputs
    gt_bu = target
    gt_bu = torch.where(gt_bu == ignore_index, torch.tensor(0, device=gt_bu.device), gt_bu)
    print(torch.unique(gt_bu))
    for i in range(0, len(inputs)):
        lh_bu = inputs[i, :, :, :]
        probabilities = F.softmax(lh_bu, dim=0)
        lh = probabilities[1, :, :]
        gt = gt_bu[i, :, :]
        
        loss_topo = topoloss.getTopoLoss(lh, gt)
        list_topo_loss.append(loss_topo)

    loss_topo_out = 0
    for i in list_topo_loss:
        loss_topo_out = loss_topo_out + i
    final_topo_loss = loss_topo_out / len(list_topo_loss)
    
    return final_topo_loss


def ce_CLloss(inputs, target, loss_weight=None, ignore_index: int = -100):
    cldice_loss = soft_cldice(iter_=3, smooth=1., exclude_background=False)
    ce_loss = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
    target = torch.where(target == ignore_index, torch.tensor(0, device=target.device), target)
    pred = torch.argmax(inputs, dim=1)  # Shape: (batch, height, width)
    
    pred = pred.float() 
    target = target.float()
    target = target.unsqueeze(1)  # Convert from (b, h, w) to (b, 1, h, w)
    pred = pred.unsqueeze(1)      # Shape: (batch, 1, h, w)
    cl_loss = cldice_loss(target, pred)
    
    total_loss = (1 - 0.2) * ce_loss + 0.2 * cl_loss
    return total_loss


def topo_module(lh, gt):
    topoloss_sum = []
    lh_bu = lh
    gt_bu = gt
    for i in range(0, len(lh)):
        lh_re = lh_bu[i, :, :, :].detach().cpu()
        lh_np = np.argmax(lh_re.numpy(), axis=0)
        lh = lh_np
        gt_re = gt_bu[i, :, :, :].detach().cpu()
        gt_np = np.argmax(gt_re.numpy(), axis=0)
        gt = gt_np
        lh = torch.Tensor(lh)
        gt = torch.Tensor(gt)

        loss_topo = topoloss.getTopoLoss(lh, gt)
        topoloss_sum.append(loss_topo)

    loss_topo_out = 0
    for i in topoloss_sum:
        loss_topo_out = loss_topo_out + i
    print("Average topological loss:", loss_topo_out / len(topoloss_sum))
    return loss_topo_out / len(topoloss_sum)


def ce_phLoss(inputs, target, loss_weight=None, ignore_index: int = -100):
    # Compute cross-entropy loss
    ce_loss = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
    
    # Replace ignore_index in target with 0
    target = torch.where(target == ignore_index, torch.tensor(0, device=target.device), target)
    target = target.unsqueeze(1)
    
    # Compute predictions using argmax
    pred = torch.argmax(inputs, dim=1)  # Shape: (batch, height, width)
    pred = pred.unsqueeze(1)
    pred = pred.float()
    target = target.float()

    # Initialize topological loss and batch size
    topo_loss = 0.0
    batch_size = pred.shape[0]
    # Calculate topological loss for each image in the batch
    for i in range(batch_size):
        pred_i = pred[i]
        target_i = target[i]
        topo_loss += topoloss.getTopoLoss(pred_i, target_i)
    
    # Compute average topological loss
    print(topo_loss)
    topo_loss = topo_loss / batch_size
    # Combine cross-entropy loss and topological loss
    total_loss = (1 - 0.5) * ce_loss + 0.5 * topo_loss

    topo_loss = topo_module(pred, target)
    loss = 0.6 * ce_loss + 0.4 * topo_loss
    return loss


def cb_loss(inputs, target, loss_weight=None, ignore_index: int = 255, include_dice: bool = True):
    """
    Compute the combined loss of Cross-Entropy, cbDice, and (optionally) standard Dice loss.

    Args:
        inputs: Model predictions (logits).
        target: Ground truth labels.
        loss_weight: Class weights for Cross-Entropy loss.
        ignore_index: Label to ignore (e.g., boundaries or padding).
        include_dice: Whether to include standard Dice loss (default True).

    Returns:
        Total weighted loss value.
    """
    # Compute Cross-Entropy (CE) loss
    ce_loss = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)

    # Process target by replacing ignore_index with 0
    target = torch.where(target == ignore_index, torch.tensor(0, device=target.device), target)

    # Compute cbDice loss
    softcb_dice_loss = cbdice_loss.SoftcbDiceLoss(iter_=10, smooth=1.0)
    target = target.float()
    target = target.unsqueeze(1)  # Convert from (b, h, w) to (b, 1, h, w)
    pred = F.softmax(inputs, dim=1).float()

    cb_dice_loss = softcb_dice_loss(inputs, target, t_skeletonize_flage=False)

    # Compute standard Dice loss (optional)
    dice_loss_value = 0
    if include_dice:
        dice_target = build_target(target.squeeze(1).long(), inputs.shape[1], ignore_index)  # Convert back to one-hot encoding
        dice_loss_value = dice_loss(inputs, dice_target, multiclass=True, ignore_index=ignore_index)

    # Combine total loss
    total_loss = ce_loss + cb_dice_loss + 0.5 * dice_loss_value
    return total_loss


def evaluate(model, PI_model, data_loader, device, num_classes):
    model.eval()
    PI_model.eval()
    # Initialize AUC-ROC calculator, confusion matrix, and Dice coefficient
    auc_roc_score = utils.AUCCalculator(num_classes, ignore_index=255)
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    list_clDice = list()
    list_betti_error_0 = list()
    list_betti_error_1 = list()
    header = 'Test:'
    with torch.no_grad():
        for image, target, _ in metric_logger.log_every(data_loader, 100, header):
            image, target, _ = image.to(device), target.to(device), _.to(device).float()
            
            PI_prediction = PI_model(image)
            PI_repeat = PI_prediction.unsqueeze(1).repeat(1, 3, 1, 1)
            
            output = model(image, PI_repeat)
            output = output['out']
            
            confmat.update(target.flatten(), output.argmax(1).flatten())
            auc_roc_score.update(target.flatten(), output)
            dice.update(output, target)
            list_clDice.append(clDice(output, target, ignore_index=255))
            b0_error, b1_erroe = calculate_betti(output, target, ignore_index=255)
            list_betti_error_0.append(b0_error)
            list_betti_error_1.append(b1_erroe)
            
        cleaned_list = [x for x in list_clDice if not np.isnan(x)]
        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()
        auc_roc_score.reduce_from_all_processes()
    return confmat, dice.value.item(), auc_roc_score, np.mean(cleaned_list), np.mean(list_betti_error_0), np.mean(list_betti_error_1)


def train_one_epoch(unet_model, PI_model, optimizer, optimizer_pi, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    unet_model.train()
    PI_model.train()
    loss = 0

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        # Set loss weights for background and foreground in cross-entropy loss (adjust based on your dataset)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    for i, (image, target, PI) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, target, PI = image.to(device), target.to(device), PI.to(device).float()
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            PI_prediction = PI_model(image)
            PI_repeat = PI_prediction.unsqueeze(1).repeat(1, 3, 1, 1)
            output = unet_model(image, PI_repeat)
            # Calculate loss based on the current epoch
            # For example, loss = ce_topoloss(output['out'], target, num_class_topo=num_classes, ignore_index=255)
            loss_1 = F.mse_loss(PI_prediction, PI)
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)
            # Example alternative loss functions:
            # loss_2 = cb_loss(output['out'], target, loss_weight, ignore_index=255)
            # loss_2 = ce_CLloss(output['out'], target, loss_weight, ignore_index=255)
            # loss = loss_2
            # loss_2 = ce_phLoss(output['out'], target, loss_weight, ignore_index=255)

            loss = loss_1 + loss_2

        optimizer.zero_grad()
        optimizer_pi.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(optimizer_pi)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            optimizer_pi.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        PI_lr = optimizer_pi.param_groups[0]["lr"]

        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr, PI_lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        Return a learning rate multiplier based on the step number.
        Note that PyTorch calls lr_scheduler.step() once before training starts.
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # During warmup, the learning rate multiplier increases from warmup_factor to 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # After warmup, the learning rate multiplier decreases from 1 to 0
            # Reference: DeepLab_v2 Learning Rate Policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
