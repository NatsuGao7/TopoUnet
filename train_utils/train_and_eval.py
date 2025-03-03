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
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']



'''
def topo_module(lh, gt):
    topoloss_sum = []
    # print(lh.shape)
    lh_bu = lh
    gt_bu = gt
    # print("lh:",lh,gt)
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
        # print("loss_topo:",loss_topo)
        topoloss_sum.append(loss_topo)

    loss_topo_out = 0
    # print("topoloss_sum:",topoloss_sum)
    for i in topoloss_sum:
        loss_topo_out = loss_topo_out + i
    print("topoloss average:", loss_topo_out / len(topoloss_sum))
    return loss_topo_out / len(topoloss_sum)
'''
'''
def ce_topoloss(inputs, target, num_class_topo = int, ignore_index: int = -100):
    list_topo_loss = list()
    lh_bu = inputs
    gt_bu = target
    gt_bu = torch.where(gt_bu == ignore_index, torch.tensor(0, device=gt_bu.device), gt_bu)
    print(torch.unique(gt_bu))
    for i in range(0, len(inputs)):
        lh_bu = inputs[i,:,:, :]
        probabilities = F.softmax(lh_bu, dim=0)
        lh = probabilities[1, :, :]
        gt = gt_bu[i, :, :]
        

        loss_topo = topoloss.getTopoLoss(lh, gt)
        list_topo_loss.append(loss_topo)

    loss_topo_out = 0
    # print("topoloss_sum:",topoloss_sum)
    for i in list_topo_loss:
        loss_topo_out = loss_topo_out + i
    final_topo_loss = loss_topo_out / len(list_topo_loss)

    
    return final_topo_loss
'''

def ce_CLloss(inputs, target, loss_weight=None,ignore_index: int = -100):
    cldice_loss = soft_cldice(iter_=3, smooth=1., exclude_background=False)
    ce_loss = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
    target = torch.where(target == ignore_index, torch.tensor(0, device=target.device), target)
    pred = torch.argmax(inputs, dim=1)  # Shape: (batch, 584, 565)
    
    pred = pred.float() 
    target = target.float()
    target = target.unsqueeze(1)  # 从 (b, h, w) 到 (b, 1, h, w)
    pred = pred.unsqueeze(1) # Shape: (batch, 584, 565)
    cl_loss = cldice_loss(target, pred)
    

    total_loss = (1-0.2)*ce_loss+0.2*cl_loss
    return total_loss

def topo_module(lh,gt):
    topoloss_sum=[]
    # print(lh.shape)
    lh_bu=lh
    gt_bu=gt
    # print("lh:",lh,gt)
    for i in range(0,len(lh)):
        # print(i)
        lh_re = lh_bu[i, :, :, :].detach().cpu()
        # print("lh_re:",lh_re)
        lh_np = np.argmax(lh_re.numpy(), axis=0)
        lh=lh_np
        # print("lh_after_split:",lh)
        # lh=torch.from_numpy(lh)

        gt_re = gt_bu[i, :, :, :].detach().cpu()
        # print("gt_re:",gt_re.shape)
        gt_np = np.argmax(gt_re.numpy(), axis=0)
        # print("gt_np:",gt_np.shape)
        gt = gt_np
        # gt=torch.from_numpy(gt)
        # print("lh,gt:",lh,gt)
        lh=torch.Tensor(lh)
        gt=torch.Tensor(gt)

        loss_topo = topoloss.getTopoLoss(lh, gt)
        # print("loss_topo:",loss_topo)
        # loss_topo = Variable(loss_topo, requires_grad=True)
        topoloss_sum.append(loss_topo)

    loss_topo_out=0
    # print("topoloss_sum:",topoloss_sum)
    for i in topoloss_sum:
        # print("out_topo::",i)
        loss_topo_out=loss_topo_out+i
    print("topoloss average:",loss_topo_out/len(topoloss_sum))
    return loss_topo_out/len(topoloss_sum)

def ce_phLoss(inputs, target, loss_weight=None, ignore_index: int = -100):
    # 计算交叉熵损失
    ce_loss = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
    
    # 处理 ignore_index：将其替换为 0
    target = torch.where(target == ignore_index, torch.tensor(0, device=target.device), target)
    target = target.unsqueeze(1)
    
    # 计算预测结果，应用sigmoid并找到最大概率的类别
    # pred = torch.sigmoid(inputs)  # Shape: (batch, 1, 500, 500)
    pred = torch.argmax(inputs, dim=1)  # Shape: (batch, 584, 565)
    pred = pred.unsqueeze(1)
    pred = pred.float()
    target = target.float()
     #print(pred.shape)
    # 对sigmoid后的概率取最大值（如果是二分类，可以用 torch.sigmoid(inputs) > 0.5 代替）
    #pred, max_indices = torch.max(pred, dim=1, keepdim=True)  # Shape: (batch, 1, 500, 500)
    # 去掉预测维度中的 1（如果是二分类，实际上可以直接把1维去掉）
     #pred = pred.squeeze(1)
    # 确保类型是 float 类型
    # 初始化拓扑损失和 batch 大小
    '''
    topo_loss = 0.0
    batch_size = pred.shape[0]
    # 遍历 batch 维度，计算每张图像的拓扑损失
    for i in range(batch_size):
        pred_i = pred[i]  # 取出第 i 张预测
        target_i = target[i]  # 取出第 i 张目标
        # 计算拓扑损失（替换为实际拓扑损失函数）
        topo_loss += topoloss.getTopoLoss(pred_i, target_i)  # 加上当前的损失
    
    # 计算平均拓扑损失
    print(topo_loss)
    topo_loss = topo_loss / batch_size
    # 返回总损失（交叉熵损失 + 拓扑损失）
    total_loss = (1 - 0.5) * ce_loss + 0.5 * topo_loss
    '''
    topo_loss = topo_module(pred,target)
    loss=0.6*ce_loss+0.4*topo_loss
    return loss

def cb_loss(inputs, target, loss_weight=None, ignore_index: int = 255, include_dice: bool = True):
    """
    计算 CE, cbDice 和（可选的）普通 Dice 损失。

    Args:
        inputs: 模型的预测输出 (logits)。
        target: 目标标签。
        loss_weight: 用于加权 CE 损失的权重 (class weights)。
        ignore_index: 忽略目标标签中指定的索引（如边缘或填充）。
        include_dice: 是否加入普通 Dice 损失（默认 True）。

    Returns:
        加权组合的总损失值。
    """
    # 计算 Cross-Entropy (CE) 损失
    ce_loss = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)

    # 处理 target，忽略指定的 ignore_index
    target = torch.where(target == ignore_index, torch.tensor(0, device=target.device), target)

    # 计算 cbDice 损失
    softcb_dice_loss = cbdice_loss.SoftcbDiceLoss(iter_=10, smooth=1.0)
    target = target.float()
    target = target.unsqueeze(1)  # 从 (b, h, w) 转换到 (b, 1, h, w)
    pred = F.softmax(inputs, dim=1).float()

    cb_dice_loss = softcb_dice_loss(inputs, target, t_skeletonize_flage=False)

    # 计算普通 Dice 损失（可选）
    dice_loss_value = 0
    if include_dice:
        dice_target = build_target(target.squeeze(1).long(), inputs.shape[1], ignore_index)  # 转回 one-hot 编码格式
        dice_loss_value = dice_loss(inputs, dice_target, multiclass=True, ignore_index=ignore_index)

    # 综合总损失
    total_loss = ce_loss + cb_dice_loss + 0.5 * dice_loss_value
    return total_loss


def evaluate(model,PI_model, data_loader, device, num_classes):
    model.eval()
    PI_model.eval()
    # AUC_ROC_values = utils.AUCMeterBinary()
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
            
            output = model(image,PI_repeat)
            output = output['out']
            
            confmat.update(target.flatten(), output.argmax(1).flatten())
            auc_roc_score.update(target.flatten(),output)
            dice.update(output, target)
            list_clDice.append(clDice(output,target,ignore_index=255))
            b0_error,b1_erroe = calculate_betti(output,target,ignore_index=255)
            list_betti_error_0.append(b0_error)
            list_betti_error_1.append(b1_erroe)
            #list_betti_error_0 = [0]
            #list_betti_error_1 = [0]

        cleaned_list = [x for x in list_clDice if not np.isnan(x)]
        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()
        auc_roc_score.reduce_from_all_processes()
    return confmat, dice.value.item(),auc_roc_score,np.mean(cleaned_list),np.mean(list_betti_error_0),np.mean(list_betti_error_1)
'''
def evaluate(model, PI_model, data_loader, device, num_classes, save_dir="./final_results/"):
    model.eval()
    PI_model.eval()
    
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
            
            # Update metrics
            confmat.update(target.flatten(), output.argmax(1).flatten())
            auc_roc_score.update(target.flatten(), output)
            dice.update(output, target)
            list_clDice.append(clDice(output, target, ignore_index=255))
            b0_error, b1_error = calculate_betti(output, target, ignore_index=255)
            list_betti_error_0.append(b0_error)
            list_betti_error_1.append(b1_error)
            
            # Save the visualization of the output for each batch
            for idx in range(image.size(0)):  # Iterate over the batch
                # Convert the prediction to a binary mask (0 or 255)
                prediction = output[idx].argmax(0).cpu().numpy()
                prediction[prediction == 1] = 255  # Set foreground to 255 (white)
                prediction = Image.fromarray(prediction.astype(np.uint8))

                # Create a filename for saving
                image_name = f"{str(idx).zfill(2)}_result.png"  # Using idx to create filenames like "01_result.png"

                prediction.save(os.path.join(save_dir, image_name))
                
        # Reduce metrics across all processes if using distributed training
        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()
        auc_roc_score.reduce_from_all_processes()

    return confmat, dice.value.item(), auc_roc_score, np.mean(list_clDice), np.mean(list_betti_error_0), np.mean(list_betti_error_1)
'''
def train_one_epoch(unet_model, PI_model,optimizer,optimizer_pi, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    unet_model.train()
    PI_model.train()
    loss = 0

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    for i, (image, target,PI) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        image, target, PI = image.to(device), target.to(device), PI.to(device).float()
        
    
        
        # topo_1 = topofeature.extract_topological_features_one(target).to(device)
        # topo_0 = topofeature.extract_topological_features_zero(target).to(device)
        
        # topo_feature = topofeature.extract_topological_features_batch(target)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            PI_prediction = PI_model(image)
            
            PI_repeat = PI_prediction.unsqueeze(1).repeat(1, 3, 1, 1)

            
            output = unet_model(image,PI_repeat)
            # print(output['out'].shape, target.shape)
            # 根据 epoch 决定使用的 loss 函数
            # if epoch > 15:
            #loss = ce_topoloss(output['out'], target, num_class_topo=num_classes, ignore_index=255)
            # else:
            loss_1 = F.mse_loss(PI_prediction,PI)
            
            #loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)
            loss_2 = cb_loss(output['out'], target,loss_weight,ignore_index=255)
            #loss_2 = ce_CLloss(output['out'], target, loss_weight,ignore_index=255)
            #loss_2 = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)
            #loss_2 =  ce_phLoss(output['out'], target, loss_weight,ignore_index=255)

            loss = loss_1+loss_2

            

        optimizer.zero_grad()
        optimizer_pi.zero_grad()
        #optimizer_pi.zero_grad()
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

    return metric_logger.meters["loss"].global_avg, lr,PI_lr


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
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
