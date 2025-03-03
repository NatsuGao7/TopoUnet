import os
import time
import datetime
import torch.optim as optim
from src import topoUnet
from datasetER import ERDataset
from datasetOberon import OberonDataset
import torch
import random
import numpy as np
from src import unet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import DriveDataset
from src import PI_image
import transforms as T


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 512
    crop_size = 500

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)




def create_model(num_classes):
    # model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    model = topoUnet.UNetWithPI(in_channels=3, pi_channels = 3,num_classes=num_classes, base_c=32)
    pi_model = PI_image.Image_PINet()
    return model,pi_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

 
    mean=[0.32920446, 0.32920446, 0.32920446]
    std=[0.09668911, 0.09668911, 0.09668911]
    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    '''
    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True))

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    '''
    # using compute_mean_std.py

    train_dataset = OberonDataset(args.data_path,
                                 train=True,
                                 val = False,
                                 test = False,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = OberonDataset(args.data_path,
                               train=False,
                               val = True,
                               test = False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    
    model, pi_model = create_model(num_classes)
    PI_model = PI_image.Image_PINet()
    PI_model.to(device)
    model.to(device)
    

    # 在优化器中添加两个模型的参数
    params_to_optimize = []
    params_to_optimize.extend([p for p in model.parameters() if p.requires_grad])  # model 的参数
    params_to_optimize.extend([p for p in pi_model.parameters() if p.requires_grad])  # pi_model 的参数

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    optimizer_pi = optim.Adam(PI_model.parameters(), lr=args.PI_lr)


    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        PI_model.load_state_dict(checkpoint['PI_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_pi.load_state_dict(checkpoint['optimizer_pi'])  # 恢复 PI_model 的优化器状态
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr,PI_lr = train_one_epoch(model, PI_model,optimizer,optimizer_pi, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat, dice,auc_roc_score,cl_dice,b0_error,b1_error = evaluate(model,PI_model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"Dice coefficient: {dice:.6f}")
        print(f"AUC ROC Score is: {auc_roc_score.compute():.6f}")
        print(f"ClDice Score is: {cl_dice:.6f}")
        print(f"betti 0 error is: {b0_error:.6f}")
        print(f"betti 1 error is: {b1_error:.6f}")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
             f"train_loss: {mean_loss:.4f}\n" \
             f"lr: {lr:.6f}\n" \
             f"PI_lr:{PI_lr:.6f}\n"\
             f"dice coefficient: {dice:.6f}\n" \
             f"AUC-ROC: {auc_roc_score.compute():.6f}\n"\
             f"ClDice Score is: {cl_dice:.6f}\n"\
             f"betti 0 error is: {b0_error:.6f}\n"\
             f"betti 1 error is: {b1_error:.6f}\n"
            f.write(train_info + val_info + "\n\n")
        

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue
        with open(args.save_metric, "a") as bf:
            best_info = f"[epoch: {epoch}]\n" \
                        f"dice coefficient: {dice:.6f}\n" \
                        f"AUC-ROC: {auc_roc_score.compute():.6f}\n" \
                        f"ClDice Score: {cl_dice:.6f}\n" \
                        f"confusion matrix:\n{val_info}\n" \
                        f"betti 0 error is: {b0_error:.6f}\n"\
                        f"betti 1 error is: {b1_error:.6f}\n"\
                        f"model saved to: {args.save_metric}\n\n"
            bf.write(best_info)

        print(f'The best Dice is:{best_dice}')

        save_file = {"model": model.state_dict(),
                     "PI_model": PI_model.state_dict(),  # 添加 PI_model 的权重
                     "optimizer": optimizer.state_dict(),
                     "optimizer_pi":optimizer_pi.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file,f"./my_Oberon_cb/best_model_epoch{epoch}_dice{dice:.3f}.pth")
        else:
            continue

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=1000, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--PI-lr', default=0.001, type=float, help='initial PI learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    parser.add_argument('--save-metric', default='./my_Oberon_cb.txt', type=str)
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./my_Oberon_cb"):
        os.mkdir("./my_Oberon_cb")

    main(args)
