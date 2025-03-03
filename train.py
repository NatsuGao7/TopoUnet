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
    model = topoUnet.UNetWithPI(in_channels=3, pi_channels=3, num_classes=num_classes, base_c=32)
    pi_model = PI_image.Image_PINet()
    return model, pi_model


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
    # Segmentation num_classes + background
    num_classes = args.num_classes + 1

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # File to store training and validation process information
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
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
    
    model, pi_model = create_model(num_classes)
    PI_model = PI_image.Image_PINet()
    PI_model.to(device)
    model.to(device)
    
    # Add parameters from both models to the optimizer
    params_to_optimize = []
    params_to_optimize.extend([p for p in model.parameters() if p.requires_grad])  # Parameters of model
    params_to_optimize.extend([p for p in pi_model.parameters() if p.requires_grad])  # Parameters of pi_model

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    optimizer_pi = optim.Adam(PI_model.parameters(), lr=args.PI_lr)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Create a learning rate update strategy, updating at each step (not each epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        PI_model.load_state_dict(checkpoint['PI_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_pi.load_state_dict(checkpoint['optimizer_pi'])  # Restore optimizer state of PI_model
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr, PI_lr = train_one_epoch(model, PI_model, optimizer, optimizer_pi, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat, dice, auc_roc_score, cl_dice, b0_error, b1_error = evaluate(model, PI_model, val_loader, device=device, num_classes=num_classes)
        
        print(f"The best Dice is: {best_dice}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time: {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch UNet training")
    parser.add_argument("--data-path", default="./", help="DRIVE root")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="Training device")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./Best_weight"):
        os.mkdir("./Best_weight")

    main(args)
