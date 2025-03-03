import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src import PI_image
from src import topoUnet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # exclude background
    weights_path = "/home/zhuangzhigao/Desktop/Topo_Unet/er_clDice/best_model_epoch33_dice0.869.pth"
    img_root = "/home/zhuangzhigao/Desktop/deep-learning-for-image-processing/ER/test/images"
    #PI_root = "/home/zhuangzhigao/Desktop/Topo_Unet/DRIVE/test/PI"
    
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_root), f"image {img_root} not found."
    #assert os.path.exists(PI_root), f"PI {PI_root} not found."

    list_img = os.listdir(img_root)
    # using compute_mean_std.py
    mean = (0.01724677,0.01724677,0.01724677)
    std = (0.03484293,0.03484293,0.03484293)

    std = tuple(s if s > 0 else 1e-6 for s in std)


    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    # model = UNet(in_channels=3, num_classes=classes+1, base_c=32)
    PI_model = PI_image.Image_PINet()
    model = topoUnet.UNetWithPI(in_channels=3, pi_channels = 3,num_classes=classes+1, base_c=32)

    # load weights
    PI_model.load_state_dict(torch.load(weights_path, map_location='cpu')['PI_model'])
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    PI_model.to(device)
    model.to(device)

    for index in range(len(list_img)):
        # load image
        original_img = Image.open(os.path.join(img_root,list_img[index])).convert('RGB')

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            t_start = time_synchronized()
            PI = PI_model(img.to(device))
            PI = PI.unsqueeze(1).repeat(1, 3, 1, 1)
            output = model(img.to(device),PI.to(device))
            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start))
            
            prediction = output['out'].argmax(1).squeeze(0)  # 获取预测结果

            prediction = prediction.to("cpu").numpy().astype(np.uint8)

            # 将前景部分改为红色（255, 0, 0）
            img_array = np.array(original_img)
            img_array[prediction == 1] = [255, 0, 0]  # Red color for foreground

            # 合成图像：将原图和预测图像合成
            img_with_red = Image.fromarray(img_array)

            # 保存结果
            img_with_red.save(os.path.join("./er_ours_", list_img[index].split('.')[0]+'.png'))

if __name__ == '__main__':
    main()
