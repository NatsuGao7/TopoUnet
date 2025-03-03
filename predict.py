import os
import time
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
from src import UNet
from src import PI_image
from scipy.io import loadmat
from src import topoUnet
import torch.nn.functional as F
from src.post_hoc import apply_gaussian_filter,apply_median_filter


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # exclude background
    weights_path = "/home/zhuangzhigao/Desktop/Topo_Unet/my_idea_final/best_model_epoch406_dice0.821.pth"
    img_root = "/home/zhuangzhigao/Desktop/Topo_Unet/DRIVE/test/images"
    mask_root = "/home/zhuangzhigao/Desktop/Topo_Unet/DRIVE/test/mask"
    #PI_root = "/home/zhuangzhigao/Desktop/Topo_Unet/DRIVE/test/PI"
    
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_root), f"image {img_root} not found."
    assert os.path.exists(mask_root), f"mask {mask_root} not found."
    #assert os.path.exists(PI_root), f"PI {PI_root} not found."

    list_img = os.listdir(img_root)
    list_mask = os.listdir(mask_root)
    #list_pi = os.listdir(PI_root)


    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    #  model = UNet(in_channels=3, num_classes=classes+1, base_c=32)
    # load weights
    PI_model = PI_image.Image_PINet()
    model = topoUnet.UNetWithPI(in_channels=3, pi_channels = 3,num_classes=classes+1, base_c=32)


    PI_model.load_state_dict(torch.load(weights_path, map_location='cpu')['PI_model'])
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    PI_model.to(device)
    model.to(device)
    for index in range(len(list_img)):
        # load roi mask

        roi_img = Image.open(os.path.join(mask_root,list_mask[index])).convert('L')
        roi_img = np.array(roi_img)
        #prefix = list_img[index].split('_')[0]
        #matching_files = [f for f in list_pi if f.startswith(prefix)]
        #PI = loadmat(os.path.join(PI_root,matching_files[0]))['image']

        #PI = Image.open(os.path.join(PI_root,matching_files[0]))

        

        # load image
        original_img = Image.open(os.path.join(img_root,list_img[index])).convert('RGB')

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        #PI = torch.tensor(PI, dtype=torch.float32)
        #PI = torch.unsqueeze(PI, dim=0)



        model.eval()  # 进入验证模式
        PI_model.eval()
        with torch.no_grad():
            # init model
            '''
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            PI_height, PI_width = PI.shape[-2:]
            init_PI = torch.zeros((1, 4, PI_height, PI_width), device=device)
            

            model(init_img,init_PI)
            '''
            t_start = time_synchronized()
            PI = PI_model(img.to(device))
            PI = PI.unsqueeze(1).repeat(1, 3, 1, 1)
            output = model(img.to(device),PI.to(device))
            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start))
            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            
            # 将前景对应的像素值改成255(白色)
            prediction[prediction == 1] = 255
            # 将不敢兴趣的区域像素设置成0(黑色)
            roi_img = np.where(roi_img > 128, 255, 0).astype(np.uint8)
            prediction[roi_img == 0] = 0
            mask = Image.fromarray(prediction)
            mask.save(os.path.join("./final_results/", list_img[index].split('_')[0] + '_result.png'))



if __name__ == '__main__':
    main()