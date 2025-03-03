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
from src.post_hoc import apply_gaussian_filter, apply_median_filter


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # Exclude background
    weights_path = "./my_idea_final/best_model_epoch406_dice0.821.pth"
    img_root = "./Topo_Unet/DRIVE/test/images"
    mask_root = "./DRIVE/test/mask"

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_root), f"image {img_root} not found."
    assert os.path.exists(mask_root), f"mask {mask_root} not found."

    list_img = os.listdir(img_root)
    list_mask = os.listdir(mask_root)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    # Create model
    PI_model = PI_image.Image_PINet()
    model = topoUnet.UNetWithPI(in_channels=3, pi_channels=3, num_classes=classes+1, base_c=32)

    # Load weights
    PI_model.load_state_dict(torch.load(weights_path, map_location='cpu')['PI_model'])
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    PI_model.to(device)
    model.to(device)

    for index in range(len(list_img)):
        # Load ROI mask
        roi_img = Image.open(os.path.join(mask_root, list_mask[index])).convert('L')
        roi_img = np.array(roi_img)
        
        # Load image
        original_img = Image.open(os.path.join(img_root, list_img[index])).convert('RGB')

        # Transform image from PIL to tensor and normalize
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        img = data_transform(original_img)
        # Expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()
        PI_model.eval()
        with torch.no_grad():
            # Perform inference
            t_start = time_synchronized()
            PI = PI_model(img.to(device))
            PI = PI.unsqueeze(1).repeat(1, 3, 1, 1)
            output = model(img.to(device), PI.to(device))
            t_end = time_synchronized()
            print("Inference time: {}".format(t_end - t_start))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            
            # Convert foreground pixels to 255 (white)
            prediction[prediction == 1] = 255
            # Set non-region of interest pixels to 0 (black)
            roi_img = np.where(roi_img > 128, 255, 0).astype(np.uint8)
            prediction[roi_img == 0] = 0
            
            # Save the result mask
            mask = Image.fromarray(prediction)
            mask.save(os.path.join("./final_results/", list_img[index].split('_')[0] + '_result.png'))


if __name__ == '__main__':
    main()
