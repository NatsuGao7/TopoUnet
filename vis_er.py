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
    classes = 1  # Excluding background
    weights_path = "./er_clDice/best_model_epoch33_dice0.869.pth"
    img_root = "./ER/test/images"

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_root), f"image {img_root} not found."

    list_img = os.listdir(img_root)
    # Using compute_mean_std.py
    mean = (0.01724677, 0.01724677, 0.01724677)
    std = (0.03484293, 0.03484293, 0.03484293)

    std = tuple(s if s > 0 else 1e-6 for s in std)

    # Get device (GPU if available, otherwise CPU)
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
        # Load image
        original_img = Image.open(os.path.join(img_root, list_img[index])).convert('RGB')

        # Convert PIL image to tensor and normalize
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        
        # Expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # Switch to evaluation mode
        with torch.no_grad():
            t_start = time_synchronized()
            PI = PI_model(img.to(device))
            PI = PI.unsqueeze(1).repeat(1, 3, 1, 1)
            output = model(img.to(device), PI.to(device))
            t_end = time_synchronized()
            print("Inference time: {}".format(t_end - t_start))

            prediction = output['out'].argmax(1).squeeze(0)  # Get prediction result

            prediction = prediction.to("cpu").numpy().astype(np.uint8)

            # Change the foreground area to red (255, 0, 0)
            img_array = np.array(original_img)
            img_array[prediction == 1] = [255, 0, 0]  # Red color for the foreground

            # Combine the original image and the prediction image
            img_with_red = Image.fromarray(img_array)

            # Save the result
            img_with_red.save(os.path.join("./er_ours_", list_img[index].split('.')[0] + '.png'))


if __name__ == '__main__':
    main()
