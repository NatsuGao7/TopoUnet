'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Image_PINet(nn.Module):
    def __init__(self):
        super(Image_PINet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc = nn.Linear(1024, 2500)
        
        self.deconv1 = nn.ConvTranspose2d(
    in_channels=1,
    out_channels=256,
    kernel_size=3,
    stride=2,
    padding=1,
    output_padding=1  # Ensure the output size matches the target size
)
        self.bn_deconv1 = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(
    in_channels=256,
    out_channels=128,
    kernel_size=3,
    stride=2,
    padding=1,
    output_padding=1  # Ensure the output size matches the target size
)
        self.bn_deconv2 = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(
    in_channels=128,
    out_channels=64,
    kernel_size=3,
    stride=2,
    padding=1,
    output_padding=1  # Ensure the output size matches the target size
)
        self.bn_deconv3 = nn.BatchNorm2d(64)

        self.deconv4 = nn.ConvTranspose2d(
    in_channels=64,
    out_channels=3,
    kernel_size=3,
    stride=2,
    padding=1,
    output_padding=1  # Ensure the output size matches the target size
)
        self.bn_deconv4 = nn.BatchNorm2d(3)
        self.conv1x1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)  # Mapping 3 channels to 3 channels

    def forward(self, x):
        # Initial input shape: (b, 3, 500, 500)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (b, 128, 250, 250)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (b, 256, 125, 125)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (b, 512, 63, 63)
        x = F.relu(self.bn4(self.conv4(x)))             # (b, 1024, 63, 63)
    
        # Global average pooling
        x = self.global_pool(x)                         # (b, 1024, 1, 1)
        x = torch.flatten(x, 1)                         # (b, 1024)
    
        # Fully connected layer
        x = F.relu(self.fc(x))                          # (b, 2500)
    
        # Reshape to (b, 1, 50, 50)
        x = x.view(-1, 1, 50, 50)
        
    
        # 逐步反卷积，逐步恢复图像
        x = F.relu(self.bn_deconv1(self.deconv1(x)))    # (b, 256, 100, 100)
        
        x = F.relu(self.bn_deconv2(self.deconv2(x)))    # (b, 128, 200, 200)
        
        x = F.relu(self.bn_deconv3(self.deconv3(x)))    # (b, 64, 400, 400)
        
        x = F.relu(self.bn_deconv4(self.deconv4(x)))    # (b, 3, 800, 800)
        
        x = F.interpolate(x, size=(500, 500), mode='bicubic', align_corners=False)
        x = self.conv1x1(x)  # (b, 3, 500, 500)
        # Sigmoid activation function, ensure output is in range [0, 1]
        output = torch.sigmoid(x)  # (b, 4, 500, 500)
        # Add a V channel manually (255 for max value)
        v_channel = torch.ones(x.size(0), 1, 500, 500, device=x.device) 
        
        # Concatenate the V channel to the RGB output (only once)
        output = torch.cat((x, v_channel), dim=1)       # (b, 4, 500, 500)


        return output
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class Image_PINet(nn.Module):
    def __init__(self, pretrained=True):
        super(Image_PINet, self).__init__()

        # Encoder: Pretrained ResNet-50
        resnet = resnet50(pretrained=pretrained)
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.encoder1 = resnet.layer1  # (b, 256, H/4, W/4)
        self.encoder2 = resnet.layer2  # (b, 512, H/8, W/8)
        self.encoder3 = resnet.layer3  # (b, 1024, H/16, W/16)
        self.encoder4 = resnet.layer4  # (b, 2048, H/32, W/32)

        # Decoder: Transposed convolutions
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_deconv1 = nn.BatchNorm2d(1024)

        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_deconv2 = nn.BatchNorm2d(512)

        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_deconv3 = nn.BatchNorm2d(256)

        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_deconv4 = nn.BatchNorm2d(128)

        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_deconv5 = nn.BatchNorm2d(64)

        # Final 1x1 convolution to map to output channels (1 for single-channel output)
        self.conv1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        x = self.initial(x)  # Initial convolutional layer
        x = self.encoder1(x)  # Layer 1
        x = self.encoder2(x)  # Layer 2
        x = self.encoder3(x)  # Layer 3
        x = self.encoder4(x)  # Layer 4

        # Decoder
        x = F.relu(self.bn_deconv1(self.deconv1(x)))  # (b, 1024, H/16, W/16)
        x = F.relu(self.bn_deconv2(self.deconv2(x)))  # (b, 512, H/8, W/8)
        x = F.relu(self.bn_deconv3(self.deconv3(x)))  # (b, 256, H/4, W/4)
        x = F.relu(self.bn_deconv4(self.deconv4(x)))  # (b, 128, H/2, W/2)
        x = F.relu(self.bn_deconv5(self.deconv5(x)))  # (b, 64, H, W)

        # Resize output to match the input size (500x500)
        x = F.interpolate(x, size=(500, 500), mode='bilinear', align_corners=False)

        # Final output layer
        x = self.conv1x1(x)  # (b, 1, 500, 500)
        output = torch.sigmoid(x)  # Normalize output to [0, 1]
        output = output.squeeze(1) 
        return output

