import torch

class H0Mapper(torch.nn.Module):
    def __init__(self, cnn_channels, target_resolution):
        """
        将 H0 映射到 CNN 特征图分辨率和通道数。
        
        参数:
            cnn_channels (int): CNN 的目标通道数。
            target_resolution (tuple): CNN 特征图的空间分辨率 (H, W)。
        """
        
        super(H0Mapper, self).__init__()
        self.target_resolution = target_resolution  # (H, W)
        self.channel_mapping = torch.nn.Linear(2, cnn_channels)  # 将 H0 转为 C 通道
        self.resize = torch.nn.AdaptiveAvgPool2d(target_resolution)  # 调整空间分辨率

    def forward(self, h0_features):
        """
        前向传播，处理 H0 特征。
        
        参数:
            h0_features (torch.Tensor): 持久性条形码，形状为 (B, N, 2)。
        
        返回:
            torch.Tensor: 映射到 CNN 特征图形状的张量 (B, C, H, W)。
        """
        print(type(h0_features))
        batch_size, num_points, _ = h0_features.shape
        
        # 将每个持久性点映射到 CNN 通道空间
        h0_mapped = self.channel_mapping(h0_features)  # 形状 (B, N, C)
        
        # 转换为 2D 特征图：累加点贡献
        h0_image = torch.sum(h0_mapped, dim=1)  # 形状 (B, C)
       
        h0_image = h0_image.unsqueeze(-1).unsqueeze(-1)  # 调整为 (B, C, 1, 1)
       
        h0_image = self.resize(h0_image)  # 调整为 (B, C, H, W)
        return h0_image
    
class H1Mapper(torch.nn.Module):
    def __init__(self, cnn_channels, target_resolution):
        """
        将 H1 映射到 CNN 特征图分辨率和通道数。
        
        参数:
            cnn_channels (int): CNN 的目标通道数。
            target_resolution (tuple): CNN 特征图的空间分辨率 (H, W)。
        """
        super(H1Mapper, self).__init__()
        self.channel_mapping = torch.nn.Conv2d(1, cnn_channels, kernel_size=1)  # 单通道到 C 通道
        self.resize = torch.nn.AdaptiveAvgPool2d(target_resolution)  # 调整空间分辨率

    def forward(self, h1_features):
        """
        前向传播，处理 H1 特征。
        
        参数:
            h1_features (torch.Tensor): 持久性图像，形状为 (B, H', W')。
        
        返回:
            torch.Tensor: 映射到 CNN 特征图形状的张量 (B, C, H, W)。
        """
        batch_size, height, width = h1_features.shape
        h1_features = h1_features.unsqueeze(1)  # 添加通道维度，变为 (B, 1, H', W')
        h1_mapped = self.channel_mapping(h1_features)  # (B, C, H', W')
        h1_resized = self.resize(h1_mapped)  # 调整到 (B, C, H, W)
        return h1_resized
'''   
    import torch

if __name__ == "__main__":
    # 模拟 H0 的输入，形状为 (B, 1000, 2)
    batch_size = 8
    num_h0_points = 1000
    h0_features = torch.rand(batch_size, num_h0_points, 2)  # 随机生成 H0 特征
    print(type(h0_features))
    # 模拟 H1 的输入，形状为 (B, 50, 30)
    h1_height = 50
    h1_width = 30
    h1_features = torch.rand(batch_size, h1_height, h1_width)  # 随机生成 H1 特征

    # 定义映射器
    cnn_channels = 16  # CNN 通道数
    target_resolution = (500, 500)  # 目标分辨率

    h0_mapper = H0Mapper(cnn_channels, target_resolution)
    h1_mapper = H1Mapper(cnn_channels, target_resolution)

    # 前向传播
    h0_mapped_features = h0_mapper(h0_features)
    h1_mapped_features = h1_mapper(h1_features)

    print("H0 Mapped Features Shape:", h0_mapped_features.device)  # 应该是 (B, C, H, W)
    print("H1 Mapped Features Shape:", h1_mapped_features.device)  # 应该是 (B, C, H, W)
'''