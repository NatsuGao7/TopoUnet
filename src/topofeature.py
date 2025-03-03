import numpy as np
from PIL import Image
from ripser import Rips
from persim import PersistenceImager
import torch
import warnings
warnings.filterwarnings("ignore", message="The input matrix is square, but the distance_matrix flag is off.")
import torch.nn.functional as F
import matplotlib.pyplot as plt


def extract_topological_features_one(target_batch, output_size=(50, 30)):
    """
    从一个批次的目标图像中提取拓扑特征，并统一调整到相同尺寸。
    
    参数:
        target_batch (torch.Tensor): 输入的目标图像，形状为 (B, H, W)。
        output_size (tuple): 统一的输出尺寸，形如 (H', W')。
    
    返回:
        torch.Tensor: 形状为 (B, H', W') 的拓扑特征张量。
    """
    if target_batch.ndim != 3:
        raise ValueError("Input target batch must have 3 dimensions: (B, H, W).")

    batch_size = target_batch.shape[0]
    persistence_images = []

    rips = Rips(coeff=2)  # 初始化 Rips 对象
    pimgr = PersistenceImager(pixel_size=40)  # 初始化持久性图像生成器

    for i in range(batch_size):
        # 提取单张图片并转换为 NumPy 数组
        target_np = target_batch[i].cpu().numpy() if isinstance(target_batch, torch.Tensor) else np.array(target_batch[i])
        
        # 替换 255 为 0
        target_np[target_np == 255] = 0
        target_np[target_np == 1] = 255
        
        # 计算持久性条形码
        diagrams = rips.fit_transform(target_np)
        
        # 过滤无穷大的条目
        diagrams_array = np.array(diagrams, dtype=object)
        valid_diagrams = [d[~np.isinf(d[:, 1])] for d in diagrams_array]
        
        # 使用 H1 的拓扑信息
        H1_diagram = valid_diagrams[1] if len(valid_diagrams) > 1 else np.array([])
        if H1_diagram.size == 0:
            raise ValueError(f"No valid H1 features for image {i} in the batch.")
        
        # 生成持久性图像
        pimgr.fit(H1_diagram)
        persistence_image = pimgr.transform(H1_diagram)
        
        # 转为 Torch 张量
        persistence_image_tensor = torch.tensor(persistence_image, dtype=torch.float32)
        # 插值到目标尺寸
        persistence_image_resized = F.interpolate(
            persistence_image_tensor.unsqueeze(0).unsqueeze(0),  # 添加批次和通道维度
            size=output_size,  # 调整尺寸
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)  # 去掉批次和通道维度
        
        persistence_images.append(persistence_image_resized)

    # 堆叠为一个张量 (B, H', W')
    return torch.stack(persistence_images, dim=0)


def pad_persistence_points(diagram, max_points):
    """
    填充单个持久性图，使其具有固定数量的点。
    """
    # 计算缺少的点的数量
    num_missing_points = max_points - diagram.shape[0]
    
    # 如果缺少点，就填充为 (0, 0)，也可以使用其他值
    if num_missing_points > 0:
        padding = np.zeros((num_missing_points, 2), dtype=diagram.dtype)  # 填充的点
        padded_diagram = np.vstack([diagram, padding])
    else:
        padded_diagram = diagram[:max_points]  # 截断到 max_points 个点

    return padded_diagram

def extract_topological_features_zero(target_batch, max_points=1000):
    """
    提取0维拓扑特征，并将每个持久性图填充到相同大小。
    """
    persistence_points = []
    if target_batch.ndim != 3:
        raise ValueError("Input target batch must have 3 dimensions: (B, H, W).")
    batch_size = target_batch.shape[0]
    rips = Rips(coeff=2)
    for i in range(batch_size):
        # 提取单张图片并转换为 NumPy 数组
        target_np = target_batch[i].cpu().numpy() if isinstance(target_batch, torch.Tensor) else np.array(target_batch[i])
        
        # 替换 255 为 0
        target_np[target_np == 255] = 0
        target_np[target_np == 1] = 255

        diagrams = rips.fit_transform(target_np)
        # 过滤无穷大
        valid_diagrams = [d[~np.isinf(d[:, 1])] for d in diagrams]
        H0_diagram = valid_diagrams[0]

        # 填充至固定大小
        H0_feature = pad_persistence_points(H0_diagram, max_points=max_points)

        # 转换为 PyTorch tensor 并添加到列表中
        persistence_points.append(torch.tensor(H0_feature, dtype=torch.float32))

    # 将所有持久性特征堆叠为一个 batch tensor
    return torch.stack(persistence_points, dim=0)