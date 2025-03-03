import cv2
import torch
import numpy as np

def apply_gaussian_filter(logits, kernel_size=5, sigma=1):
    """
    对 U-Net 输出的 logits 图应用高斯滤波。
    logits: [N, C, H, W]，torch.Tensor
    kernel_size: 高斯滤波核大小
    sigma: 高斯滤波标准差
    """
    logits_np = logits.cpu().detach().numpy()  # 转为 numpy
    filtered_logits = []
    for i in range(logits_np.shape[0]):  # 对每个 batch 进行滤波
        filtered_batch = []
        for j in range(logits_np.shape[1]):  # 对每个通道进行滤波
            filtered = cv2.GaussianBlur(logits_np[i, j], (kernel_size, kernel_size), sigma)
            filtered_batch.append(filtered)
        filtered_logits.append(np.stack(filtered_batch, axis=0))
    return torch.tensor(np.stack(filtered_logits, axis=0), dtype=torch.float32).to(logits.device)

def apply_median_filter(logits, kernel_size=3):
    logits_np = logits.cpu().detach().numpy()
    filtered_logits = []
    for i in range(logits_np.shape[0]):
        filtered_batch = []
        for j in range(logits_np.shape[1]):
            filtered = cv2.medianBlur(logits_np[i, j].astype(np.float32), kernel_size)
            filtered_batch.append(filtered)
        filtered_logits.append(np.stack(filtered_batch, axis=0))
    return torch.tensor(np.stack(filtered_logits, axis=0), dtype=torch.float32).to(logits.device)
