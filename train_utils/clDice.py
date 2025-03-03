from skimage.morphology import skeletonize
import numpy as np
import torch

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(prediction, label, ignore_index=None):
    """
    计算 clDice 指标，支持忽略指定标签。

    Args:
        v_p (np.ndarray or torch.Tensor): 预测图（2D 或 3D）。
        v_l (np.ndarray or torch.Tensor): 真实标签图（2D 或 3D）。
        ignore_index (int, optional): 要忽略的标签值，默认为 None。

    Returns:
        float: clDice 指标值。
    """
    # 如果有 ignore_index，则过滤对应的像素或体素
    pred = torch.argmax(prediction, dim=1)  # Shape: (1, 584, 565)
    v_p = (pred == 1).squeeze(0).cpu().numpy()
    v_l = (label == 1).squeeze(0).cpu().numpy()
    
    if ignore_index is not None:
        mask = label != ignore_index  # 仅保留非忽略区域
        mask = mask.cpu().numpy()
        v_p = v_p * mask  # 将预测图中的忽略区域置为 0
        v_l = v_l * mask  # 将标签图中的忽略区域置为 0
    
    # 2D 图像处理
    if len(v_p.shape) == 2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    # 3D 图像处理
    elif len(v_p.shape) == 3:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    else:
        raise ValueError("Input images must be either 2D or 3D.")

    # 计算 clDice 指标
    if tprec + tsens == 0:  # 防止除以零的情况
        return 0.0
    return 2 * tprec * tsens / (tprec + tsens)
