from skimage.morphology import skeletonize
import numpy as np
import torch

def cl_score(v, s):
    """Computes the skeleton volume overlap.

    Args:
        v (bool): Image.
        s (bool): Skeleton.

    Returns:
        float: Computed skeleton volume intersection.
    """
    return np.sum(v * s) / np.sum(s)


def clDice(prediction, label, ignore_index=None):
    """
    Compute the clDice metric, supporting the exclusion of a specified label.

    Args:
        prediction (np.ndarray or torch.Tensor): Predicted image (2D or 3D).
        label (np.ndarray or torch.Tensor): Ground truth label image (2D or 3D).
        ignore_index (int, optional): Label value to ignore. Defaults to None.

    Returns:
        float: clDice metric value.
    """
    # If ignore_index is provided, filter out the corresponding pixels/voxels
    pred = torch.argmax(prediction, dim=1)  # Shape: (1, 584, 565)
    v_p = (pred == 1).squeeze(0).cpu().numpy()
    v_l = (label == 1).squeeze(0).cpu().numpy()
    
    if ignore_index is not None:
        mask = label != ignore_index  # Retain only non-ignored regions
        mask = mask.cpu().numpy()
        v_p = v_p * mask  # Set ignored regions in prediction to 0
        v_l = v_l * mask  # Set ignored regions in label to 0
    
    # Process 2D images
    if len(v_p.shape) == 2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    # Process 3D images
    elif len(v_p.shape) == 3:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    else:
        raise ValueError("Input images must be either 2D or 3D.")

    # Compute the clDice metric
    if tprec + tsens == 0:  # Prevent division by zero
        return 0.0
    return 2 * tprec * tsens / (tprec + tsens)
