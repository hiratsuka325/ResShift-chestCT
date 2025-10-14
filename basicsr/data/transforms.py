import cv2
import random
import torch
import numpy as np


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img

def calculate_zero_ratio(images):
    """Calculate the ratio of zero pixels in the image(s)."""
    if isinstance(images, list):
        total_zero_pixels = 0
        total_pixels = 0
        for img in images:
            if torch.is_tensor(img):
                total_zero_pixels += (img == 0).sum().item()  # ゼロ画素の数
                total_pixels += img.numel()  # 総画素数
            else:
                total_zero_pixels += np.sum(img == 0)  # NumPy配列の場合、ゼロ画素の数
                total_pixels += img.size  # NumPy配列の場合、総画素数
        return total_zero_pixels / total_pixels  # ゼロ画素の割合を返す


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img

def paired_random_crop_mask(img_gts, img_lqs, img_masks, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of LQ, GT, and mask images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images.
        img_lqs (list[ndarray] | ndarray): LQ images.
        img_masks (list[ndarray] | ndarray | list[Tensor] | Tensor): Mask images.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images, LQ images, and Mask images. 
        If only one image is returned, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]
    if not isinstance(img_masks, list):
        img_masks = [img_masks]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
        h_mask, w_mask = img_masks[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
        h_mask, w_mask = img_masks[0].shape[0:2]

    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x '
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    ## Perform 10 random crops and choose the one with the least number of zero pixels in the GT image
    min_zero_count = float('inf')
    min_zero_ratio = float('inf')
    best_crop_lq = None
    best_crop_gt = None
    best_crop_mask = None

    for _ in range(10000):
        # Randomly choose top and left coordinates for LQ patch
        top = random.randint(0, h_lq - lq_patch_size)
        left = random.randint(0, w_lq - lq_patch_size)

        # Crop LQ patch
        if input_type == 'Tensor':
            img_lqs_cropped = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
        else:
            img_lqs_cropped = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

        # Crop corresponding GT patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        if input_type == 'Tensor':
            img_gts_cropped = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
        else:
            img_gts_cropped = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]

        # Crop corresponding Mask patch
        if input_type == 'Tensor':
            img_masks_cropped = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_masks]
        else:
            img_masks_cropped = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_masks]

        # Calculate the total number of zero pixels in the GT image crop
        #zero_count_mask = calculate_zero_count(img_masks_cropped)

        # Check if the current crop has fewer zero pixels in the GT image
        #if zero_count_mask < min_zero_count:
        #    min_zero_count = zero_count_mask
        #    best_crop_lq = img_lqs_cropped
        #    best_crop_gt = img_gts_cropped
        #    best_crop_mask = img_masks_cropped
        
        # Calculate the ratio of zero pixels in the Mask image crop
        zero_ratio_mask = calculate_zero_ratio(img_masks_cropped)

        # Check if the zero ratio is less than 20%
        if zero_ratio_mask < 0.20:
            # Break the loop if zero ratio is less than 20%
            best_crop_lq = img_lqs_cropped
            best_crop_gt = img_gts_cropped
            best_crop_mask = img_masks_cropped
            break

        # Check if the current crop has fewer zero pixels in the Mask image
        if zero_ratio_mask < min_zero_ratio:
            min_zero_ratio = zero_ratio_mask
            best_crop_lq = img_lqs_cropped
            best_crop_gt = img_gts_cropped
            best_crop_mask = img_masks_cropped   

    # If only one image is returned, return the image directly (not as a list)
    if len(best_crop_lq) == 1:
        best_crop_lq = best_crop_lq[0]
    if len(best_crop_gt) == 1:
        best_crop_gt = best_crop_gt[0]
    if len(best_crop_mask) == 1:
        best_crop_mask = best_crop_mask[0]

    return best_crop_gt, best_crop_lq, best_crop_mask