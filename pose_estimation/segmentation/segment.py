"""This module contains methods used to segment the brick of interest."""

import os
import numpy as np
import cv2

from segment_anything import SamPredictor, sam_model_registry


def segment_thresh(img, args):
    """This method segments the brick of interest based on adaptive thresholding."""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,args.blur_kernel_thresh)

    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, args.thresh_blocksize, args.thresh_c)

    return mask


def segment_canny(img, args):
    """This method segments the brick of interest based on canny edges.
        The edges are created based on a voting of multiple canny masks."""

    # Assert voting threshold is smaller than number of masks
    assert args.voting_thresh <= (len(args.blur_kernel_canny) * len(args.canny_thresh))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, dtype=np.float64)

    for blur_kernel in args.blur_kernel_canny:
        blur = cv2.GaussianBlur(gray, blur_kernel[0], blur_kernel[1])
        for canny_thresh in args.canny_thresh:
            mask += cv2.Canny(blur, canny_thresh[0], canny_thresh[1]) / 255
    
    mask = mask > args.voting_thresh
    mask.dtype = "uint8"
    mask *= 255

    return mask


def segment_sam(img, args):
    """This method segments the brick of interest based on Meta's Segment
        Anything Model (SAM)."""

    # Assert SAM backbones exist in model_ckpts/
    assert os.path.exists(f"model_ckpts/sam_{args.sam_model}.pth")

    sam = sam_model_registry[args.sam_model](checkpoint=f"model_ckpts/sam_{args.sam_model}.pth")
    predictor = SamPredictor(sam)
    predictor.set_image(img)

    sam_masks, scores, _ = predictor.predict(
        point_coords=np.array([[args.roi_winsize[0] // 2, args.roi_winsize[1] // 2]]),
        point_labels=np.array([1]),
        multimask_output=True,
        )
    
    max_score_idx = np.argmax(scores)
    sam_mask = sam_masks[max_score_idx] * 255
    sam_mask = sam_mask.astype(np.uint8)

    blur = cv2.blur(sam_mask, (10,10))

    mask = cv2.Canny(blur, 50, 200, None, 3)

    return mask