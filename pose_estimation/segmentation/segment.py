"""This module contains methods used to segment the brick of interest."""

import numpy as np
import cv2


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