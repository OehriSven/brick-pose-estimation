"""This module contains methods used to extract features from segmentation masks."""

import numpy as np
import cv2

from ..utils.utils import (
    cluster_lines,
    line_duplicator,
)

def hough_transformation(mask, top, bot, args):
    """Extract horizontal hough lines from binary thresh mask"""
    lines_xy = []

    if args.sam:
        thresh_hough = 120 # args.thresh_hough
        lines = None
        while lines is None or not any([item[0][0] > 50 for item in lines]):
            lines = cv2.HoughLines(mask, 1, np.pi / 180, thresh_hough, None, 0, 0)
            thresh_hough -= 10
    else:
        lines = cv2.HoughLines(mask, 1, np.pi / 180, args.thresh_hough, None, 0, 0)

    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        lines_xy.append((pt1, pt2))

    # Non-Max-Suppression of detected Hough lines
    clustered_lines = [line[0] for line in cluster_lines(lines_xy, args.thresh_cluster)]

    if args.sam:
        for line in clustered_lines:
            if line[1][1] > 50:
                clustered_lines = [line]

    # Assert at least one, but at most to resulting hough lines for top and bottom edge
    assert 1 <= len(clustered_lines) <= 2, "Hough transformation not successful"

    # Translate top line to bottom edge and vice versa if only one edge is detected
    if len(clustered_lines) == 1:
        clustered_lines = line_duplicator(clustered_lines, top, bot, args)
    # Arrange order of lines so that top edge line is first and bottom edge line is second
    else:
        slope = (clustered_lines[0][1][1] - clustered_lines[0][0][1]) / (clustered_lines[0][1][0] - clustered_lines[0][0][0])
        if clustered_lines[0][0][1] - clustered_lines[0][0][0] * slope > 50:  # If bottom line first, switch
            clustered_lines[0], clustered_lines[1] = clustered_lines[1], clustered_lines[0]
    
    return clustered_lines

def horizontal_edge_extractor(mask, args):
    """Extract horizontal edges from binary mask
    starting the search from center of ROI"""

    assert args.edge_nsteps%2 == 1, "Use odd number of steps"
    ksize = args.edge_horizontal_kernel

    x, y = mask.shape[1] // 2, mask.shape[0] // 2
    top, bot = [], []

    for w in range(x-args.edge_stepsize*(args.edge_nsteps//2), x+args.edge_stepsize*(args.edge_nsteps//2), args.edge_stepsize):
        while y >= 0:
            if np.sum(mask[y-ksize[0]//2:y+ksize[0]//2+1,w-ksize[1]//2:w+ksize[1]//2+1]) / 255 >= args.edge_thresh:
                top.append((w, y))
                break
            y -= 1

    x, y = mask.shape[1] // 2, mask.shape[0] // 2
    for w in range(x-args.edge_stepsize*(args.edge_nsteps//2), x+args.edge_stepsize*(args.edge_nsteps//2), args.edge_stepsize):
        while y <= mask.shape[1]:
            if np.sum(mask[y-ksize[0]//2:y+ksize[0]//2+1,w-ksize[1]//2:w+ksize[1]//2+1]) / 255 >= args.edge_thresh:
                bot.append((w, y))
                break
            y += 1

    top = np.median([p[1] for p in top]).astype(int) if top else -1
    bot = np.median([p[1] for p in bot]).astype(int) if bot else -1
    
    return (top, bot)

def vertical_edge_extractor(mask, args):
    """Extract vertical edges from binary mask
    starting the search from center of ROI"""

    assert args.edge_nsteps%2 == 1, "Use odd number of steps"
    ksize = args.edge_vertical_kernel

    x, y = mask.shape[1] // 2, mask.shape[0] // 2
    left, right = [], []

    for h in range(y-args.edge_stepsize*(args.edge_nsteps//2), y+args.edge_stepsize*(args.edge_nsteps//2), args.edge_stepsize):
        while x >= 0:
            if np.sum(mask[h-ksize[0]//2:h+ksize[0]//2+1,x-ksize[1]//2:x+ksize[1]//2+1]) / 255 >= args.edge_thresh:
                left.append((x, h))
                break
            x -= 1

    x, y = mask.shape[1] // 2, mask.shape[0] // 2
    for h in range(y-args.edge_stepsize*(args.edge_nsteps//2), y+args.edge_stepsize*(args.edge_nsteps//2), args.edge_stepsize):
        while x <= mask.shape[1]:
            if np.sum(mask[h-ksize[0]//2:h+ksize[0]//2+1,x-ksize[1]//2:x+ksize[1]//2+1]) / 255 >= args.edge_thresh:
                right.append((x, h))
                break
            x += 1

    left = np.median([p[0] for p in left]).astype(int) if left else -1
    right = np.median([p[0] for p in right]).astype(int) if right else -1

    assert left > 0 and right > 0, "No sufficient information for vertical edges"
    
    return (left, right)