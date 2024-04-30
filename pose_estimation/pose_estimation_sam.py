"""This module runs the pose estimation script"""

import cv2
import numpy as np

from .utils.utils import (
    roi,
    roi2glob,
    draw_feats,
    angle_from_points,
    angle_from_lines,
    output_parser
)
from .segmentation.segment import segment_sam, segment_canny
from .features.feature_extractor import (
    horizontal_edge_extractor,
    vertical_edge_extractor,
    hough_transformation,
)
from .features.points import feats2points, imgcoord2camcoord


def pose_estimation_sam(color_png, depth_png, args):
    color, depth = cv2.imread(color_png), cv2.imread(depth_png, -1)

    assert color is not None, "Color image file could not be read, check with os.path.exists()"
    assert color is not None, "Depth image file could not be read, check with os.path.exists()"
    assert color.shape[:2] == depth.shape[:2], "Different shape of color and depth image"
    args.img_size = (color.shape[:2])

    # ROI definition
    color_roi = roi(color, args)

    # Brick segmentation in ROI window
    mask_sam = segment_sam(color_roi, args)
    mask_canny = segment_canny(color_roi, args)

    # Feature extraction from segmented brick in ROI window
    top, bot = horizontal_edge_extractor(mask_canny, args)
    left, right = vertical_edge_extractor(mask_sam, args)
    hough_lines = hough_transformation(mask_sam, top, bot, args)

    # True point adaption
    img_coord_roi = feats2points(left, right, hough_lines)

    # Draw features in output image
    draw_roi = draw_feats(color_roi, img_coord_roi, hough_lines)

    # Re-transformation of ROI coordinates to global image coordinates
    img_coord_glob = roi2glob(img_coord_roi, args)

    # Camera coordinates from image coordinates, camera intrinsics and depth image
    cam_coord = imgcoord2camcoord(depth, img_coord_glob, args)

    # Angles from camera coordinates and hough lines
    roll = angle_from_points(cam_coord, angle="roll")
    pitch = angle_from_lines(hough_lines)
    yaw = angle_from_points(cam_coord, angle="yaw")

    # JSON pose
    output = output_parser(cam_coord, roll, pitch, yaw, draw_roi, args)

    return output