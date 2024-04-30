"""This module contains methods used to extract the estimated image and
camera coordinates from features."""

import numpy as np

def feats2points(left, right, lines):
    """Adapt image coordinates according to features."""
    points = {}
    top_line, bot_line = lines
    mid_x = int(np.mean([left, right]))
    top = int(
        (top_line[0][1] * (top_line[1][0] - mid_x) + top_line[1][1] * (mid_x - top_line[0][0])) / (top_line[1][0] - top_line[0][0])
    )
    bot = int(
        (bot_line[0][1] * (bot_line[1][0] - mid_x) + bot_line[1][1] * (mid_x - bot_line[0][0])) / (bot_line[1][0] - bot_line[0][0])
    )
    mid_y = int(np.mean([bot, top]))

    points["mid"] = [mid_x, mid_y]
    points["top"] = [mid_x, top]
    points["bot"] = [mid_x, bot]
    points["left"] = [left, mid_y]
    points["right"] = [right, mid_y]

    return points

def imgcoord2camcoord(depth, img_coord_glob, args):
    """Transform image coordinates to camera coordinates with camera intrinsics and depth image"""
    cam_coord = {}
    for k, v in img_coord_glob.items():
        y = depth[v[1], v[0]] / 10
        x = (v[0] - args.px) * y / args.fx
        z = (v[1] - args.py) * y / args.fy
        cam_coord[k] = (x, y, z)

    assert ((args.brick_width - 15 <= (cam_coord["right"][0]-cam_coord["left"][0]) <= args.brick_width + 15) or     # check width for normal stone
            (args.brick_depth - 10 <= (cam_coord["right"][0]-cam_coord["left"][0]) <= args.brick_depth + 10) or     # check width for corner stone
            (args.brick_height - 10 <= (cam_coord["bot"][-1]-cam_coord["bot"][-1]) <= args.brick_height + 10)), "Pose uncertain: invalid coordinate estimation"
    
    if args.brick_depth - 10 <= (cam_coord["right"][0]-cam_coord["left"][0]) <= args.brick_depth + 10:
        args.corner_stone = True
    else:
        args.corner_stone = False
    
    return cam_coord

