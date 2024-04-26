"""This module contains helper methods for calculation operations."""

import json
import numpy as np

def roi(img, args):
    """Return the region of interest window
    based on roi_center (x, y) and roi_winsize(width, height)"""

    return img[args.roi_center[1]-args.roi_winsize[1]//2:args.roi_center[1]+args.roi_winsize[1]//2,
               args.roi_center[0]-args.roi_winsize[0]//2:args.roi_center[0]+args.roi_winsize[0]//2]

def distance(line1, line2):
    """Calculate the distance between the closest points of two lines."""

    dists = []
    for point1 in line1:
        for point2 in line2:
            dist = np.linalg.norm(np.array(point1) - np.array(point2))
            dists.append(dist)

    return min(dists)

def cluster_lines(lines, thresh=40.0):
    """Clustering multiple lines (Non-Maximum-Suppression)."""

    clusters = []
    for line in lines:
        # Find an existing cluster to add the line to, if any
        added = False
        for cluster in clusters:
            if any(distance(line, existing_line) < thresh for existing_line in cluster):
                cluster.append(line)
                added = True
                break
        # If no existing cluster fits, create a new cluster
        if not added:
            clusters.append([line])

    return clusters

def line_duplicator(line, top, bot, args):
    """Translate single Hough line to edge with missing Hough line based on detected edge point."""

    slope = (line[0][1][1] - line[0][0][1]) / (line[0][1][0] - line[0][0][0])
    if line[0][0][1] - line[0][0][0] * slope > 50:  # If bottom line successfully derived, translate to top line
        assert top > 0 , "No sufficient information for Hough line transformation"
        line.insert(0, ((args.roi_winsize[0]//2-1000, int(top-1000*slope)), (args.roi_winsize[0]//2+1000, int(top+1000*slope))))
    else:                                           # If top line successfully derived, translate to bottom line
        line.append(((args.roi_winsize[0]//2-1000, int(bot-1000*slope)), (args.roi_winsize[0]//2+1000, int(bot+1000*slope))))
        assert bot > 0 , "No sufficient information for Hough line transformation"

    return line

def angle_from_lines(lines):
    """Calculate angle from Hough line."""

    slope = []
    for line in lines:
        slope.append((line[1][1] - line[0][1]) / (line[1][0] - line[0][0]))
    
    angle = np.rad2deg(np.arctan(np.mean(slope)))

    return angle

def angle_from_points(cam_coord, angle="roll"):
    """Calculate angle from depth coordinates."""

    if angle == "roll":
        slope = (cam_coord["bot"][1]-cam_coord["top"][1]) / (cam_coord["bot"][2]-cam_coord["top"][2])
    elif angle == "yaw":
        slope = (cam_coord["right"][1]-cam_coord["left"][1]) / (cam_coord["right"][0]-cam_coord["left"][0])
    angle = np.rad2deg(np.arctan(slope))

    return angle

def roi2glob(img_coord_roi, args):
    "Translate image coordinates from ROI coordinates to global image coordinates."
    
    img_coord_glob = {}
    for k, v in img_coord_roi.items():
        img_coord_glob[k] = [args.roi_center[0] - args.roi_winsize[0]//2 + v[0], args.roi_center[1] - args.roi_winsize[1]//2 + v[1]]

    return img_coord_glob

def json_parser(cam_coord, roll, pitch, yaw, args):
    "Parse pose into JSON format including corner stone transformation."

    pose = {}
    pose["x"] = np.round(cam_coord["mid"][0], 1)
    
    if args.corner_stone:
        pose["y"] = np.round(cam_coord["mid"][1], 1) + args.brick_width/2
    else:
        pose["y"] = np.round(cam_coord["mid"][1], 1) + args.brick_depth/2
    pose["z"] = np.round(cam_coord["mid"][2], 1)
    
    if args.corner_stone:
        pose["roll"] = np.round(pitch, 1)
        pose["pitch"] = np.round(roll, 1)
    else:
        pose["roll"] = np.round(roll, 1)
        pose["pitch"] = np.round(pitch, 1)

    pose["yaw"] = np.round(yaw, 1)

    return json.dumps(pose, indent = 4)
