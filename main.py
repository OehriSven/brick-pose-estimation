"""This module runs the main script"""

import os
import sys
import traceback
import argparse
from flask import Flask, request, jsonify, render_template

from pose_estimation.pose_estimation import pose_estimation


def get_args_parser():
    parser = argparse.ArgumentParser('Place quality system', add_help=False)

    # Temporary image directory
    parser.add_argument('--savedir', default="temp", type=str, help="Temporary save directory for uploaded images")

    # Brick parameters
    parser.add_argument('--brick-width', default=210.0, type=float, help="Physical width of brick")
    parser.add_argument('--brick-height', default=50.0, type=float, help="Physical height of brick")
    parser.add_argument('--brick-depth', default=100.0, type=float, help="Physical depth of brick")


    # ROI parameters
    parser.add_argument('--roi-center', default=(424,300), type=tuple, help="ROI center image coordinates (x, y)")
    parser.add_argument('--roi-winsize', default=(400, 100), type=tuple, help="Windows size of image ROI (width, height)")

    # Canny edge parameters
    parser.add_argument('--blur-kernel-canny', default=[((0,0), 1.5), ((5,5), 0), ((7,7), 0)], type=list, help="List of blurring kernel sizes for canny edge detector")
    parser.add_argument('--canny-thresh', default=[(30, 150), (50, 200), (210, 250)], type=list, help="List of thresholds for canny edge detector")
    parser.add_argument('--voting-thresh', default=3, type=int, help="Voting threshold for canny edge voting")

    # Threshold masking parameters
    parser.add_argument('--blur-kernel-thresh', default=9, type=int, help="Blurring kernel size for adaptive threshold detector")
    parser.add_argument('--thresh-blocksize', default=11, type=int, help="Blocksize of adaptive threshold window")
    parser.add_argument('--thresh-c', default=2.0, type=float, help="Subtraction constant of adaptive threshold window")

    # Hough transformation parameters
    parser.add_argument('--thresh-hough', default=220, type=int, help="Threshold of Hough line transformation")
    parser.add_argument('--thresh-cluster', default=40.0, type=float, help="Threshold of Non-Max-Suppression of Hough lines")

    # Edge extractor parameters
    parser.add_argument('--edge-horizontal-kernel', default=(5, 1), type=tuple, help="Kernel shape of horizontal edge extractor")
    parser.add_argument('--edge-vertical-kernel', default=(1, 5), type=tuple, help="Kernel shape of vertical edge extractor")
    parser.add_argument('--edge-thresh', default=1, type=int, help="Threshold of edge extractor")
    parser.add_argument('--edge-nsteps', default=3, type=int, help="Number of vertical/horizontal lines of edge extractor")
    parser.add_argument('--edge-stepsize', default=10, type=int, help="Distance between each line of edge extractor")

    # Camera intrinsics
    parser.add_argument('--fx', default=434.5079345703125, type=float, help="Focal length in x-direction fx")
    parser.add_argument('--fy', default=434.5079345703125, type=float, help="Focal length in y-direction fy")
    parser.add_argument('--px', default=427.6170654296875, type=float, help="Principal point x-coordinate px")
    parser.add_argument('--py', default=238.77597045898438, type=float, help="Principal point y-coordinate py")

    return parser


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('upload.html')  # Render the HTML form

@app.route('/process_images', methods=['GET', 'POST'])
def process_images():
    # Check if both color and depth images are present in the request
    if 'color_image' not in request.files or 'depth_image' not in request.files:
        return jsonify({'error': 'Color image or depth image missing'}), 400

    color_image = request.files['color_image']
    depth_image = request.files['depth_image']

    # Save images to disk before passing to pose estimation method
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    color_image.save(os.path.join(args.savedir, 'color.png'))
    depth_image.save(os.path.join(args.savedir, 'depth.png'))

    # Call the pose estimation method with the input images
    try:
        brick_pose = pose_estimation(os.path.join(args.savedir, 'color.png'), os.path.join(args.savedir, 'depth.png'), args)
    # On insufficient/invalid pose estimation print error message and return dictionary of None
    except AssertionError:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        print(traceback.format_exc())
        return {None: None}

    return brick_pose

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Place quality system', parents=[get_args_parser()])
    args = parser.parse_args()
    app.run(debug=True)  # Run the Flask app in debug mode
