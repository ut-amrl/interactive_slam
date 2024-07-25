"""
Author: Donmgmyeong Lee (domlee[at]utexas.edu)
Date:   Feb 11, 2024
Description: Get the poses from the map directory (result of interactive_slam)
"""

import os
import argparse
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation as R


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map_dir",
        type=str,
        required=True,
        help="Path to the map directory (result of interactive_slam)",
    )
    args = parser.parse_args()

    args.map_dir = pathlib.Path(args.map_dir)

    return args


def load_keyframe_pose(keyframe_pose_file: str) -> np.ndarray:
    """
    Load estimated pose from a keyframe file (.data) from interactive_slam

    Args:
        keyframe_pose_file: Path to the (.data) file containing the estimated pose.

    Returns:
        keyframe_pose: (8,) estimated pose (timestamp, x, y, z, qw, qx, qy, qz)
    """
    with open(keyframe_pose_file, "r") as f:
        lines = f.readlines()

        # timestamp
        timestamp_line = lines[0].strip().split(" ")
        timestamp = float(timestamp_line[1])

        # estimated pose (SE3)
        pose_lines = lines[2:6]
        keyframe_pose_matrix = np.array(
            [list(map(float, line.split())) for line in pose_lines]
        )
        r = R.from_matrix(keyframe_pose_matrix[:3, :3])

        # estimated pose (timestamp, x, y, z, qw, qx, qy, qz)
        keyframe_pose = np.zeros(8)
        keyframe_pose[0] = timestamp
        keyframe_pose[1:4] = keyframe_pose_matrix[:3, 3]
        keyframe_pose[4:] = r.as_quat()[[3, 0, 1, 2]]
    return keyframe_pose


def main():
    args = get_args()

    # Get the paths to the pose files and the timestamps
    keyframes_files = list(args.map_dir.glob("[0-9]*/data"))
    print(f"Load {len(keyframes_files)} keyframe poses from: ", args.map_dir)

    # Get the keyframe poses
    keyframe_poses = np.zeros((len(keyframes_files), 8))
    for i, pose_file in enumerate(keyframes_files):
        keyframe_poses[i] = load_keyframe_pose(pose_file)
    sorted_keyframes = keyframe_poses[keyframe_poses[:, 0].argsort()]

    # remove duplicates rows
    _, idx = np.unique(sorted_keyframes[:, 0], axis=0, return_index=True)
    sorted_keyframes = sorted_keyframes[idx]

    # Save the keyframe poses
    pose_file = args.map_dir.parent / f"{args.map_dir.name}.txt"
    with open(pose_file, "w") as f:
        for keyframe in sorted_keyframes:
            ts = keyframe[0]
            pose = keyframe[1:]
            f.write(f"{ts:.6f} " + " ".join(f"{p:.8f}" for p in pose) + "\n")
    print(f"Saved {pose_file}")


if __name__ == "__main__":
    main()
