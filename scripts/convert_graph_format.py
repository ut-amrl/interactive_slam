"""
Author: Donmgmyeong Lee (domlee[at]utexas.edu)
Date:   Feb 11, 2024
Description: Get the poses and pointclouds from the map directory (result of interactive_slam)
"""
import os
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map_dir",
        type=str,
        required=True,
        help="Path to the map directory (result of interactive_slam)",
    )
    parser.add_argument(
        "--pcd",
        action="store_true",
        help="Get the .bin files of the pointclouds corresponding to the keyframes",
    )
    return parser.parse_args()


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


def pc_to_bin(pcd_file: str, bin_file: str):
    """
    Convert a .pcd file to a .bin file

    Args:
        pcd_file: Path to the .pcd file
        bin_file: Path to the .bin file
    """
    # load the pointcloud
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)

    # save the pointcloud as a .bin file
    points.astype(np.float32).tofile(bin_file)


def main():
    args = get_args()

    # Get the paths to the pose files and the timestamps
    keyframes_files = list(Path(args.map_dir).glob("[0-9]*/data"))
    print("Load keyframe poses from: ", args.map_dir)

    # Get the keyframe poses
    keyframe_poses = [
        (pose_file.parent, load_keyframe_pose(pose_file))
        for pose_file in keyframes_files
    ]
    sorted_keyframes = sorted(keyframe_poses, key=lambda x: x[1][0])

    # Save the keyframe poses
    pose_file = Path(args.map_dir) / "poses.txt"
    os.makedirs(pose_file.parent, exist_ok=True)
    with open(pose_file, "w") as f:
        for _, keyframe in sorted_keyframes:
            ts = keyframe[0]
            pose = keyframe[1:]
            f.write(f"{ts:.6f} " + " ".join(f"{p:.8f}" for p in pose) + "\n")
    print(f"Saved {pose_file}")

    # Save the pointclouds (.pcd -> .bin)
    if not args.pcd:
        return

    pc_out_dir = Path(args.map_dir) / "points"
    os.makedirs(pc_out_dir, exist_ok=True)
    frame = 0
    for keyframe_dir, pose in sorted_keyframes:
        pcd_file = str(keyframe_dir / "cloud.pcd")
        bin_file = str(pc_out_dir / f"{frame}.bin")
        pc_to_bin(pcd_file, bin_file)
        frame += 1
    print(f"saved {len(sorted_keyframes)} pointclouds to {pc_out_dir}")


if __name__ == "__main__":
    main()
