"""
Author: Donmgmyeong Lee (domlee[at]utexas.edu)
Date:   Feb 26, 2024
Description: Remove node from the graph by timestamp
"""
import os
import argparse
import tempfile
from pathlib import Path
import shutil
import time


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
    parser.add_argument(
        "--time_range",
        nargs=2,
        type=float,
        help="Time range to remove nodes",
        default=[0, 0],
    )

    return parser.parse_args()


def load_keyframe_pose(keyframe_pose_file: str) -> np.ndarray:
    """
    Load estimated pose from a keyframe file (.data) from interactive_slam

    Args:
        keyframe_pose_file: Path to the (.data) file containing the estimated pose.

    Returns:
        keyframe_pose: (9,) estimated pose (timestamp, id, x, y, z, qw, qx, qy, qz)
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
        keyframe_pose = np.zeros(9)
        keyframe_pose[0] = timestamp
        keyframe_pose[2:5] = keyframe_pose_matrix[:3, 3]
        keyframe_pose[5:] = r.as_quat()[[3, 0, 1, 2]]

        # id of the keyframe
        keyframe_pose[1] = int(lines[12].strip().split(" ")[1])

    return keyframe_pose


def main():
    args = get_args()

    # Get the paths to the pose files and the timestamps
    keyframes_files = list(Path(args.map_dir).glob("[0-9]*/data"))
    print("Load keyframe poses from: ", args.map_dir)
    print("time range: ", args.time_range)

    # Get the keyframe poses
    keyframe_poses = [
        (pose_file.parent, load_keyframe_pose(pose_file))
        for pose_file in keyframes_files
    ]
    sorted_keyframes = sorted(keyframe_poses, key=lambda x: x[1][1])

    existing_timestamps = {}
    removed_nodes = set()

    new_id = 0
    count = 0
    old_id_to_new_id = {}
    for keyframe_dir, pose in sorted_keyframes:
        timestamp = pose[0]
        old_id = int(pose[1])
        if args.time_range[0] <= timestamp <= args.time_range[1]:
            print("Removing node: ", old_id)
            removed_nodes.add(old_id)
            shutil.rmtree(keyframe_dir)
        elif timestamp in existing_timestamps:
            print("duplicate timestamp: ", timestamp)
            print("existing: ", existing_timestamps[timestamp])
            print("new: ", old_id)
            print("removing: ", keyframe_dir)
            shutil.rmtree(keyframe_dir)
            removed_nodes.add(old_id)
        else:
            existing_timestamps[timestamp] = old_id
            shutil.move(keyframe_dir, keyframe_dir.parent / f"{count:06d}_temp")
            old_id_to_new_id[old_id] = new_id
            count += 1
            new_id = 100 if new_id == 97 else new_id + 1

    for count in range(count):
        shutil.move(
            keyframe_dir.parent / f"{count:06d}_temp",
            keyframe_dir.parent / f"{count:06d}",
        )

    # Remove the nodes from the g2o file
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    g2o_file = Path(args.map_dir) / "graph.g2o"
    existing_nodes = set()

    with open(g2o_file, "r") as infile, open(temp_file.name, "w") as outfile:
        for line in infile:
            if line.startswith("VERTEX_SE3:QUAT"):
                vertex_id = int(line.split()[1])
                if vertex_id not in (removed_nodes | existing_nodes | {98}):
                    existing_nodes.add(vertex_id)
                    lines = line.split()
                    lines[1] = str(old_id_to_new_id[vertex_id])
                    outfile.write(" ".join(lines) + "\n")

            elif line.startswith("EDGE_SE3:QUAT"):
                vertex_ids = list(map(int, line.split()[1:3]))
                if (
                    vertex_ids[0] not in removed_nodes
                    and vertex_ids[1] not in removed_nodes
                ):
                    lines = line.split()
                    lines[1] = str(old_id_to_new_id[vertex_ids[0]])
                    lines[2] = str(old_id_to_new_id[vertex_ids[1]])
                    outfile.write(" ".join(lines) + "\n")
                else:
                    print("Removing edge: ", vertex_ids)
            else:
                print("Skipping line: ", line)
                outfile.write(line)

    # Replace the original g2o file with the new one
    os.replace(temp_file.name, g2o_file)


if __name__ == "__main__":
    main()
