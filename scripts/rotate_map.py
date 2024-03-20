import os
import argparse
import tempfile
from pathlib import Path
from tqdm import tqdm

import numpy as np
from scipy.spatial.transform import Rotation as R


def get_args():
    parser = argparse.ArgumentParser(description="Rotate a map")
    parser.add_argument("--map_dir", type=str, help="Map to rotate (.g2o file)")
    parser.add_argument("--degree", type=float, help="Rotation in degrees")
    return parser.parse_args()


def quaternion_multiply(rotation, quaternion):
    x0, y0, z0, w0 = rotation
    x, y, z, w = quaternion

    return np.array(
        [
            w0 * x + x0 * w + y0 * z - z0 * y,
            w0 * y - x0 * z + y0 * w + z0 * x,
            w0 * z + x0 * y - y0 * x + z0 * w,
            w0 * w - x0 * x - y0 * y - z0 * z,
        ],
        dtype=np.float64,
    )


def main():
    args = get_args()

    rot = R.from_euler("z", args.degree, degrees=True)
    rot_matrix = rot.as_matrix()
    rot_quat = rot.as_quat()  # x, y, z, w

    transformation = np.eye(4)
    transformation[:3, :3] = rot_matrix

    # Data
    kf_files = list(Path(args.map_dir).glob("[0-9]*/data"))
    for kf_file in tqdm(kf_files):
        temp_kf_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        with open(kf_file, "r") as infile, open(temp_kf_file.name, "w") as outfile:
            lines = infile.readlines()
            pose = np.array(
                [list(map(float, line.strip().split())) for line in lines[2:6]]
            )
            rotated_pose = np.dot(transformation, pose)
            lines[2:6] = ["\t" + " ".join(map(str, row)) + "\n" for row in rotated_pose]

            temp_kf_file.writelines(lines)
            temp_kf_file.flush()

        os.replace(temp_kf_file.name, kf_file)

    # g2o format:
    map_file = os.path.join(args.map_dir, "graph.g2o")
    temp_map_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    with open(map_file, "r") as infile, open(temp_map_file.name, "w") as outfile:
        for line in infile:
            parts = line.strip().split()
            if parts[0] == "VERTEX_SE3:QUAT":
                x, y, z, qx, qy, qz, qw = map(float, parts[2:9])

                position = np.array([x, y, z])
                rotated_position = np.dot(rot_matrix, position)
                parts[2:5] = map(str, rotated_position)

                quaternion = np.array([qx, qy, qz, qw])
                rotated_quaternion = quaternion_multiply(rot_quat, quaternion)
                parts[5:] = map(str, rotated_quaternion)

            outfile.write(" ".join(parts) + "\n")

    os.replace(temp_map_file.name, map_file)


if __name__ == "__main__":
    main()
