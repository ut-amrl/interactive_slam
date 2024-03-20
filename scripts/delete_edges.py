import os
import argparse
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation as R


def get_args():
    parser = argparse.ArgumentParser(
        description="Delete edges with large rotatation and small translation."
    )
    parser.add_argument("-i", "--input_file", type=str, required=True)
    parser.add_argument(
        "-d",
        "--distance",
        type=float,
        default=0,
        help="maximum distance threshold (m) between consecutive edges.",
    )
    parser.add_argument(
        "-a",
        "--angle",
        type=float,
        default=0,
        help="minimum angle threshold (rad) between consecutive edges.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete all edges, regardless of distance and angle.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)

    vertices = {}
    anchors = set()
    delete_count = 0
    with open(args.input_file, "r") as infile, open(temp_file.name, "w") as outfile:
        for line in infile:
            parts = line.strip().split()
            if parts[0] == "VERTEX_SE3:QUAT":
                node_id = int(parts[1])
                x, y, z = map(float, parts[2:5])
                qx, qy, qz, qw = map(float, parts[5:9])
                vertices[node_id] = (np.array([x, y, z]), np.array([qx, qy, qz, qw]))

            if parts[0] == "FIX":
                anchors.add(int(parts[1]))

            if parts[0] == "EDGE_SE3:QUAT":
                id1, id2 = map(int, parts[1:3])
                # is_consecutive_edge = (
                # (abs(id1 - id2) == 1)
                # or (id1 == 0 and id2 == max_id)
                # or (id2 == 0 and id1 == max_id)
                # )
                # if not is_anchor and is_consecutive_edge:
                # Compute the distance between the two nodes
                p1, q1 = vertices[id1]
                p2, q2 = vertices[id2]
                distance = np.linalg.norm(p2 - p1)
                angle = (R.from_quat(q1).inv() * R.from_quat(q2)).magnitude()

                if args.all or (distance < args.distance and angle > args.angle):
                    delete_count += 1
                    continue

            outfile.write(line)

    os.replace(temp_file.name, args.input_file)
    print(f"Deleted {delete_count} edges.")


if __name__ == "__main__":
    main()
