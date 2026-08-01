"""
Convert iSign/.pose-format keypoint files into the exact layout
app/training/isltranslate.py's ISLTranslateKeypointDataset expects:

    data/processed/isltranslate/
    ├── ISLTranslate.csv
    ├── pose/<uid>.npy   (frames, 132)
    └── face/<uid>.npy   (frames, 1434)

Use this if the dataset already ships pre-extracted MediaPipe Holistic
keypoints in .pose-format (the iSign HuggingFace release does this — see
https://huggingface.co/datasets/Exploration-Lab/iSign). If you only have
raw video, use extract_keypoints.py instead.

Requires: pip install pose-format numpy pandas

Usage:
    python scripts/convert_isign_pose.py \\
        --pose_dir path/to/iSign-poses \\
        --labels_csv path/to/iSign_v1.1.csv \\
        --out_dir data/processed/isltranslate

IMPORTANT — VERIFY BEFORE TRUSTING THIS SCRIPT'S OUTPUT:
The pose-format library exports MediaPipe Holistic keypoints as named
components (typically POSE_LANDMARKS, FACE_LANDMARKS, LEFT_HAND_LANDMARKS,
RIGHT_HAND_LANDMARKS), but the exact component names/point counts in the
real dataset haven't been verified against an actual file — this sandbox
has no access to the gated/228GB dataset to inspect one. Before running on
your full dataset:
  1. Run: python scripts/convert_isign_pose.py --inspect_only <one_file.pose>
  2. Confirm the printed component names/shapes match POSE_COMPONENT /
     FACE_COMPONENT below.
  3. Adjust the constants if they don't match.
Skipping this check risks silently feeding wrong-shaped arrays into
training — fail loud on file 1, not after converting 5000 files.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

# Adjust these after verifying against --inspect_only output.
POSE_COMPONENT = "POSE_LANDMARKS"
FACE_COMPONENT = "FACE_LANDMARKS"
EXPECTED_POSE_DIM = 132   # 33 landmarks * 4 (x,y,z,confidence)
EXPECTED_FACE_DIM = 1434  # 478 landmarks * 3 (x,y,z)


def inspect_pose_file(pose_path: str):
    from pose_format import Pose

    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    print(f"File: {pose_path}")
    print(f"Total frames: {pose.body.data.shape[0]}")
    for component in pose.header.components:
        print(f"  Component: {component.name}, points: {len(component.points)}")


def extract_component(pose, component_name: str) -> np.ndarray:
    component_idx = next(
        i for i, c in enumerate(pose.header.components) if c.name == component_name
    )
    start = sum(len(c.points) for c in pose.header.components[:component_idx])
    n_points = len(pose.header.components[component_idx].points)

    data = pose.body.data[:, 0, start:start + n_points, :]  # assume single person
    frames = data.shape[0]
    return data.reshape(frames, -1).astype(np.float32)


def convert_file(pose_path: str, uid: str, pose_dir: str, face_dir: str):
    from pose_format import Pose

    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    pose_arr = extract_component(pose, POSE_COMPONENT)
    face_arr = extract_component(pose, FACE_COMPONENT)

    if pose_arr.shape[1] != EXPECTED_POSE_DIM:
        print(f"  WARNING {uid}: pose dim {pose_arr.shape[1]} != expected "
              f"{EXPECTED_POSE_DIM} — verify POSE_COMPONENT before trusting this clip")
    if face_arr.shape[1] != EXPECTED_FACE_DIM:
        print(f"  WARNING {uid}: face dim {face_arr.shape[1]} != expected "
              f"{EXPECTED_FACE_DIM} — verify FACE_COMPONENT before trusting this clip")

    np.save(os.path.join(pose_dir, f"{uid}.npy"), pose_arr)
    np.save(os.path.join(face_dir, f"{uid}.npy"), face_arr)


def resolve_uid_column(columns: list[str]) -> str:
    lower = {c.lower(): c for c in columns}
    for candidate in ("uid", "video_uid", "id"):
        if candidate in lower:
            return lower[candidate]
    raise ValueError(f"No uid-like column found in CSV. Columns: {columns}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_dir", help="Folder of .pose files")
    parser.add_argument("--labels_csv", help="iSign metadata CSV")
    parser.add_argument("--out_dir", help="e.g. data/processed/isltranslate")
    parser.add_argument("--inspect_only", metavar="POSE_FILE",
                         help="Just inspect one file's component layout and exit")
    args = parser.parse_args()

    if args.inspect_only:
        inspect_pose_file(args.inspect_only)
        return

    if not (args.pose_dir and args.labels_csv and args.out_dir):
        raise SystemExit("--pose_dir, --labels_csv, and --out_dir are all required (or use --inspect_only alone)")

    pose_dir_out = os.path.join(args.out_dir, "pose")
    face_dir_out = os.path.join(args.out_dir, "face")
    os.makedirs(pose_dir_out, exist_ok=True)
    os.makedirs(face_dir_out, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    uid_col = resolve_uid_column(list(df.columns))
    valid_uids = set(df[uid_col].astype(str))

    pose_files = sorted(glob.glob(os.path.join(args.pose_dir, "*.pose")))
    if not pose_files:
        raise SystemExit(f"No .pose files found in {args.pose_dir}")

    converted = 0
    for pose_path in pose_files:
        uid = os.path.splitext(os.path.basename(pose_path))[0]
        if uid not in valid_uids:
            print(f"  Skipping {uid}: not found in {args.labels_csv}")
            continue
        convert_file(pose_path, uid, pose_dir_out, face_dir_out)
        converted += 1
        print(f"[{converted}] {uid} converted")

    print(f"Done. Converted {converted}/{len(pose_files)} files to {args.out_dir}. "
          f"Now copy {args.labels_csv} into {args.out_dir}/ISLTranslate.csv if not already there.")


if __name__ == "__main__":
    main()
