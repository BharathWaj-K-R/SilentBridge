"""
Extract pose + face keypoints from raw ISLTranslate video clips using
MediaPipe Holistic, saving them in the exact layout
app/training/isltranslate.py's ISLTranslateKeypointDataset expects:

    data/processed/isltranslate/
    ├── ISLTranslate.csv   (you already have this from the dataset)
    ├── pose/<uid>.npy     (frames, 132)
    └── face/<uid>.npy     (frames, 1434)

Use this if you have RAW VIDEO clips. If the dataset already ships
pre-extracted MediaPipe features (some ISLTranslate/iSign releases do, in
.pose-format), use convert_isign_pose.py instead — that reads .pose files
directly rather than re-running MediaPipe on video.

Requires: pip install mediapipe opencv-python pandas

Usage:
    python scripts/extract_keypoints.py \\
        --videos_dir path/to/raw_videos \\
        --labels_csv path/to/ISLTranslate.csv \\
        --out_dir data/processed/isltranslate

Expects labels_csv to have a uid column (or video_uid/id) matching video
filenames as <uid>.mp4, plus a text/translation/english column — same
column-name flexibility as ISLTranslateKeypointDataset._read_examples().
"""
import argparse
import os

import numpy as np
import pandas as pd


def extract_clip_keypoints(video_path: str, holistic) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    pose_frames, face_frames = [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        if results.pose_landmarks:
            pose_frames.append(
                np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
                    dtype=np.float32,
                ).flatten()
            )
        else:
            pose_frames.append(np.zeros(33 * 4, dtype=np.float32))

        if results.face_landmarks:
            face_frames.append(
                np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark],
                    dtype=np.float32,
                ).flatten()
            )
        else:
            face_frames.append(np.zeros(478 * 3, dtype=np.float32))

    cap.release()
    if not pose_frames:
        raise ValueError(f"No frames read from {video_path} — check the file isn't corrupt")
    return np.stack(pose_frames), np.stack(face_frames)


def resolve_uid_column(columns: list[str]) -> str:
    lower = {c.lower(): c for c in columns}
    for candidate in ("uid", "video_uid", "id"):
        if candidate in lower:
            return lower[candidate]
    raise ValueError(f"No uid-like column found in CSV. Columns: {columns}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos_dir", required=True, help="Folder of <uid>.mp4 files")
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--out_dir", required=True,
                         help="e.g. data/processed/isltranslate — pose/ and face/ subfolders created here")
    args = parser.parse_args()

    import mediapipe as mp

    pose_dir = os.path.join(args.out_dir, "pose")
    face_dir = os.path.join(args.out_dir, "face")
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(face_dir, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    uid_col = resolve_uid_column(list(df.columns))
    uids = df[uid_col].astype(str).tolist()

    mp_holistic = mp.solutions.holistic
    done, skipped = 0, 0
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        for uid in uids:
            video_path = os.path.join(args.videos_dir, f"{uid}.mp4")
            if not os.path.exists(video_path):
                print(f"  Skipping {uid}: no video file at {video_path}")
                skipped += 1
                continue

            pose, face = extract_clip_keypoints(video_path, holistic)
            np.save(os.path.join(pose_dir, f"{uid}.npy"), pose)
            np.save(os.path.join(face_dir, f"{uid}.npy"), face)
            done += 1
            print(f"[{done}] {uid} -> {pose.shape[0]} frames")

    print(f"Done. {done} clips extracted, {skipped} skipped (missing video). "
          f"Now copy/symlink {args.labels_csv} into {args.out_dir}/ISLTranslate.csv if not already there.")


if __name__ == "__main__":
    main()
