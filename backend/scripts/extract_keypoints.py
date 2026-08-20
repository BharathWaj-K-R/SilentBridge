"""
Extract pose + face keypoints from raw ISLTranslate video clips using
MediaPipe Holistic, saving them in the exact layout
app/training/isltranslate.py's ISLTranslateKeypointDataset expects:

    data/processed/isltranslate/
    ├── ISLTranslate.csv   (written by this script — only successfully
    │                       extracted, dimension-validated uids)
    ├── pose/<uid>.npy     (frames, 132)
    └── face/<uid>.npy     (frames, 1404)

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

Resumable: if pose/<uid>.npy and face/<uid>.npy already exist and pass
dimension validation, that uid is skipped and counted under "already
processed" rather than re-extracted. Re-run this script freely — it will
only do new work.

Fault-tolerant: one bad clip is logged to <out_dir>/extraction_failures.csv
and skipped; it does not abort the run for the remaining clips.

Both <out_dir>/ISLTranslate.csv (the completion manifest) and
<out_dir>/extraction_failures.csv are written row-by-row and flushed to
disk immediately after each video, not batched up and written once at the
end — so if the process is killed partway through (Studio stop, OOM,
Ctrl-C), both files still reflect every video completed up to that point.
"""
import argparse
import os
import sys
import types

import numpy as np
import pandas as pd


# Single source of truth for the MediaPipe Holistic feature contract —
# mirrors app/models/base_model.py's POSE_INPUT_DIM / FACE_INPUT_DIM. Legacy
# mp.solutions.holistic (no iris refinement) gives 33 pose landmarks
# (x,y,z,visibility) and 468 face landmarks (x,y,z).
POSE_FEATURE_DIM = 33 * 4   # 132
FACE_FEATURE_DIM = 468 * 3  # 1404


def _stub_tensorflow_for_mediapipe() -> None:
    """mediapipe's __init__.py unconditionally imports mediapipe.tasks.python,
    whose audio submodule eagerly does
    `from tensorflow.tools.docs import doc_controls` purely for a no-op docs
    decorator — unrelated to anything this script actually uses (legacy
    mp.solutions.holistic). On environments with a real tensorflow already
    installed (e.g. Colab) whose protobuf pin conflicts with mediapipe's own,
    that unrelated import chain blows up with
    "ImportError: cannot import name 'runtime_version' from 'google.protobuf'".
    Fix: pre-register a fake tensorflow.tools.docs module in sys.modules
    before mediapipe is ever imported, so Python's import system uses this
    stub instead of loading the real (version-conflicting) tensorflow at all.
    Skipped if tensorflow isn't installed or already imported successfully."""
    if "tensorflow" in sys.modules:
        return
    try:
        import tensorflow  # noqa: F401 — if this actually works, no stub needed
        return
    except Exception:
        pass

    def _noop_decorator(f):
        return f

    fake_tf = types.ModuleType("tensorflow")
    fake_tf_tools = types.ModuleType("tensorflow.tools")
    fake_tf_tools_docs = types.ModuleType("tensorflow.tools.docs")
    fake_tf_tools_docs.doc_controls = types.SimpleNamespace(
        do_not_generate_docs=_noop_decorator,
        for_subclass_implementers=_noop_decorator,
        do_not_doc_inheritable=_noop_decorator,
    )
    fake_tf.tools = fake_tf_tools
    fake_tf_tools.docs = fake_tf_tools_docs
    sys.modules["tensorflow"] = fake_tf
    sys.modules["tensorflow.tools"] = fake_tf_tools
    sys.modules["tensorflow.tools.docs"] = fake_tf_tools_docs


_stub_tensorflow_for_mediapipe()


def extract_clip_keypoints(video_path: str, holistic) -> tuple[np.ndarray, np.ndarray]:
    """Runs MediaPipe Holistic over every frame of one clip. Raises
    ValueError if any frame produces an unexpected landmark count — this
    used to only happen inconsistently (zeros-fallback used a different
    dimension than the real-landmark case), which crashed np.stack() later
    with mismatched per-frame shapes. Now every frame — real or
    zeros-fallback — is exactly POSE_FEATURE_DIM / FACE_FEATURE_DIM, and any
    genuine mismatch (a MediaPipe version returning a different landmark
    count) fails loudly right here instead of silently corrupting the
    stacked array."""
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
            pose = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
                dtype=np.float32,
            ).flatten()
            if pose.shape[0] != POSE_FEATURE_DIM:
                raise ValueError(
                    f"Unexpected pose feature dimension: {pose.shape[0]}, "
                    f"expected {POSE_FEATURE_DIM}"
                )
        else:
            pose = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
        pose_frames.append(pose)

        if results.face_landmarks:
            face = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark],
                dtype=np.float32,
            ).flatten()
            if face.shape[0] != FACE_FEATURE_DIM:
                raise ValueError(
                    f"Unexpected face feature dimension: {face.shape[0]}, "
                    f"expected {FACE_FEATURE_DIM}"
                )
        else:
            face = np.zeros(FACE_FEATURE_DIM, dtype=np.float32)
        face_frames.append(face)

    cap.release()
    if not pose_frames:
        raise ValueError(f"No frames read from {video_path} — check the file isn't corrupt")

    pose_arr = np.stack(pose_frames)
    face_arr = np.stack(face_frames)
    if pose_arr.shape[0] != face_arr.shape[0]:
        raise ValueError(
            f"pose/face frame count mismatch: {pose_arr.shape[0]} vs {face_arr.shape[0]}"
        )
    return pose_arr, face_arr


def _load_and_validate(path: str, expected_dim: int) -> np.ndarray | None:
    """Loads a previously-saved .npy and checks it's actually usable —
    2D, right feature dimension. Returns None (treat as not-yet-processed)
    on any problem, so a corrupted leftover from an interrupted run gets
    reprocessed rather than silently accepted."""
    if not os.path.exists(path):
        return None
    try:
        arr = np.load(path)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        return None
    return arr


def resolve_uid_column(columns: list[str]) -> str:
    lower = {c.lower(): c for c in columns}
    for candidate in ("uid", "video_uid", "id"):
        if candidate in lower:
            return lower[candidate]
    raise ValueError(f"No uid-like column found in CSV. Columns: {columns}")


def resolve_text_column(columns: list[str]) -> str:
    lower = {c.lower(): c for c in columns}
    for candidate in ("text", "translation", "english"):
        if candidate in lower:
            return lower[candidate]
    raise ValueError(f"No text-like column found in CSV. Columns: {columns}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos_dir", required=True, help="Folder of <uid>.mp4 files")
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--out_dir", required=True,
                         help="e.g. data/processed/isltranslate — pose/ and face/ subfolders created here")
    args = parser.parse_args()

    import csv

    import mediapipe as mp

    pose_dir = os.path.join(args.out_dir, "pose")
    face_dir = os.path.join(args.out_dir, "face")
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(face_dir, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    uid_col = resolve_uid_column(list(df.columns))
    text_col = resolve_text_column(list(df.columns))

    already, newly, missing_video, failed = 0, 0, 0, 0

    # Both the manifest and the failure log are written row-by-row and
    # flushed to disk immediately, not accumulated in memory and written
    # once at the end. If the process is killed (Studio stop, OOM, Ctrl-C)
    # partway through, both files still reflect everything completed up to
    # that point — the manifest doubles as a live completion record, not
    # just a final artifact. Each run rewrites both from scratch (truncated
    # here), which is safe: uids already valid on disk are re-validated
    # near-instantly by _load_and_validate and re-appended within the same
    # pass, so a resumed run quickly restores full manifest coverage before
    # spending any time on genuinely new extraction work.
    manifest_path = os.path.join(args.out_dir, "ISLTranslate.csv")
    failures_path = os.path.join(args.out_dir, "extraction_failures.csv")

    mp_holistic = mp.solutions.holistic
    with open(manifest_path, "w", newline="", encoding="utf-8") as manifest_fh, \
         open(failures_path, "w", newline="", encoding="utf-8") as failures_fh, \
         mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:

        manifest_writer = csv.writer(manifest_fh)
        manifest_writer.writerow(["uid", "text"])
        manifest_fh.flush()

        failures_writer = csv.writer(failures_fh)
        failures_writer.writerow(["uid", "video_path", "exception_type", "exception_message"])
        failures_fh.flush()

        def _mark_complete(uid: str, text: str) -> None:
            manifest_writer.writerow([uid, text])
            manifest_fh.flush()
            os.fsync(manifest_fh.fileno())

        def _mark_failed(uid: str, video_path: str, exc: Exception) -> None:
            failures_writer.writerow([uid, video_path, type(exc).__name__, str(exc)])
            failures_fh.flush()
            os.fsync(failures_fh.fileno())

        for _, row in df.iterrows():
            uid = str(row[uid_col]).strip()
            text = str(row[text_col]).strip()
            if not uid or not text:
                continue

            pose_path = os.path.join(pose_dir, f"{uid}.npy")
            face_path = os.path.join(face_dir, f"{uid}.npy")

            existing_pose = _load_and_validate(pose_path, POSE_FEATURE_DIM)
            existing_face = _load_and_validate(face_path, FACE_FEATURE_DIM)
            if (
                existing_pose is not None
                and existing_face is not None
                and existing_pose.shape[0] == existing_face.shape[0]
            ):
                already += 1
                _mark_complete(uid, text)
                continue

            video_path = os.path.join(args.videos_dir, f"{uid}.mp4")
            if not os.path.exists(video_path):
                print(f"  Skipping {uid}: no video file at {video_path}")
                missing_video += 1
                continue

            try:
                pose, face = extract_clip_keypoints(video_path, holistic)
                # Save immediately on success, before moving to the next
                # video — so a completed clip is never lost to an
                # interruption that happens later in the loop.
                np.save(pose_path, pose)
                np.save(face_path, face)
                newly += 1
                _mark_complete(uid, text)
                print(f"[{already + newly}] {uid} -> {pose.shape[0]} frames")
            except Exception as exc:
                failed += 1
                _mark_failed(uid, video_path, exc)
                print(f"  FAILED {uid}: {type(exc).__name__}: {exc}")
                continue

    total_valid = already + newly
    print(f"\nAlready processed: {already}")
    print(f"Newly processed: {newly}")
    print(f"Failed: {failed}")
    print(f"Skipped (missing video): {missing_video}")
    print(f"Total valid outputs: {total_valid}")
    print(f"Failure report: {failures_path}")
    print(f"Validated manifest written: {manifest_path} ({total_valid} rows)")


if __name__ == "__main__":
    main()
