"""Regression test for the clip0076-style crash:

    RuntimeError: The size of tensor a (4500) must match the size of
    tensor b (1024) at non-singleton dimension 1

which happened because one ISL-CSLTR clip has 4500 frames while
StreamEncoder's positional embedding only supports MAX_SEQUENCE_LENGTH
(1024). collate_ctc_batch must uniformly downsample any clip longer than
MAX_SEQUENCE_LENGTH to exactly that many frames — for both pose and face,
using one shared index array so the two streams stay aligned — before the
batch ever reaches the model.
"""
import torch

from app.models.base_model import (
    FACE_INPUT_DIM,
    MAX_SEQUENCE_LENGTH,
    POSE_INPUT_DIM,
    SilentBridgeBaseModel,
)
from app.training.isltranslate import collate_ctc_batch


def _make_item(uid: str, frames: int, n_labels: int = 5) -> dict:
    return {
        "uid": uid,
        "text": "placeholder",
        "pose": torch.randn(frames, POSE_INPUT_DIM),
        "face": torch.randn(frames, FACE_INPUT_DIM),
        "labels": torch.randint(1, 20, (n_labels,), dtype=torch.long),
    }


def test_oversized_clip_is_downsampled_to_max_sequence_length():
    # Mirrors the real failure: one 4500-frame outlier (clip0076) mixed
    # with normal-length clips in the same batch.
    batch = [_make_item("clip0076", frames=4500), _make_item("clip0001", frames=119)]

    out = collate_ctc_batch(batch)

    assert out["pose"].shape == (2, MAX_SEQUENCE_LENGTH, POSE_INPUT_DIM)
    assert out["face"].shape == (2, MAX_SEQUENCE_LENGTH, FACE_INPUT_DIM)
    # input_lengths must reflect the *post*-downsampling length, not 4500.
    assert out["input_lengths"].tolist() == [MAX_SEQUENCE_LENGTH, 119]
    assert (out["input_lengths"] <= MAX_SEQUENCE_LENGTH).all()


def test_normal_length_clips_are_left_unchanged():
    for frames in (25, 50, 100, 119, 500, MAX_SEQUENCE_LENGTH):
        batch = [_make_item("clip", frames=frames)]
        out = collate_ctc_batch(batch)
        assert out["pose"].shape[1] == frames
        assert out["input_lengths"].tolist() == [frames]


def test_pose_face_frame_mismatch_raises_with_uid():
    bad_item = _make_item("clip_bad", frames=100)
    bad_item["face"] = torch.randn(99, FACE_INPUT_DIM)  # deliberately misaligned
    try:
        collate_ctc_batch([bad_item])
    except ValueError as exc:
        assert "clip_bad" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched pose/face frame counts")


def test_downsampled_batch_passes_through_model_with_finite_ctc_loss():
    # End-to-end proof that the original crash cannot happen again: the
    # exact shapes from the bug report (4500 frames) go through the real
    # collator and the real model.
    batch = [_make_item("clip0076", frames=4500), _make_item("clip0001", frames=119)]
    out = collate_ctc_batch(batch)

    vocab_size = 49  # matches the vocab size in the bug report's logits shape
    model = SilentBridgeBaseModel(vocab_size=vocab_size)
    logits = model(out["pose"], out["face"])

    assert logits.shape[:2] == (2, MAX_SEQUENCE_LENGTH)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    loss = loss_fn(log_probs, out["labels"], out["input_lengths"], out["label_lengths"])

    assert torch.isfinite(loss)
