import torch
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import SignerAdapter, TranslationLog
from app.db.session import get_db
from app.models.base_model import FACE_INPUT_DIM, POSE_INPUT_DIM
from app.schemas.schemas import TranslationRequest, TranslationResult
from app.services.calibration_service import load_adapter_for_signer
from app.services.inference_service import get_base_model, run_inference

router = APIRouter(prefix="/translate", tags=["translate"])
settings = get_settings()


def _validate_keypoints(pose_keypoints: list[list[float]], face_keypoints: list[list[float]]) -> None:
    """Fails fast with a clear 422 on a malformed payload, instead of either
    crashing torch.tensor() on ragged input or letting a wrong-shaped tensor
    reach the model and fail deep inside a matmul with a confusing error.
    Mirrors the same contract enforced client-side in translate-live.js and
    at training time in isltranslate.py's collate_ctc_batch."""
    if not pose_keypoints or not face_keypoints:
        raise HTTPException(status_code=422, detail="pose_keypoints and face_keypoints must not be empty")
    if len(pose_keypoints) != len(face_keypoints):
        raise HTTPException(
            status_code=422,
            detail=f"pose/face frame count mismatch: {len(pose_keypoints)} vs {len(face_keypoints)}",
        )
    for idx, frame in enumerate(pose_keypoints):
        if len(frame) != POSE_INPUT_DIM:
            raise HTTPException(
                status_code=422,
                detail=f"pose frame {idx} has {len(frame)} dims, expected {POSE_INPUT_DIM}",
            )
    for idx, frame in enumerate(face_keypoints):
        if len(frame) != FACE_INPUT_DIM:
            raise HTTPException(
                status_code=422,
                detail=f"face frame {idx} has {len(frame)} dims, expected {FACE_INPUT_DIM}",
            )


@router.post("", response_model=TranslationResult)
def translate(
    payload: TranslationRequest,
    db: Session = Depends(get_db),
):
    """Translate one clip's worth of pre-extracted pose + face keypoints.

    Expects a flat JSON body:
    { "user_id": ..., "adapter_id": ..., "pose_keypoints": [[...]], "face_keypoints": [[...]] }

    pose_keypoints / face_keypoints: shape (frames, feature_dim), already
    extracted client- or server-side (e.g. via MediaPipe Holistic). This
    endpoint does NOT do video decoding or keypoint extraction itself.
    """
    _validate_keypoints(payload.pose_keypoints, payload.face_keypoints)

    pose = torch.tensor(payload.pose_keypoints, dtype=torch.float32).unsqueeze(0)  # (1, frames, dim)
    face = torch.tensor(payload.face_keypoints, dtype=torch.float32).unsqueeze(0)

    adapter = None
    if payload.adapter_id is not None:
        row = db.query(SignerAdapter).filter(SignerAdapter.id == payload.adapter_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Adapter not found")
        base_model = get_base_model()
        adapter = load_adapter_for_signer(
            row.weights_path, d_model=base_model.d_model,
            n_layers=len(base_model.shared_encoder.layers),
        )

    result = run_inference(pose, face, adapter=adapter)

    log = TranslationLog(
        user_id=payload.user_id,
        adapter_id=payload.adapter_id,
        predicted_text=result["predicted_text"],
        confidence=result["confidence"],
        latency_ms=result["latency_ms"],
        used_adapter=int(result["used_adapter"]),
    )
    db.add(log)
    db.commit()

    return TranslationResult(**result)
