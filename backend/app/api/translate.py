import torch
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import SignerAdapter, TranslationLog
from app.db.session import get_db
from app.schemas.schemas import TranslationRequest, TranslationResult
from app.services.calibration_service import load_adapter_for_signer
from app.services.inference_service import get_base_model, run_inference

router = APIRouter(prefix="/translate", tags=["translate"])
settings = get_settings()


@router.post("", response_model=TranslationResult)
def translate(
    payload: TranslationRequest,
    pose_keypoints: list[list[float]],
    face_keypoints: list[list[float]],
    db: Session = Depends(get_db),
):
    """Translate one clip's worth of pre-extracted pose + face keypoints.

    pose_keypoints / face_keypoints: shape (frames, feature_dim), already
    extracted client- or server-side (e.g. via MediaPipe Holistic). This
    endpoint does NOT do video decoding or keypoint extraction itself.
    """
    pose = torch.tensor(pose_keypoints, dtype=torch.float32).unsqueeze(0)  # (1, frames, dim)
    face = torch.tensor(face_keypoints, dtype=torch.float32).unsqueeze(0)

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
