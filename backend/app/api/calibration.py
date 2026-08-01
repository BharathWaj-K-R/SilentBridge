import torch
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.models import SignerAdapter
from app.db.session import get_db
from app.schemas.schemas import CalibrationRequest, CalibrationResult
from app.services.calibration_service import calibrate_new_adapter

router = APIRouter(prefix="/calibration", tags=["calibration"])


@router.post("", response_model=CalibrationResult)
def calibrate(
    payload: CalibrationRequest,
    db: Session = Depends(get_db),
):
    """Train a new BridgeAdapter for this signer from a calibration clip
    (target: ~300 seconds / 5 minutes) and persist it.

    Expects a flat JSON body:
    { "user_id": ..., "calibration_seconds": ..., "pose_keypoints": [[...]],
      "face_keypoints": [[...]], "target_labels": [...] }

    pose_keypoints / face_keypoints: (frames, feature_dim) — one clip.
    target_labels: sentence-level token ids for this clip (NOT per-frame —
    CTC loss handles the frame-to-token alignment internally; see
    BridgeAdapterStack.calibrate() for why).
    """
    pose = torch.tensor(payload.pose_keypoints, dtype=torch.float32).unsqueeze(0)
    face = torch.tensor(payload.face_keypoints, dtype=torch.float32).unsqueeze(0)
    target_labels = torch.tensor(payload.target_labels, dtype=torch.long).unsqueeze(0)
    target_lengths = torch.tensor([len(payload.target_labels)], dtype=torch.long)

    result = calibrate_new_adapter(pose, face, target_labels, target_lengths, payload.calibration_seconds)

    adapter_row = SignerAdapter(
        owner_id=payload.user_id,
        weights_path=result["weights_path"],
        calibration_seconds=result["calibration_seconds"],
        param_count=result["param_count"],
    )
    db.add(adapter_row)
    db.commit()
    db.refresh(adapter_row)

    return CalibrationResult(
        adapter_id=adapter_row.id,
        calibration_seconds=adapter_row.calibration_seconds,
        param_count=adapter_row.param_count,
    )
