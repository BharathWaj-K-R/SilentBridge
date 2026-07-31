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
      "face_keypoints": [[...]], "label_ids": [...] }

    pose_keypoints / face_keypoints: (frames, feature_dim)
    label_ids: (frames,) token ids aligned to each frame (or use CTC-style
    sparse labels once the labeling scheme is finalized).
    """
    pose = torch.tensor(payload.pose_keypoints, dtype=torch.float32).unsqueeze(0)
    face = torch.tensor(payload.face_keypoints, dtype=torch.float32).unsqueeze(0)
    labels = torch.tensor(payload.label_ids, dtype=torch.long).unsqueeze(0)

    result = calibrate_new_adapter(pose, face, labels, payload.calibration_seconds)

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
