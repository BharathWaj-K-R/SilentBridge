"""
Calibration service: given ~5 minutes of a new signer's keypoint data +
ground-truth labels, trains a fresh BridgeAdapterStack and saves it to disk,
keyed to that user, so it can be loaded again for inference later.
"""
import os
import uuid

import torch

from app.core.config import get_settings
from app.models.base_model import load_frozen_base_model
from app.models.bridge_adapter import BridgeAdapterStack

settings = get_settings()


def calibrate_new_adapter(
    pose: torch.Tensor,
    face: torch.Tensor,
    labels: torch.Tensor,
    calibration_seconds: float,
) -> dict:
    """Trains a new adapter for one signer and saves its weights.
    Returns the info needed to create a SignerAdapter DB row (see
    db/models.py) — caller is responsible for persisting that row.
    """
    base_model = load_frozen_base_model(settings.BASE_MODEL_PATH)
    n_layers = len(base_model.shared_encoder.layers)

    adapter = BridgeAdapterStack(d_model=base_model.d_model, n_layers=n_layers)

    base_param_count = sum(p.numel() for p in base_model.parameters())
    stats = adapter.calibrate(base_model, pose, face, labels)

    if not adapter.param_budget_ok(base_param_count):
        # Still save it — but flag it, so the ablation report can note the
        # budget was missed rather than pretending it wasn't.
        stats["param_budget_ok"] = False
    else:
        stats["param_budget_ok"] = True

    os.makedirs(settings.ADAPTER_WEIGHTS_DIR, exist_ok=True)
    weights_path = os.path.join(settings.ADAPTER_WEIGHTS_DIR, f"{uuid.uuid4().hex}.pt")
    torch.save(adapter.state_dict(), weights_path)

    return {
        "weights_path": weights_path,
        "calibration_seconds": calibration_seconds,
        "param_count": stats["param_count"],
        "param_budget_ok": stats["param_budget_ok"],
        "final_loss": stats["final_loss"],
    }


def load_adapter_for_signer(weights_path: str, d_model: int, n_layers: int) -> BridgeAdapterStack:
    adapter = BridgeAdapterStack(d_model=d_model, n_layers=n_layers)
    state = torch.load(weights_path, map_location="cpu")
    adapter.load_state_dict(state)
    adapter.eval()
    return adapter
