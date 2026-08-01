"""
BridgeAdapter — the core novel contribution of SilentBridge.

Goal: personalize the frozen SilentBridgeBaseModel to a NEW signer's style
(hand shape variation, signing speed, regional dialect) using only ~5
minutes of calibration video, updating < 2% of the base model's parameter
count, while keeping inference latency < 500ms and memory overhead < 10MB.

Design: bottleneck adapter layers (Houlsby-style) inserted after each
TransformerEncoderLayer in the base model's shared_encoder. Only these small
bottleneck layers are trained during calibration; everything else stays
frozen. This is the same family of idea as adapters in NLP transfer
learning, applied here to a pose+face fusion transformer for sign language.

    frozen_layer_output --> down_proj(d_model -> bottleneck) --> ReLU
                         --> up_proj(bottleneck -> d_model) --> + residual

Target sizing to hit the <2% param budget (example, tune against your
actual base model size once trained):
    d_model = 256, bottleneck = 16, n_layers_to_adapt = 4
    params_per_adapter_layer = 2 * (256*16 + 16) ≈ 8,320
    total_adapter_params ≈ 4 * 8,320 ≈ 33,280
    -> compare against total base model param count to confirm <2%.
"""
import time

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """A single Houlsby-style bottleneck adapter, inserted after one frozen
    transformer layer's output."""

    def __init__(self, d_model: int = 256, bottleneck_dim: int = 16):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        # Zero-init the up-projection so the adapter starts as a no-op
        # (identity function) and calibration nudges it gently from there.
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.up_proj(h)
        return residual + h

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class BridgeAdapterStack(nn.Module):
    """Holds one BottleneckAdapter per adapted layer of the base model's
    shared_encoder, plus the calibration/inference logic around them.

    Usage:
        base_model = load_frozen_base_model(...)
        adapter = BridgeAdapterStack(d_model=base_model.d_model,
                                      n_layers=len(base_model.shared_encoder.layers))
        adapter.calibrate(base_model, calibration_pose, calibration_face, labels)
        logits = adapter.forward_with_base(base_model, pose, face)
    """

    def __init__(self, d_model: int = 256, n_layers: int = 4, bottleneck_dim: int = 16):
        super().__init__()
        self.adapters = nn.ModuleList([
            BottleneckAdapter(d_model, bottleneck_dim) for _ in range(n_layers)
        ])
        self.confidence_aware = False  # stretch goal flag; off until core adapter works

    def total_param_count(self) -> int:
        return sum(a.param_count() for a in self.adapters)

    def param_budget_ok(self, base_model_param_count: int, budget_pct: float = 2.0) -> bool:
        pct = 100.0 * self.total_param_count() / max(base_model_param_count, 1)
        return pct <= budget_pct

    def forward_with_base(self, base_model, pose: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        """Runs the frozen base model's forward pass but inserts each
        BottleneckAdapter after the corresponding shared_encoder layer.
        Mirrors SilentBridgeBaseModel.forward but layer-by-layer so adapters
        can be spliced in."""
        pose_emb = base_model.pose_encoder(pose)
        face_emb = base_model.face_encoder(face)
        hidden = base_model.fusion(pose_emb, face_emb)

        for layer, adapter in zip(base_model.shared_encoder.layers, self.adapters):
            hidden = layer(hidden)
            hidden = adapter(hidden)

        return base_model.output_head(hidden)

    def calibrate(
        self,
        base_model,
        calibration_pose: torch.Tensor,
        calibration_face: torch.Tensor,
        target_labels: torch.Tensor,
        target_lengths: torch.Tensor,
        epochs: int = 20,
        lr: float = 1e-3,
        blank_id: int = 0,
    ) -> dict:
        """Trains ONLY the adapter parameters on the signer's calibration
        clip, using CTC loss.

        WHY CTC, NOT PER-FRAME CROSS-ENTROPY: a calibration/training clip has
        ONE sentence-level label (a sequence of tokens), not a label for
        every individual frame — there's no frame-to-token alignment given
        by the dataset. CTC loss is built exactly for this: it marginalizes
        over all possible frame-to-token alignments internally, which is
        the standard approach for speech/handwriting/sign recognition.
        (Same approach as app/training/train_base_model.py uses for the
        base model itself — this keeps adapter calibration consistent
        with how the base model was trained.)

        calibration_pose / calibration_face: (batch, frames, feature_dim)
        target_labels: (batch, max_target_len) — token ids, no blank inside,
            padded with any value beyond target_lengths[i] (ignored).
        target_lengths: (batch,) — true (unpadded) length of each target.
        blank_id: vocab index reserved for the CTC blank token. Must match
            what decode_logits() in inference_service.py treats as blank.

        Base model stays frozen (its params already have requires_grad=False
        from load_frozen_base_model). Returns timing/param stats for the
        ablation study and for storing in SignerAdapter.accuracy_gain_pct
        upstream."""
        start = time.time()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

        batch_size, n_frames = calibration_pose.shape[0], calibration_pose.shape[1]
        input_lengths = torch.full((batch_size,), n_frames, dtype=torch.long)

        self.train()
        loss = torch.tensor(0.0)
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.forward_with_base(base_model, calibration_pose, calibration_face)
            log_probs = torch.log_softmax(logits, dim=-1)  # (batch, frames, vocab)
            log_probs = log_probs.permute(1, 0, 2)  # CTCLoss wants (frames, batch, vocab)
            loss = ctc_loss(log_probs, target_labels, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
        self.eval()

        elapsed = time.time() - start
        return {
            "calibration_wall_seconds": elapsed,
            "final_loss": float(loss.item()),
            "param_count": self.total_param_count(),
        }

    def maybe_update(self, base_confidence: float, threshold: float = 0.6) -> bool:
        """Stretch-goal hook: confidence-aware calibration. Only relevant
        once confidence_aware=True is turned on deliberately after the core
        adapter is working end-to-end. Not part of core scope."""
        if not self.confidence_aware:
            return True
        return base_confidence < threshold
