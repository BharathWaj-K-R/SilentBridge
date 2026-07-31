"""
Base pose + facial-expression fusion transformer.

This is the FROZEN backbone: once pretrained on ISLTranslate / ISL-CSLTR /
iSign / INCLUDE, its weights never change at inference time. Signer-specific
personalization happens entirely in BridgeAdapter (see bridge_adapter.py),
which is the actual novel contribution of this project.

Architecture (encoder-only, since we're doing continuous sign -> text and
plan to decode with a separate lightweight head or CTC, not full seq2seq):

    pose_stream   -> PoseEncoder   -\
                                     +--> CrossModalFusion --> TransformerEncoder --> output head
    face_stream   -> FaceEncoder   -/

Both streams take pre-extracted keypoints (e.g. from MediaPipe Holistic),
NOT raw video, so this stays lightweight enough to run on a laptop/cloud CPU
for the demo, with GPU only speeding things up.
"""
import torch
import torch.nn as nn


class StreamEncoder(nn.Module):
    """Encodes a single stream of keypoints (pose OR face) into a sequence
    of d_model-dim embeddings, one per frame."""

    def __init__(self, input_dim: int, d_model: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1024, d_model))  # supports up to 1024 frames
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, frames, input_dim)
        seq_len = x.shape[1]
        h = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]
        return self.encoder(h)  # (batch, frames, d_model)


class CrossModalFusion(nn.Module):
    """Fuses pose and face embeddings via cross-attention: pose attends to
    face as context, so facial expression (grammar/negation/emotion markers
    in ISL) modulates the pose-based sign representation."""

    def __init__(self, d_model: int = 256, n_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, pose_emb: torch.Tensor, face_emb: torch.Tensor) -> torch.Tensor:
        fused, _ = self.cross_attn(query=pose_emb, key=face_emb, value=face_emb)
        return self.norm(pose_emb + fused)


class SilentBridgeBaseModel(nn.Module):
    """Full frozen backbone: pose + face -> fusion -> shared transformer ->
    logits over the output vocabulary (glosses or subword text tokens,
    decided at data-pipeline stage)."""

    def __init__(
        self,
        pose_input_dim: int = 132,   # e.g. 33 MediaPipe pose landmarks * 4 (x,y,z,visibility)
        face_input_dim: int = 1434,  # e.g. 478 face mesh landmarks * 3 (x,y,z)
        d_model: int = 256,
        vocab_size: int = 3000,      # placeholder; set from actual tokenizer/vocab at train time
        shared_layers: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        self.pose_encoder = StreamEncoder(pose_input_dim, d_model, n_layers=2, n_heads=n_heads)
        self.face_encoder = StreamEncoder(face_input_dim, d_model, n_layers=2, n_heads=n_heads)
        self.fusion = CrossModalFusion(d_model, n_heads)

        shared_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
        )
        self.shared_encoder = nn.TransformerEncoder(shared_layer, num_layers=shared_layers)
        self.output_head = nn.Linear(d_model, vocab_size)

        # This is the hook point BridgeAdapter attaches to (see bridge_adapter.py):
        # adapters wrap self.shared_encoder's layers rather than replacing them.
        self.d_model = d_model

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pose: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        pose_emb = self.pose_encoder(pose)
        face_emb = self.face_encoder(face)
        fused = self.fusion(pose_emb, face_emb)
        hidden = self.shared_encoder(fused)
        return self.output_head(hidden)  # (batch, frames, vocab_size)


def load_frozen_base_model(weights_path: str | None = None, **kwargs) -> SilentBridgeBaseModel:
    """Instantiate the base model and freeze it. If weights_path is given and
    exists, load pretrained weights; otherwise return a randomly initialized
    model (useful for wiring up the API before training is done)."""
    model = SilentBridgeBaseModel(**kwargs)
    if weights_path:
        import os
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state)
    model.freeze()
    model.eval()
    return model
