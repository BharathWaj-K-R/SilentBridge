"""ISLTranslate dataset loading utilities.

The upstream ISLTranslate release provides an ``ISLTranslate.csv`` metadata
file and pre-extracted MediaPipe Holistic feature archives. This module expects
those archives to be unpacked locally and converted to per-UID feature files
containing pose and face arrays before model training.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ISLTranslateExample:
    uid: str
    text: str
    pose_path: Path
    face_path: Path


class SimpleCharTokenizer:
    """Small deterministic character tokenizer for initial CTC training.

    It is intentionally lightweight so the base model can be trained before a
    production subword vocabulary is chosen. Token id 0 is reserved for the CTC
    blank/pad symbol used by ``torch.nn.CTCLoss``.
    """

    blank_token = "<blank>"

    def __init__(self, alphabet: str | None = None):
        if alphabet is None:
            alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 .,?!'\"-:;()"
        self.id_to_token = [self.blank_token, *dict.fromkeys(alphabet.lower())]
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id[ch] for ch in text.lower() if ch in self.token_to_id and ch != self.blank_token]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id_to_token": self.id_to_token}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SimpleCharTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokenizer = cls("")
        tokenizer.id_to_token = payload["id_to_token"]
        tokenizer.token_to_id = {token: idx for idx, token in enumerate(tokenizer.id_to_token)}
        return tokenizer


class ISLTranslateKeypointDataset(Dataset):
    """Loads aligned ISLTranslate metadata and pose/face feature arrays.

    Directory convention after preprocessing::

        data/processed/isltranslate/
        ├── ISLTranslate.csv
        ├── pose/<uid>.npy
        └── face/<uid>.npy
    """

    def __init__(self, root: str | Path, tokenizer: SimpleCharTokenizer | None = None):
        self.root = Path(root)
        self.tokenizer = tokenizer or SimpleCharTokenizer()
        self.examples = self._read_examples()
        if not self.examples:
            raise ValueError(f"No usable ISLTranslate examples found under {self.root}")

    def _read_examples(self) -> list[ISLTranslateExample]:
        metadata_path = self.root / "ISLTranslate.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")

        examples: list[ISLTranslateExample] = []
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = {name.lower(): name for name in (reader.fieldnames or [])}
            uid_field = fieldnames.get("uid") or fieldnames.get("video_uid") or fieldnames.get("id")
            text_field = fieldnames.get("text") or fieldnames.get("translation") or fieldnames.get("english")
            if not uid_field or not text_field:
                raise ValueError("ISLTranslate.csv must include uid and translation/text columns")

            for row in reader:
                uid = row[uid_field].strip()
                text = row[text_field].strip()
                pose_path = self.root / "pose" / f"{uid}.npy"
                face_path = self.root / "face" / f"{uid}.npy"
                if uid and text and pose_path.exists() and face_path.exists():
                    examples.append(ISLTranslateExample(uid, text, pose_path, face_path))
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        pose = torch.from_numpy(np.load(example.pose_path)).float()
        face = torch.from_numpy(np.load(example.face_path)).float()
        labels = torch.tensor(self.tokenizer.encode(example.text), dtype=torch.long)
        return {"uid": example.uid, "pose": pose, "face": face, "labels": labels, "text": example.text}


def collate_ctc_batch(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    pose_dim = int(batch[0]["pose"].shape[-1])  # type: ignore[index, union-attr]
    face_dim = int(batch[0]["face"].shape[-1])  # type: ignore[index, union-attr]
    max_frames = max(int(item["pose"].shape[0]) for item in batch)  # type: ignore[index, union-attr]

    pose = torch.zeros(len(batch), max_frames, pose_dim)
    face = torch.zeros(len(batch), max_frames, face_dim)
    input_lengths = torch.zeros(len(batch), dtype=torch.long)
    label_chunks = []
    label_lengths = torch.zeros(len(batch), dtype=torch.long)
    uids: list[str] = []
    texts: list[str] = []

    for idx, item in enumerate(batch):
        item_pose = item["pose"]  # type: ignore[assignment]
        item_face = item["face"]  # type: ignore[assignment]
        item_labels = item["labels"]  # type: ignore[assignment]
        frames = int(item_pose.shape[0])
        pose[idx, :frames] = item_pose
        face[idx, :frames] = item_face
        input_lengths[idx] = frames
        label_chunks.append(item_labels)
        label_lengths[idx] = int(item_labels.numel())
        uids.append(str(item["uid"]))
        texts.append(str(item["text"]))

    return {
        "uid": uids,
        "text": texts,
        "pose": pose,
        "face": face,
        "labels": torch.cat(label_chunks),
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
    }
