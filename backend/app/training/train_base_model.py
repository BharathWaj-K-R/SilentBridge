"""Train SilentBridgeBaseModel on preprocessed ISLTranslate keypoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from app.models.base_model import SilentBridgeBaseModel
from app.training.isltranslate import ISLTranslateKeypointDataset, SimpleCharTokenizer, collate_ctc_batch


def train(args: argparse.Namespace) -> None:
    tokenizer = SimpleCharTokenizer()
    dataset = ISLTranslateKeypointDataset(args.data_dir, tokenizer=tokenizer)
    val_size = max(1, int(len(dataset) * args.val_fraction)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    train_dataset, _ = random_split(dataset, [train_size, val_size]) if val_size else (dataset, [])

    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_ctc_batch)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SilentBridgeBaseModel(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            pose = batch["pose"].to(device)
            face = batch["face"].to(device)
            labels = batch["labels"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            optimizer.zero_grad()
            logits = model(pose, face)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = loss_fn(log_probs, labels, input_lengths, label_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.item())
        print(f"epoch={epoch} train_loss={total_loss / max(len(loader), 1):.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    tokenizer.save(output_path.with_suffix(".vocab.json"))
    print(f"saved weights to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/isltranslate")
    parser.add_argument("--output", default="backend/app/models/weights/base_model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
