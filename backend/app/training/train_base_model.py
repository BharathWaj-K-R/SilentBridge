"""Train SilentBridgeBaseModel on preprocessed ISLTranslate keypoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from app.models.base_model import SilentBridgeBaseModel
from app.training.isltranslate import ISLTranslateKeypointDataset, SimpleCharTokenizer, collate_ctc_batch


def run_epoch(model, loader, optimizer, loss_fn, device, max_grad_norm: float, train: bool) -> float:
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        pose = batch["pose"].to(device)
        face = batch["face"].to(device)
        labels = batch["labels"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(pose, face)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = loss_fn(log_probs, labels, input_lengths, label_lengths)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(args: argparse.Namespace) -> None:
    tokenizer = SimpleCharTokenizer()
    dataset = ISLTranslateKeypointDataset(args.data_dir, tokenizer=tokenizer)
    val_size = max(1, int(len(dataset) * args.val_fraction)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size]) if val_size else (dataset, None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_ctc_batch)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ctc_batch)
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SilentBridgeBaseModel(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # NOTE: previously this trained blindly for all epochs and saved only the
    # final one, even though a val split was created — the val set was never
    # actually used. Now we evaluate every epoch and only keep the
    # best-val-loss checkpoint, so a late epoch that overfits doesn't silently
    # clobber a better earlier one.
    best_val_loss = float("inf")
    start_epoch = 1

    # Full resumable checkpoint (model + optimizer + epoch + best_val_loss),
    # separate from output_path (which stays a plain best-weights state_dict
    # for the API loader). Only written/read if --checkpoint-dir is passed,
    # so default CLI behavior is unchanged.
    checkpoint_path = Path(args.checkpoint_dir) / "latest.pt" if args.checkpoint_dir else None
    if args.resume and checkpoint_path and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"resumed from {checkpoint_path}: starting at epoch={start_epoch}, best_val_loss={best_val_loss:.4f}")
    elif args.resume:
        print(f"--resume passed but no checkpoint found at {checkpoint_path} — starting from epoch 1")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device, args.max_grad_norm, train=True)

        if val_loader is not None:
            val_loss = run_epoch(model, val_loader, optimizer, loss_fn, device, args.max_grad_norm, train=False)
            print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_path)
                print(f"  -> new best (val_loss={val_loss:.4f}), saved checkpoint to {output_path}")
        else:
            # No val split possible (dataset too small) — fall back to
            # saving every epoch, since there's nothing to compare against.
            print(f"epoch={epoch} train_loss={train_loss:.4f} (no val split — dataset too small)")
            torch.save(model.state_dict(), output_path)

        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )

    tokenizer.save(output_path.with_suffix(".vocab.json"))
    print(f"training done. best_val_loss={best_val_loss if val_loader else 'n/a'}. weights at {output_path}")


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
    parser.add_argument("--checkpoint-dir", default=None,
                         help="if set, saves a full resumable checkpoint (model+optimizer+epoch) "
                              "to <checkpoint-dir>/latest.pt after every epoch")
    parser.add_argument("--resume", action="store_true",
                         help="resume from <checkpoint-dir>/latest.pt if it exists")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
