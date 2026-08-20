# Data workspace

SilentBridge trains against the public ISLTranslate release from Exploration Lab:
https://github.com/Exploration-Lab/ISLTranslate

The dataset lives on Hugging Face (`Exploration-Lab/iSign`), is **gated**
(requires a free HF account + accepting the license) and is large
(~228GB total) — download it on your own machine or a Colab instance, not
in a restricted sandbox.

**Team-of-2 scope:** don't pull the full release. Pull a small subset (a
few hundred clips, enough to cover a handful of signers for the adapter
demo) — full-corpus training isn't the goal here, proving the adapter
personalizes on top of a small base is.

## Preprocessing: two paths depending on what you have

**If the release gives you pre-extracted MediaPipe features in
`.pose`-format** (check the dataset card):
```bash
pip install -r ../backend/requirements-training.txt
# ALWAYS inspect one file first -- component layout isn't verified, see the
# warning at the top of the script:
python ../backend/scripts/convert_isign_pose.py --inspect_only path/to/one.pose
python ../backend/scripts/convert_isign_pose.py \
  --pose_dir path/to/iSign-poses \
  --labels_csv path/to/iSign_v1.1.csv \
  --out_dir data/processed/isltranslate
```

**If you only have raw video clips:**
```bash
pip install -r ../backend/requirements-training.txt
python ../backend/scripts/extract_keypoints.py \
  --videos_dir path/to/raw_videos \
  --labels_csv path/to/ISLTranslate.csv \
  --out_dir data/processed/isltranslate
```

Either way you end up with:

```text
data/processed/isltranslate/
├── ISLTranslate.csv
├── pose/<uid>.npy   # shape: frames x 132
└── face/<uid>.npy   # shape: frames x 1404
```

## Train

```bash
PYTHONPATH=backend python -m app.training.train_base_model \
  --data-dir data/processed/isltranslate \
  --output backend/app/models/weights/base_model.pt
```

The trainer now actually evaluates the held-out validation split every
epoch and only keeps the best-val-loss checkpoint (an earlier bug trained
blind and saved whatever the last epoch happened to be). It saves
`base_model.pt` and a neighboring `base_model.vocab.json`.

No GPU here to run this for real -- do it on Colab or a machine with a GPU.
Large dataset archives, extracted feature files, and trained weights should
stay out of git.
