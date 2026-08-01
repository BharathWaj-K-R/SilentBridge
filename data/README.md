# Data workspace

SilentBridge trains against the public ISLTranslate release from Exploration Lab:
https://github.com/Exploration-Lab/ISLTranslate

Download the dataset from the Hugging Face link documented by the upstream
repository, then arrange preprocessed keypoints as:

```text
data/processed/isltranslate/
├── ISLTranslate.csv
├── pose/<uid>.npy   # shape: frames x 132
└── face/<uid>.npy   # shape: frames x 1434
```

After the CSV and feature arrays are present, run:

```bash
PYTHONPATH=backend python -m app.training.train_base_model \
  --data-dir data/processed/isltranslate \
  --output backend/app/models/weights/base_model.pt
```

The trainer saves `base_model.pt` and a neighboring `base_model.vocab.json`.
Large dataset archives, extracted feature files, and trained weights should stay
out of git.
