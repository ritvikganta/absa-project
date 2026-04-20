# Data

- `raw/` — full train/test CSVs downloaded from HuggingFace (gitignored)
- `samples/` — 100-row train sample and 20-row test sample for submission

Run `python scripts/preprocess.py --download` to populate `raw/`.
Run `python scripts/preprocess.py --sample` to populate `samples/`.
