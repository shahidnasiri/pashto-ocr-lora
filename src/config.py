import os
from pathlib import Path

# ---- EDIT ONLY THIS BLOCK ----------------------------------------------
ROOT         = Path(os.environ.get("PASHTO_OCR_ROOT", r"E:\MCIT-1405\PashtoOCR\Fine_tunning"))
RAW_CSV      = ROOT / "PashtoOCR" / "psocr.csv"        # the 10K annotations
RAW_IMG_DIR  = ROOT / "PashtoOCR" / "images"           # the 10K images
BASE_MODEL   = ROOT / "DeepSeek-OCR"                   # unsloth/DeepSeek-OCR snapshot
DEEPSEEK_SRC = ROOT / "DeepSeek-OCR"                   # folder that contains deepseek_ocr/
# ------------------------------------------------------------------------

WORK        = ROOT / "experiments"
SPLIT_DIR   = WORK / "splits"
AUG_DIR     = WORK / "augmented"
AUG_FINAL   = AUG_DIR / "final"
SUBSET_DIR  = WORK / "subsets"
MODELS_DIR  = WORK / "models"
RESULTS_DIR = WORK / "results"
FIG_DIR     = WORK / "figures"

SEED            = 42
N_TEST          = 1000      # held-out test images (never trained on, never augmented)
N_VAL           = 500       # validation images (never trained on, never augmented)
N_VAL_LOSS      = 200       # subset of val used for validation loss during training
N_SHARDS        = 8
MULTIPLIER      = 100       # augmented variants per training image
N_TRAIN_SAMPLES = 100_000   # samples seen by the rawrepeat and aug100k runs

PROMPT     = "<image>\nFree OCR. "
IMAGE_SIZE = 640
BASE_SIZE  = 640

TAGS   = ["zeroshot", "raw1x", "rawrepeat", "aug100k"]
LABELS = {"zeroshot": "Zero-shot base", "raw1x": "Fine-tuned raw, 1 epoch",
          "rawrepeat": "Fine-tuned raw, repeated to 100K", "aug100k": "Fine-tuned augmented 100K"}

for _d in (SPLIT_DIR, AUG_DIR, SUBSET_DIR, MODELS_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
