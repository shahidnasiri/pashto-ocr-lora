import sys, json, random
import numpy as np, pandas as pd, cv2
from tqdm import tqdm
import config as C
from aug_lib import generate_background, replace_background, parse_annotation
from aug_quality import augment_image

SHARD = int(sys.argv[1])
OUT_DIR = C.AUG_DIR / "shards" / f"shard{SHARD}"
OUT_IMG = OUT_DIR / "images"
OUT_IMG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(C.SPLIT_DIR / "train.csv")            # TRAIN ONLY
shard_df = df.iloc[SHARD::C.N_SHARDS]
print(f"[shard {SHARD}] {len(shard_df)} sources x {C.MULTIPLIER}", flush=True)

rows = []
for _, row in tqdm(shard_df.iterrows(), total=len(shard_df), desc=f"shard{SHARD}"):
    src_id = int(row["id"])
    img = cv2.imread(str(C.RAW_IMG_DIR / row["file_name"]), cv2.IMREAD_COLOR)
    if img is None: continue
    h, w = img.shape[:2]
    theme = str(row.get("theme", "light")).lower()
    theme = theme if theme in ("light", "dark") else "light"

    main = parse_annotation(row.get("bbox"))
    if not (isinstance(main, list) and len(main) == 4): continue
    boxes, meta = [[float(v) for v in main]], [("main", None)]
    for key, kind in (("lines", "line"), ("tokens", "token")):
        items = parse_annotation(row.get(key))
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and isinstance(it.get("bbox"), list) and len(it["bbox"]) == 4:
                    boxes.append([float(v) for v in it["bbox"]]); meta.append((kind, it))
    labels = list(range(len(boxes)))

    for k in range(C.MULTIPLIER):
        try:
            s = C.SEED * 1_000_003 + src_id * 1000 + k
            random.seed(s); np.random.seed(s)
            comp = replace_background(img, generate_background(w, h, theme), theme)
            out_img, out_boxes, out_labels, applied = augment_image(img, comp, theme, boxes, labels, s)

            new_main, new_lines, new_toks = None, [], []
            for b, lab in zip(out_boxes, out_labels):
                kind, m = meta[int(lab)]
                b = [float(v) for v in b]
                if kind == "main": new_main = b
                else:
                    m = dict(m); m["bbox"] = b
                    (new_lines if kind == "line" else new_toks).append(m)
            if new_main is None: continue

            fn = f"s{SHARD}_aug_{src_id:06d}_{k:03d}.jpg"
            cv2.imwrite(str(OUT_IMG / fn), out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            r = row.to_dict()
            r.update(source_id=src_id, aug_index=k, file_name=fn, bbox=json.dumps(new_main),
                     lines=json.dumps(new_lines, ensure_ascii=False), tokens=json.dumps(new_toks, ensure_ascii=False),
                     bg_color="", font_color="", theme=theme, photometric_applied=int(applied))
            rows.append(r)
        except Exception:
            continue

pd.DataFrame(rows).to_csv(OUT_DIR / "shard.csv", index=False, encoding="utf-8-sig")
print(f"[shard {SHARD}] DONE: {len(rows)} rows", flush=True)
