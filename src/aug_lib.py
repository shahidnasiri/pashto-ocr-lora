import random, json, ast
import numpy as np
import cv2
import albumentations as A

A_MAJOR = int(A.__version__.split(".")[0])

def _light():
    if random.random() < 0.7:
        v = random.randint(210, 250); return (v, v, v)
    return tuple(random.randint(210, 250) for _ in range(3))

def _dark():
    if random.random() < 0.7:
        v = random.randint(10, 60); return (v, v, v)
    return tuple(random.randint(10, 60) for _ in range(3))

def make_solid(w, h, theme):
    bg = np.zeros((h, w, 3), np.uint8); bg[:] = _light() if theme == "light" else _dark(); return bg

def make_gradient(w, h, theme):
    base, amp = (235, 20) if theme == "light" else (30, 20)
    axis = random.choice(["x", "y", "diag"])
    yy, xx = np.mgrid[0:h, 0:w]
    if axis == "x": g = xx / max(w - 1, 1)
    elif axis == "y": g = yy / max(h - 1, 1)
    else: g = (xx / max(w - 1, 1) + yy / max(h - 1, 1)) / 2.0
    g = g * 2 - 1
    if random.random() < 0.5: g = -g
    v = np.clip(base + amp * g, 0, 255).astype(np.uint8)
    return np.stack([v, v, v], -1)

def make_paper(w, h, theme):
    coarse = np.random.rand(max(h // 8, 4), max(w // 8, 4)).astype(np.float32)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    tex = coarse * 0.6 + np.random.rand(h, w).astype(np.float32) * 0.15
    tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-6)
    v = (225 + tex * 30).astype(np.uint8) if theme == "light" else (15 + tex * 45).astype(np.uint8)
    return np.stack([v, v, v], -1)

def make_grid(w, h, theme):
    bgv, lv = (245, 200) if theme == "light" else (30, 60)
    bg = np.full((h, w, 3), bgv, np.uint8)
    step = random.choice([20, 30, 40, 50])
    for y in range(0, h, step): cv2.line(bg, (0, y), (w, y), (lv,) * 3, 1)
    for x in range(0, w, step): cv2.line(bg, (x, 0), (x, h), (lv,) * 3, 1)
    return bg

_BG = [make_solid, make_gradient, make_paper, make_paper, make_grid]   # paper texture weighted 2x
def generate_background(w, h, theme): return random.choice(_BG)(w, h, theme)

def replace_background(img, new_bg, theme):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if theme == "light":
        alpha = 1.0 - gray / 255.0
        sel = gray < np.percentile(gray, 5)
        tc = img[sel].mean(axis=0) if sel.any() else np.array([0, 0, 0], np.float32)
    else:
        alpha = gray / 255.0
        sel = gray > np.percentile(gray, 95)
        tc = img[sel].mean(axis=0) if sel.any() else np.array([255, 255, 255], np.float32)
    lo, hi = float(alpha.min()), float(alpha.max())
    if hi - lo > 1e-6: alpha = (alpha - lo) / (hi - lo)
    alpha = alpha[..., None]
    layer = np.broadcast_to(tc, img.shape).astype(np.float32)
    return np.clip(alpha * layer + (1 - alpha) * new_bg.astype(np.float32), 0, 255).astype(np.uint8)

# ---------- augmentation transforms (version-aware; nothing here can hide or blank the text) ----------
def _soft_light(image, **kwargs):
    # smooth, low-frequency brightness variation (+-12%) instead of hard dark polygon shadows
    h, w = image.shape[:2]
    field = cv2.resize(np.random.rand(3, 4).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    field = 1.0 + (field - 0.5) * 0.24
    return np.clip(image.astype(np.float32) * field[..., None], 0, 255).astype(np.uint8)

def _noise(p):
    if A_MAJOR >= 2: return A.GaussNoise(std_range=(0.02, 0.06), mean_range=(0.0, 0.0), p=p)   # about 5-15 grey levels
    return A.GaussNoise(var_limit=(10.0, 50.0), p=p)

def _affine():
    kw = dict(translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)}, scale=(0.98, 1.02), rotate=(-1, 1), p=0.40)
    # BORDER_REPLICATE extends the background at the edges, so there are never black bars
    if A_MAJOR >= 2: return A.Affine(border_mode=cv2.BORDER_REPLICATE, **kw)
    return A.Affine(mode=cv2.BORDER_REPLICATE, **kw)

def _bbox_params(min_vis):
    return A.BboxParams(format="coco", label_fields=["box_labels"], min_visibility=min_vis, min_area=1.0)

def _compose(tfms, bp, seed):
    if A_MAJOR >= 2: return A.Compose(tfms, bbox_params=bp, seed=seed)
    return A.Compose(tfms, bbox_params=bp)

def build_pipeline(theme, seed=None):
    return _compose([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.20, p=0.6),
        A.Lambda(image=_soft_light, p=0.30),
        A.RandomGamma(gamma_limit=(90, 110), p=0.30),
        A.GaussianBlur(blur_limit=(3, 5), p=0.25), _noise(0.25),
        A.ImageCompression(quality_range=(70, 95), p=0.30), _affine()], _bbox_params(0.4), seed)

def build_mild_pipeline(seed=None):
    return _compose([A.RandomBrightnessContrast(p=0.5), A.Lambda(image=_soft_light, p=0.3), A.GaussianBlur(p=0.2)],
                    _bbox_params(0.1), seed)

def parse_annotation(raw):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)): return []
    if isinstance(raw, (list, dict)): return raw
    s = str(raw).strip()
    if not s or s.lower() == "nan": return []
    try: return json.loads(s)
    except Exception:
        try: return ast.literal_eval(s)
        except Exception: return []
