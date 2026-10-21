import numpy as np, cv2
import albumentations as A
from aug_lib import A_MAJOR, _noise, _affine, _bbox_params, _compose

MIN_RATIO = 0.70      # augmented text must keep at least 70% of its contrast against the background
MAX_TRIES = 6

def _gentle_light(image, **kwargs):
    # smooth brightness variation, +-8%
    h, w = image.shape[:2]
    field = cv2.resize(np.random.rand(3, 4).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    field = 1.0 + (field - 0.5) * 0.16
    return np.clip(image.astype(np.float32) * field[..., None], 0, 255).astype(np.uint8)

def _plain_compose(tfms, seed):
    if A_MAJOR >= 2: return A.Compose(tfms, seed=seed)
    return A.Compose(tfms)

def _photometric():
    return [
        A.RandomBrightnessContrast(brightness_limit=(-0.10, 0.10), contrast_limit=(-0.10, 0.10), p=0.5),
        A.Lambda(image=_gentle_light, p=0.3),
        A.RandomGamma(gamma_limit=(90, 110), p=0.3),
        A.GaussianBlur(blur_limit=(3, 3), p=0.2),
        _noise(0.25),
        A.ImageCompression(quality_range=(75, 95), p=0.3),
    ]

def text_masks(img, theme):
    # text pixels / background pixels, found from the original image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    alpha = (1.0 - gray / 255.0) if theme == "light" else (gray / 255.0)
    lo, hi = float(alpha.min()), float(alpha.max())
    alpha = (alpha - lo) / (hi - lo + 1e-6)
    return alpha > 0.8, alpha < 0.05

def contrast(img, tmask, bmask):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return abs(float(g[bmask].mean()) - float(g[tmask].mean()))

def augment_image(orig, comp, theme, boxes, labels, seed):
    # returns (image, boxes, labels, photometric_applied)
    tmask, bmask = text_masks(orig, theme)
    img, applied = comp, False
    if tmask.any() and bmask.any():
        base = contrast(comp, tmask, bmask)
        for attempt in range(MAX_TRIES):
            cand = _plain_compose(_photometric(), seed + attempt)(image=comp)["image"]
            if contrast(cand, tmask, bmask) >= MIN_RATIO * base:     # text still clearly visible?
                img, applied = cand, True
                break
    # tiny geometric change with box tracking; if too many boxes are lost, skip the geometry
    out = _compose([_affine()], _bbox_params(0.4), seed)(image=img, bboxes=boxes, box_labels=labels)
    if len(out["bboxes"]) < max(1, int(0.85 * len(boxes))):
        return img, boxes, labels, applied
    return out["image"], out["bboxes"], out["box_labels"], applied
