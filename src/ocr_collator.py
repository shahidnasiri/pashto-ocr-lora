import math, io
from typing import Any
import torch
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
from deepseek_ocr.modeling_deepseekocr import text_encode, BasicImageTransform, dynamic_preprocess

class DeepSeekOCRDataCollator:
    def __init__(self, tokenizer, model, image_size=640, base_size=1024, crop_mode=True, train_on_responses_only=True):
        self.tokenizer, self.model = tokenizer, model
        self.image_size, self.base_size, self.crop_mode = image_size, base_size, crop_mode
        self.image_token_id = 128815
        self.dtype = model.dtype
        self.train_on_responses_only = train_on_responses_only
        self.image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        self.patch_size, self.downsample_ratio = 16, 4
        self.bos_id = getattr(tokenizer, "bos_token_id", None) or 0

    def deserialize_image(self, d):
        if isinstance(d, Image.Image): return d.convert("RGB")
        if isinstance(d, dict):
            if d.get("path") is not None: return Image.open(d["path"]).convert("RGB")
            if d.get("bytes") is not None: return Image.open(io.BytesIO(d["bytes"])).convert("RGB")
        raise ValueError(f"Unsupported image format: {type(d)}")

    def process_image(self, image):
        images_list, images_crop_list, spatial = [], [], []
        if self.crop_mode:
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio, crops = (1, 1), []
            else:
                crops, crop_ratio = dynamic_preprocess(image, min_num=2, max_num=9, image_size=self.image_size, use_thumbnail=False)
            gv = ImageOps.pad(image, (self.base_size, self.base_size), color=tuple(int(x * 255) for x in self.image_transform.mean))
            images_list.append(self.image_transform(gv).to(self.dtype))
            wc, hc = crop_ratio
            spatial.append([wc, hc])
            if wc > 1 or hc > 1:
                for ci in crops: images_crop_list.append(self.image_transform(ci).to(self.dtype))
            nq = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            nqb = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tok = ([self.image_token_id] * nqb + [self.image_token_id]) * nqb + [self.image_token_id]
            if wc > 1 or hc > 1:
                tok += ([self.image_token_id] * (nq * wc) + [self.image_token_id]) * (nq * hc)
        else:
            spatial.append([1, 1])
            if self.base_size <= 640:
                r = image.resize((self.base_size, self.base_size), Image.LANCZOS)
                images_list.append(self.image_transform(r).to(self.dtype))
            else:
                gv = ImageOps.pad(image, (self.base_size, self.base_size), color=tuple(int(x * 255) for x in self.image_transform.mean))
                images_list.append(self.image_transform(gv).to(self.dtype))
            nq = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tok = ([self.image_token_id] * nq + [self.image_token_id]) * nq + [self.image_token_id]
        return images_list, images_crop_list, spatial, tok

    def process_single_sample(self, messages):
        images = [self.deserialize_image(i) for m in messages if m.get("images") for i in m["images"] if i is not None]
        if not images: raise ValueError("No images found in sample.")
        ids, seqmask = [self.bos_id], [False]
        il, icl, isc = [], [], []
        prompt_count, started, k = -1, False, 0
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "<|Assistant|>":
                if not started:
                    prompt_count, started = len(ids), True
                content = f"{content.strip()} {self.tokenizer.eos_token}"
            parts = content.split("<image>")
            for i, p in enumerate(parts):
                t = text_encode(self.tokenizer, p, bos=False, eos=False)
                ids.extend(t); seqmask.extend([False] * len(t))
                if i < len(parts) - 1:
                    a, b, s, tok = self.process_image(images[k])
                    il.extend(a); icl.extend(b); isc.extend(s)
                    ids.extend(tok); seqmask.extend([True] * len(tok)); k += 1
        if not started: prompt_count = len(ids)
        crop = torch.stack(icl, 0) if icl else torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype)
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "images_seq_mask": torch.tensor(seqmask, dtype=torch.bool),
                "images_ori": torch.stack(il, 0), "images_crop": crop,
                "images_spatial_crop": torch.tensor(isc, dtype=torch.long), "prompt_token_count": prompt_count}

    def __call__(self, features):
        batch = []
        for f in features:
            try: batch.append(self.process_single_sample(f["messages"]))
            except Exception as e: print(f"Error processing sample: {e}")
        if not batch: raise ValueError("No valid samples in batch")
        pad = self.tokenizer.pad_token_id
        input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad)
        seqmask = pad_sequence([b["images_seq_mask"] for b in batch], batch_first=True, padding_value=False)
        labels = input_ids.clone()
        labels[labels == pad] = -100
        labels[seqmask] = -100
        if self.train_on_responses_only:
            for i, b in enumerate(batch):
                if b["prompt_token_count"] > 0: labels[i, :b["prompt_token_count"]] = -100
        return {"input_ids": input_ids, "attention_mask": (input_ids != pad).long(), "labels": labels,
                "images": [(b["images_crop"], b["images_ori"]) for b in batch], "images_seq_mask": seqmask,
                "images_spatial_crop": torch.cat([b["images_spatial_crop"] for b in batch], 0)}
