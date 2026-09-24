#!/usr/bin/env python3
"""Cityscapes 768x512 EF-LIC -> YOLOS -> routed-classifier E2E evaluation.

Shared input preprocessing (identical to the original 768x512 experiment):
    2048x1024 Cityscapes RGB -> center crop 1536x1024
    -> bicubic antialiased resize to 768x512.

Baseline:
    preprocessed RGB -> YOLOS -> bbox gate -> routed classifier.
Compressed:
    same preprocessed RGB -> EF-LIC compress/decompress -> YOLOS
    -> same bbox gate -> routed classifier.

For both branches, boxes and classifier crops are in 768x512 pixel coordinates.
The gate is applied independently and identically before matching.

For each valid route, Top-1 E2E counts missed detections and changed Top-1
labels; Top-3 E2E counts misses and the fractional difference between the
baseline and compressed *unordered* Top-3 sets. Combined errors/rates are
macro means across available routes; unweighted counts are diagnostics only.
This evaluates preservation relative to the uncompressed pipeline, NOT
accuracy against Cityscapes subtype ground-truth labels.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import (
    efficientnet_b3,
    resnet50,
    densenet121,
)
from torchvision.ops import nms, box_iou
from transformers import AutoModelForObjectDetection

from EF_LIC import model as eflic_model


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROUTE_CLASSES = ("car", "motorcycle", "bicycle")

QUALITY_TO_FORCE = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TARGET_WIDTH = 768
TARGET_HEIGHT = 512

SPATIAL_PREPROCESS = transforms.Compose([
    transforms.CenterCrop((1024, 1536)),
    transforms.Resize(
        (TARGET_HEIGHT, TARGET_WIDTH),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    ),
])


# ---------------------------------------------------------------------
# Dataset / shared reference preprocessing
# ---------------------------------------------------------------------

def list_cityscapes_val_images(root: Path) -> List[Path]:
    val_dir = root / "leftImg8bit" / "val"
    images = sorted(val_dir.rglob("*_leftImg8bit.png"))

    if not images:
        raise RuntimeError(
            f"No Cityscapes validation images found under {val_dir}"
        )

    return images


def load_reference_image(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load and spatially preprocess once, identically for both branches.

    2048x1024 -> center crop 1536x1024 -> bicubic resize 768x512.
    Returns uint8 RGB for EF-LIC and uint8 BGR for YOLOS/classifier.
    """
    with Image.open(path) as source:
        pil = source.convert("RGB")

    if pil.size != (2048, 1024):
        raise ValueError(
            f"Expected original 2048x1024 Cityscapes frame, got "
            f"{pil.size} for {path}. Do not pre-resize the source twice."
        )

    pil = SPATIAL_PREPROCESS(pil)
    rgb = np.asarray(pil, dtype=np.uint8)
    if rgb.shape != (TARGET_HEIGHT, TARGET_WIDTH, 3):
        raise RuntimeError(f"Unexpected preprocessed image shape: {rgb.shape}")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb, bgr


def rgb_uint8_to_eflic_tensor(
    rgb: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    x = torch.from_numpy(
        np.ascontiguousarray(rgb.transpose(2, 0, 1))
    ).float() / 255.0

    x = x * 2.0 - 1.0
    return x.unsqueeze(0).to(device, non_blocking=True)


def eflic_reconstruction_to_bgr(x_hat: torch.Tensor) -> np.ndarray:
    x = (
        ((x_hat[0].detach().float().cpu() + 1.0) * 0.5)
        .clamp(0.0, 1.0)
    )

    rgb = (
        x.permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------
# EF-LIC payload / BPP
# ---------------------------------------------------------------------

def pack_indices(network, indices) -> bytes:
    """
    Same fixed-width packing used in the previous EF-LIC benchmark.
    """
    n_e = tuple(int(x) for x in network.n_e)
    bit_widths = [(n - 1).bit_length() for n in n_e]

    items = [
        (indices["z_inds"], bit_widths[-1]),
        (indices["y_inds"][0], bit_widths[0]),
        (indices["y_inds"][1], bit_widths[1]),
        (indices["y_inds"][2], bit_widths[2]),
        (indices["y_inds"][3], bit_widths[3]),
    ]

    def tensor_to_bits(tensor, bits):
        values = (
            tensor.detach()
            .reshape(-1)
            .to("cpu", torch.long)
            .numpy()
            .astype(np.uint32, copy=False)
        )

        shifts = np.arange(
            bits - 1,
            -1,
            -1,
            dtype=np.uint32,
        )

        return (
            ((values[:, None] >> shifts) & 1)
            .astype(np.uint8)
            .reshape(-1)
        )

    raw_bits = np.concatenate([
        tensor_to_bits(tensor, bits)
        for tensor, bits in items
    ])

    return np.packbits(
        raw_bits,
        bitorder="big",
    ).tobytes()


def load_eflic(
    checkpoint_path: Path,
    device: torch.device,
):
    network = eflic_model().to(device).eval()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )

    network.load_state_dict(
        state_dict,
        strict=True,
    )

    return network


# ---------------------------------------------------------------------
# YOLOS
# ---------------------------------------------------------------------

def load_yolos(model_id: str, device: torch.device):
    return (
        AutoModelForObjectDetection
        .from_pretrained(model_id)
        .to(device)
        .eval()
    )


def route_label_ids(yolos) -> Dict[str, int]:
    id2label = {
        int(k): str(v).lower()
        for k, v in yolos.config.id2label.items()
    }

    name2id = {
        name: class_id
        for class_id, name in id2label.items()
    }

    missing = [
        name
        for name in ROUTE_CLASSES
        if name not in name2id
    ]

    if missing:
        raise RuntimeError(
            f"YOLOS is missing required labels: {missing}"
        )

    return {
        name: name2id[name]
        for name in ROUTE_CLASSES
    }


def yolos_preprocess(
    image_bgr: np.ndarray,
    target_short: int = 800,
    max_long: int = 1333,
) -> torch.Tensor:
    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    h, w = rgb.shape[:2]
    scale = target_short / min(h, w)

    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    if max(new_h, new_w) > max_long:
        scale = max_long / max(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

    rgb = cv2.resize(
        rgb,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    x = (
        torch.from_numpy(
            np.ascontiguousarray(
                rgb.transpose(2, 0, 1)
            )
        ).float()
        / 255.0
    )

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=x.dtype,
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        dtype=x.dtype,
    ).view(3, 1, 1)

    return ((x - mean) / std).unsqueeze(0)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]

    return torch.stack([
        cx - 0.5 * w,
        cy - 0.5 * h,
        cx + 0.5 * w,
        cy + 0.5 * h,
    ], dim=-1)


@torch.inference_mode()
def yolos_inference(
    preprocessed_tensor: torch.Tensor,
    yolos,
    device: torch.device,
):
    """
    YOLOS INFERENCE stage only.

    Input:
        preprocessed_tensor: CPU tensor [1,3,H,W] produced by yolos_preprocess()

    Output:
        raw HuggingFace YOLOS model output.
    """
    pixel_values = preprocessed_tensor.to(
        device,
        non_blocking=True,
    )

    return yolos(
        pixel_values=pixel_values
    )


def yolos_postprocess(
    model_output,
    original_image_bgr: np.ndarray,
    route_ids: Dict[str, int],
    threshold: float,
    nms_iou: float,
) -> List[Dict]:
    """
    YOLOS POSTPROCESSING stage.

    Raw YOLOS outputs
        -> softmax class probabilities
        -> remove no-object class
        -> confidence threshold
        -> keep car/motorcycle/bicycle only
        -> normalized cxcywh -> xyxy
        -> scale boxes to the 768x512 input image
        -> class-wise NMS
        -> routed detection dictionaries.
    """
    probs = torch.softmax(
        model_output.logits,
        dim=-1,
    )

    # Exclude the final "no object" class when selecting the label.
    scores, labels = probs[..., :-1].max(dim=-1)

    boxes = cxcywh_to_xyxy(
        model_output.pred_boxes
    )

    scores = scores[0]
    labels = labels[0]
    boxes = boxes[0]

    # Confidence threshold.
    keep = scores >= threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]

    # Keep only routed classes.
    if labels.numel():
        wanted = torch.tensor(
            list(route_ids.values()),
            dtype=labels.dtype,
            device=labels.device,
        )

        wanted_mask = (
            labels[:, None]
            == wanted[None, :]
        ).any(dim=1)

        scores = scores[wanted_mask]
        labels = labels[wanted_mask]
        boxes = boxes[wanted_mask]

    # YOLOS predicts normalized boxes. Convert them back to pixel
    # coordinates of the preprocessed 768x512 image supplied to YOLOS.
    h, w = original_image_bgr.shape[:2]

    if boxes.numel():
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h

    # Class-wise NMS.
    keep_parts = []

    for class_id in route_ids.values():
        idx = torch.where(
            labels == class_id
        )[0]

        if idx.numel():
            local_keep = nms(
                boxes[idx],
                scores[idx],
                nms_iou,
            )
            keep_parts.append(
                idx[local_keep]
            )

    if not keep_parts:
        return []

    final_keep = torch.cat(keep_parts)

    # Highest confidence first for deterministic ordering.
    final_keep = final_keep[
        torch.argsort(
            scores[final_keep],
            descending=True,
        )
    ]

    scores = scores[final_keep]
    labels = labels[final_keep]
    boxes = boxes[final_keep]

    id_to_name = {
        class_id: name
        for name, class_id in route_ids.items()
    }

    detections = []

    for score, label, box in zip(
        scores.detach().cpu(),
        labels.detach().cpu(),
        boxes.detach().cpu(),
    ):
        x1, y1, x2, y2 = box.tolist()

        x1 = max(0.0, min(float(x1), float(w)))
        x2 = max(0.0, min(float(x2), float(w)))
        y1 = max(0.0, min(float(y1), float(h)))
        y2 = max(0.0, min(float(y2), float(h)))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "route": id_to_name[int(label.item())],
            "score": float(score.item()),
            "bbox": [x1, y1, x2, y2],
        })

    return detections


@torch.inference_mode()
def detect_targets(
    image_bgr: np.ndarray,
    yolos,
    route_ids: Dict[str, int],
    device: torch.device,
    threshold: float,
    nms_iou: float,
) -> List[Dict]:
    """
    Explicit YOLOS three-stage pipeline:

        1. PREPROCESSING
           preprocessed 768x512 BGR uint8
             -> RGB
             -> detector resize
             -> float tensor [0,1]
             -> ImageNet normalization

        2. INFERENCE
           preprocessed tensor
             -> YOLOS raw logits + normalized boxes

        3. POSTPROCESSING
           logits/boxes
             -> softmax
             -> confidence filter
             -> target-class routing
             -> pixel-coordinate boxes
             -> class-wise NMS
    """
    # -----------------------------
    # YOLOS PREPROCESSING
    # -----------------------------
    y_cpu = yolos_preprocess(
        image_bgr
    )

    # -----------------------------
    # YOLOS INFERENCE
    # -----------------------------
    model_output = yolos_inference(
        y_cpu,
        yolos,
        device,
    )

    # -----------------------------
    # YOLOS POSTPROCESSING
    # -----------------------------
    detections = yolos_postprocess(
        model_output,
        image_bgr,
        route_ids,
        threshold,
        nms_iou,
    )

    return detections


# ---------------------------------------------------------------------
# Detection matching
# ---------------------------------------------------------------------

def match_detections(
    baseline: List[Dict],
    compressed: List[Dict],
    iou_threshold: float,
) -> Tuple[Dict[int, Tuple[int, float]], List[int]]:
    """
    Greedy matching by IoU, restricted to same YOLOS route class.

    Returns:
      matches:
         baseline_index -> (compressed_index, IoU)
      unmatched_compressed_indices
    """
    matches = {}
    used_compressed = set()

    candidates = []

    for bi, b in enumerate(baseline):
        for ci, c in enumerate(compressed):
            if b["route"] != c["route"]:
                continue

            b_box = torch.tensor(
                [b["bbox"]],
                dtype=torch.float32,
            )
            c_box = torch.tensor(
                [c["bbox"]],
                dtype=torch.float32,
            )

            iou = float(
                box_iou(b_box, c_box)[0, 0].item()
            )

            if iou >= iou_threshold:
                candidates.append(
                    (iou, bi, ci)
                )

    # Highest-IoU pair gets first claim.
    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    used_baseline = set()

    for iou, bi, ci in candidates:
        if bi in used_baseline:
            continue
        if ci in used_compressed:
            continue

        matches[bi] = (ci, iou)
        used_baseline.add(bi)
        used_compressed.add(ci)

    unmatched_compressed = [
        ci
        for ci in range(len(compressed))
        if ci not in used_compressed
    ]

    return matches, unmatched_compressed


# ---------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------

class PadToSquare:
    def __init__(self, fill):
        self.fill = fill

    def __call__(self, image):
        w, h = image.size
        side = max(w, h)

        left = (side - w) // 2
        right = side - w - left
        top = (side - h) // 2
        bottom = side - h - top

        return ImageOps.expand(
            image,
            border=(left, top, right, bottom),
            fill=self.fill,
        )


def checkpoint_classes(ckpt: Dict) -> List[str]:
    classes = ckpt.get(
        "class_names",
        ckpt.get("classes"),
    )

    if classes is None:
        raise KeyError(
            "Checkpoint has neither 'class_names' nor 'classes'."
        )

    return list(classes)


def load_car_classifier(
    checkpoint_path: Path,
    device: torch.device,
):
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    classes = checkpoint_classes(ckpt)

    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes),
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    model = model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize(
            320,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),
    ])

    return model, classes, preprocess


def load_motorcycle_classifier(
    checkpoint_path: Path,
    device: torch.device,
):
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    classes = checkpoint_classes(ckpt)

    model = resnet50(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(classes),
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    model = model.to(device).eval()

    # Matches the ResNet-50 motorcycle training eval transform.
    preprocess = transforms.Compose([
        PadToSquare(fill=(0, 0, 0)),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),
    ])

    return model, classes, preprocess


def load_bicycle_classifier(
    checkpoint_path: Path,
    device: torch.device,
):
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    classes = checkpoint_classes(ckpt)

    model = densenet121(weights=None)
    model.classifier = nn.Linear(
        model.classifier.in_features,
        len(classes),
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    model = model.to(device).eval()

    # Matches the DenseNet-121 bicycle training eval transform.
    preprocess = transforms.Compose([
        PadToSquare(fill=(255, 255, 255)),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),
    ])

    return model, classes, preprocess


def bgr_crop_to_pil(
    image_bgr: np.ndarray,
    bbox: List[float],
) -> Optional[Image.Image]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(round(x1)), w))
    x2 = max(0, min(int(round(x2)), w))
    y1 = max(0, min(int(round(y1)), h))
    y2 = max(0, min(int(round(y2)), h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop_bgr = image_bgr[y1:y2, x1:x2]

    if crop_bgr.size == 0:
        return None

    crop_rgb = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(crop_rgb)


def classifier_preprocess(
    image_bgr: np.ndarray,
    detection: Dict,
    classifier_spec: Dict,
) -> Optional[torch.Tensor]:
    """
    CLASSIFIER PREPROCESSING stage.

    YOLOS pixel-space detection box
        -> crop from the SAME image that YOLOS saw
        -> BGR -> RGB/PIL
        -> classifier-specific evaluation transform
        -> batch tensor [1,3,H,W].

    Classifier-specific transforms:
      car:
        Resize(320) -> CenterCrop(300) -> ImageNet normalize
      motorcycle:
        PadToSquare(black) -> Resize(224x224) -> ImageNet normalize
      bicycle:
        PadToSquare(white) -> Resize(224x224) -> ImageNet normalize
    """
    pil = bgr_crop_to_pil(
        image_bgr,
        detection["bbox"],
    )

    if pil is None:
        return None

    preprocess = classifier_spec[
        "preprocess"
    ]

    return preprocess(
        pil
    ).unsqueeze(0)


@torch.inference_mode()
def classifier_inference(
    classifier_input: torch.Tensor,
    classifier_spec: Dict,
    device: torch.device,
) -> torch.Tensor:
    """
    CLASSIFIER INFERENCE stage only.

    Preprocessed crop tensor
        -> model
        -> raw class logits.
    """
    model = classifier_spec["model"]

    x = classifier_input.to(
        device,
        non_blocking=True,
    )

    return model(x)


def classifier_postprocess(
    logits: torch.Tensor,
    classifier_spec: Dict,
) -> Dict:
    """
    CLASSIFIER POSTPROCESSING stage.

    Raw logits
        -> softmax probabilities
        -> Top-1 class
        -> Top-3 classes
        -> full probability vector.

    Top-3 is stored in descending confidence order, but the Top-3
    comparison metric treats the three labels as an unordered set.
    """
    classes = classifier_spec[
        "classes"
    ]

    probs_tensor = torch.softmax(
        logits,
        dim=1,
    )[0]

    probs = (
        probs_tensor
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    top_k = min(
        3,
        len(classes),
    )

    top_probs_tensor, top_indices_tensor = torch.topk(
        probs_tensor,
        k=top_k,
        largest=True,
        sorted=True,
    )

    top_indices = (
        top_indices_tensor
        .detach()
        .cpu()
        .tolist()
    )

    top_probs = (
        top_probs_tensor
        .detach()
        .float()
        .cpu()
        .tolist()
    )

    top3_classes = [
        classes[int(i)]
        for i in top_indices
    ]

    return {
        "top1_index":
            int(top_indices[0]),
        "top1_class":
            top3_classes[0],
        "top1_confidence":
            float(top_probs[0]),
        "top3_indices":
            [int(i) for i in top_indices],
        "top3_classes":
            top3_classes,
        "top3_confidences":
            [float(p) for p in top_probs],
        "probabilities":
            probs,
    }


@torch.inference_mode()
def classify_detection(
    image_bgr: np.ndarray,
    detection: Dict,
    classifier_spec: Dict,
    device: torch.device,
) -> Optional[Dict]:
    """
    Explicit classifier three-stage pipeline:

        1. PREPROCESSING
           detection box -> crop -> classifier-specific transform

        2. INFERENCE
           preprocessed crop -> classifier logits

        3. POSTPROCESSING
           logits -> softmax -> Top-1 + Top-3 predictions
    """
    # -----------------------------
    # CLASSIFIER PREPROCESSING
    # -----------------------------
    classifier_input = (
        classifier_preprocess(
            image_bgr,
            detection,
            classifier_spec,
        )
    )

    if classifier_input is None:
        return None

    # -----------------------------
    # CLASSIFIER INFERENCE
    # -----------------------------
    logits = classifier_inference(
        classifier_input,
        classifier_spec,
        device,
    )

    # -----------------------------
    # CLASSIFIER POSTPROCESSING
    # -----------------------------
    return classifier_postprocess(
        logits,
        classifier_spec,
    )


def classifier_probability_nmae(
    baseline_probs: np.ndarray,
    compressed_probs: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    NMAE = sum |P - P_hat| / (sum |P| + eps)

    For a softmax probability vector, sum(P)=1, so this is numerically
    the L1 distance between the two probability vectors.
    """
    numerator = float(
        np.abs(
            baseline_probs - compressed_probs
        ).sum()
    )
    denominator = float(
        np.abs(baseline_probs).sum()
    )

    return numerator / (
        denominator + eps
    )


# ---------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------

def route_counts(
    detections: List[Dict],
) -> Dict[str, int]:
    out = {
        route: 0
        for route in ROUTE_CLASSES
    }

    for d in detections:
        out[d["route"]] += 1

    return out


def safe_percent(
    numerator: float,
    denominator: float,
) -> Optional[float]:
    if denominator <= 0:
        return None

    return (
        100.0
        * float(numerator)
        / float(denominator)
    )


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def fmt_optional(x, digits=3):
    if x is None:
        return "N/A"
    return round(float(x), digits)


def bbox_to_string(bbox):
    if bbox is None:
        return ""

    return ",".join(
        f"{float(x):.2f}"
        for x in bbox
    )



def bbox_size(
    bbox,
) -> Tuple[float, float]:
    """
    Return bounding-box width and height in pixels.
    """
    x1, y1, x2, y2 = [
        float(v)
        for v in bbox
    ]

    return (
        max(0.0, x2 - x1),
        max(0.0, y2 - y1),
    )


def passes_bbox_size_gate(
    detection: Dict,
    min_box_side: float,
) -> bool:
    """
    Keep a detection only when BOTH box dimensions are at least min_box_side.

    On preprocessed Cityscapes 768x512, the default 48 px threshold avoids
    sending extremely small / heavily-upsampled crops to the downstream
    classifiers.
    """
    width, height = bbox_size(
        detection["bbox"]
    )

    return (
        width >= min_box_side
        and height >= min_box_side
    )


def apply_bbox_size_gate(
    detections: List[Dict],
    min_box_side: float,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split YOLOS detections into:
      eligible: sent to the classifier / object matching
      excluded: too small for this experiment
    """
    eligible = []
    excluded = []

    for det in detections:
        target = (
            eligible
            if passes_bbox_size_gate(
                det,
                min_box_side,
            )
            else excluded
        )

        target.append(det)

    return eligible, excluded



def save_csv(
    path: Path,
    rows: List[Dict],
):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Union of all fields, preserving first-seen order.
    fields = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def checkpoint_identity(checkpoint_path: Path) -> Dict:
    """Return metadata proving which trained checkpoint is being used."""
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    return {
        "path": str(checkpoint_path),
        "architecture": ckpt.get("architecture"),
        "task": ckpt.get("task"),
        "stage": ckpt.get("stage"),
        "epoch": ckpt.get("epoch"),
        "classes": ckpt.get(
            "class_names",
            ckpt.get("classes"),
        ),
    }



def main():
    parser = argparse.ArgumentParser(
        description=(
            "768x512 preprocessed object-matched end-to-end error: "
            "EF-LIC -> YOLOS -> routed trained classifiers."
        )
    )

    parser.add_argument(
        "--cityscapes-root",
        type=Path,
        default=Path(
            "/home/common/EF-LIC/datasets/cityscapes"
        ),
    )

    parser.add_argument(
        "--eflic-ckpt",
        type=Path,
        default=Path(
            "/home/common/EF-LIC/ckpt/checkpoint.pth.tar"
        ),
    )

    parser.add_argument(
        "--car-ckpt",
        type=Path,
        default=Path(
            "/home/common/EF-LIC/"
            "checkpoints_effb3_compcars_types/"
            "efficientnet_b3_compcars_types_best.pth"
        ),
    )

    parser.add_argument(
        "--motorcycle-ckpt",
        type=Path,
        default=Path(
            "/home/common/EF-LIC/"
            "checkpoints_resnet50_motorcycle_types_full/"
            "resnet50_motorcycle_types_best.pth"
        ),
    )

    parser.add_argument(
        "--bicycle-ckpt",
        type=Path,
        default=Path(
            "/home/common/EF-LIC/"
            "checkpoints_densenet121_biked_6class/"
            "densenet121_biked_6class_best.pth"
        ),
    )

    parser.add_argument(
        "--yolos-model-id",
        default="hustvl/yolos-small",
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
        help=(
            "Minimum IoU for matching the same object between the "
            "uncompressed and compressed YOLOS outputs."
        ),
    )

    parser.add_argument(
        "--min-box-side",
        type=float,
        default=48.0,
        help=(
            "Minimum YOLOS bounding-box width AND height, in pixels, "
            "required before a detection is sent to a subtype classifier. "
            "Default: 48 pixels on 768x512 frames (equivalent to 96 px before 2x downscaling for objects inside the center crop)."
        ),
    )

    parser.add_argument(
        "--qualities",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test image limit.",
    )

    parser.add_argument(
        "--per-object-csv",
        type=Path,
        default=Path(
            "results_eflic_e2e_top1_top3_768x512_bboxgate_macro_per_object.csv"
        ),
    )

    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(
            "results_eflic_e2e_top1_top3_768x512_bboxgate_macro_summary.csv"
        ),
    )

    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "results_eflic_e2e_top1_top3_768x512_bboxgate_macro_summary.json"
        ),
    )

    args = parser.parse_args()

    if not np.isfinite(args.min_box_side) or args.min_box_side <= 0:
        parser.error("--min-box-side must be a positive finite pixel value.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive image count.")

    # Retain user-selected paths, but distinguish the defaults by threshold
    # and image subset (e.g. ...bbox48_n10...) to prevent silent overwrites.
    suffix = f"bbox{args.min_box_side:g}"
    if args.limit is not None:
        suffix += f"_n{args.limit}"
    for output_arg in ("per_object_csv", "summary_csv", "out_json"):
        path = getattr(args, output_arg)
        if "_bboxgate_macro_" in path.name:
            setattr(args, output_arg, Path(
                str(path).replace("_bboxgate_macro_", f"_{suffix}_macro_", 1)
            ))


    for quality in args.qualities:
        if quality not in QUALITY_TO_FORCE:
            parser.error(
                f"Unsupported quality Q{quality}. "
                "Valid values are 1 2 3 4 5."
            )

    device = torch.device(args.device)

    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is required for this experiment."
        )

    torch.backends.cudnn.benchmark = True

    images = list_cityscapes_val_images(
        args.cityscapes_root
    )

    if args.limit is not None:
        images = images[:args.limit]

    print("=" * 80)
    print("768x512 BBOX-GATED END-TO-END TOP-1 + TOP-3 ERROR")
    print("=" * 80)
    print(f"Device              : {device}")
    print(f"Images              : {len(images)}")
    print(
        "Input               : 2048x1024 -> center crop 1536x1024 "
        "-> bicubic resize 768x512"
    )
    print(
        "Baseline            : 768x512 -> YOLOS -> gate -> trained classifier"
    )
    print(
        "Compressed          : 768x512 -> EF-LIC -> reconstruction "
        "-> YOLOS -> gate -> trained classifier"
    )
    print(
        "YOLOS stages        : preprocess -> inference -> postprocess"
    )
    print(
        "Classifier stages   : preprocess -> inference -> postprocess"
    )
    print(f"YOLOS threshold     : {args.threshold}")
    print(f"YOLOS NMS IoU       : {args.nms_iou}")
    print(f"Object match IoU    : {args.match_iou}")
    print("BBox coordinates    : 768x512 preprocessed pixels")
    print("BPP denominator     : 768 x 512 pixels")
    print(
        f"Minimum bbox side   : {args.min_box_side:.1f} px "
        "(both width and height)"
    )

    print("\nMetrics:")
    print(
        "  The SAME bbox-size gate is applied independently to both "
        "uncompressed and compressed YOLOS detections."
    )
    print(
        "  Baseline denominator = uncompressed YOLOS detections that PASS "
        "the bbox-size gate."
    )
    print(
        "  Top-1 E2E error = "
        "(missed eligible baseline objects + matched Top-1 changes) "
        "/ eligible baseline objects."
    )
    print(
        "  Top-3 compares the BASELINE Top-3 set with the COMPRESSED "
        "Top-3 set for each matched object."
    )
    print(
        "  Matched Top-3 overlap error = "
        "1 - |baseline Top-3 intersection compressed Top-3| / 3."
    )
    print(
        "  Top-3 E2E overlap error = "
        "(missed objects + summed matched Top-3 overlap error) "
        "/ eligible baseline objects."
    )
    print(
        "  A strict Top-3 exact-set change rate is also reported as a "
        "diagnostic; rank order inside Top-3 is ignored."
    )
    print(
        "  Every combined error/rate is a MACRO average across valid "
        "car, motorcycle, and bicycle route rates."
    )
    print(
        "  No micro/object-count-weighted ALL error is reported."
    )

    # -------------------------------------------------------------
    # Verify checkpoints
    # -------------------------------------------------------------

    print("\nCheckpoint verification:")

    for label, ckpt_path in (
        ("CAR", args.car_ckpt),
        ("MOTORCYCLE", args.motorcycle_ckpt),
        ("BICYCLE", args.bicycle_ckpt),
    ):
        info = checkpoint_identity(
            ckpt_path
        )

        print(
            f"  {label:10s}: "
            f"architecture={info['architecture']} | "
            f"stage={info['stage']} | "
            f"epoch={info['epoch']}"
        )
        print(
            f"              {info['path']}"
        )

    # -------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------

    print("\nLoading YOLOS...")
    yolos = load_yolos(
        args.yolos_model_id,
        device,
    )

    route_ids = route_label_ids(
        yolos
    )

    print("YOLOS route IDs:", route_ids)

    print("\nLoading EF-LIC...")
    eflic = load_eflic(
        args.eflic_ckpt,
        device,
    )

    print("\nLoading car classifier...")
    car_model, car_classes, car_pre = (
        load_car_classifier(
            args.car_ckpt,
            device,
        )
    )

    print("\nLoading motorcycle classifier...")
    moto_model, moto_classes, moto_pre = (
        load_motorcycle_classifier(
            args.motorcycle_ckpt,
            device,
        )
    )

    print("\nLoading bicycle classifier...")
    bike_model, bike_classes, bike_pre = (
        load_bicycle_classifier(
            args.bicycle_ckpt,
            device,
        )
    )

    classifiers = {
        "car": {
            "model": car_model,
            "classes": car_classes,
            "preprocess": car_pre,
        },
        "motorcycle": {
            "model": moto_model,
            "classes": moto_classes,
            "preprocess": moto_pre,
        },
        "bicycle": {
            "model": bike_model,
            "classes": bike_classes,
            "preprocess": bike_pre,
        },
    }

    print("\nFinal subtype labels:")

    for route in ROUTE_CLASSES:
        print(
            f"  {route:10s}: "
            + ", ".join(
                classifiers[route]["classes"]
            )
        )

    # -------------------------------------------------------------
    # Baseline:
    # preprocessed 768x512 -> YOLOS -> bbox gate -> classifier
    # -------------------------------------------------------------

    print(
        "\nComputing uncompressed baseline objects and Top-1 labels..."
    )

    baseline_cache = {}

    baseline_raw_route_totals = {
        route: 0
        for route in ROUTE_CLASSES
    }

    baseline_route_totals = {
        route: 0
        for route in ROUTE_CLASSES
    }

    baseline_excluded_route_totals = {
        route: 0
        for route in ROUTE_CLASSES
    }

    for image_index, path in enumerate(
        images,
        start=1,
    ):
        _, reference_bgr = (
            load_reference_image(path)
        )

        raw_detections = detect_targets(
            reference_bgr,
            yolos,
            route_ids,
            device,
            args.threshold,
            args.nms_iou,
        )

        detections, excluded_detections = (
            apply_bbox_size_gate(
                raw_detections,
                args.min_box_side,
            )
        )

        predictions = []

        for det in detections:
            route = det["route"]

            pred = classify_detection(
                reference_bgr,
                det,
                classifiers[route],
                device,
            )

            if pred is None:
                raise RuntimeError(
                    "Baseline classifier crop failed for "
                    f"{path}, route={route}, bbox={det['bbox']}"
                )

            predictions.append(
                pred
            )

        raw_counts = route_counts(
            raw_detections
        )
        eligible_counts = route_counts(
            detections
        )
        excluded_counts = route_counts(
            excluded_detections
        )

        for route in ROUTE_CLASSES:
            baseline_raw_route_totals[
                route
            ] += raw_counts[route]

            baseline_route_totals[
                route
            ] += eligible_counts[route]

            baseline_excluded_route_totals[
                route
            ] += excluded_counts[route]

        baseline_cache[
            str(path)
        ] = {
            "detections": detections,
            "predictions": predictions,
            "raw_counts": raw_counts,
            "eligible_counts": eligible_counts,
            "excluded_counts": excluded_counts,
        }

        if (
            image_index == 1
            or image_index % 25 == 0
            or image_index == len(images)
        ):
            print(
                f"[baseline "
                f"{image_index:03d}/{len(images):03d}] "
                f"raw C/M/B="
                f"{raw_counts['car']}/"
                f"{raw_counts['motorcycle']}/"
                f"{raw_counts['bicycle']} | "
                f"eligible="
                f"{eligible_counts['car']}/"
                f"{eligible_counts['motorcycle']}/"
                f"{eligible_counts['bicycle']} | "
                f"excluded="
                f"{excluded_counts['car']}/"
                f"{excluded_counts['motorcycle']}/"
                f"{excluded_counts['bicycle']}"
            )

    total_raw_baseline_objects = sum(
        baseline_raw_route_totals.values()
    )

    total_baseline_objects = sum(
        baseline_route_totals.values()
    )

    total_excluded_baseline_objects = sum(
        baseline_excluded_route_totals.values()
    )

    print("\nBaseline bbox-gate coverage:")

    for route in ROUTE_CLASSES:
        raw_n = baseline_raw_route_totals[
            route
        ]
        eligible_n = baseline_route_totals[
            route
        ]
        excluded_n = baseline_excluded_route_totals[
            route
        ]

        coverage = safe_percent(
            eligible_n,
            raw_n,
        )

        print(
            f"  {route:10s}: "
            f"raw={raw_n} | "
            f"eligible={eligible_n} | "
            f"excluded={excluded_n} | "
            f"coverage={fmt_optional(coverage, 2)}%"
        )

    overall_coverage = safe_percent(
        total_baseline_objects,
        total_raw_baseline_objects,
    )

    print(
        f"  TOTAL      : raw={total_raw_baseline_objects} | "
        f"eligible={total_baseline_objects} | "
        f"excluded={total_excluded_baseline_objects} | "
        f"coverage={fmt_optional(overall_coverage, 2)}%"
    )

    # -------------------------------------------------------------
    # Compressed qualities
    # -------------------------------------------------------------

    per_object_rows = []
    summary_rows = []
    json_summary = {}

    for quality in args.qualities:
        force_ind = QUALITY_TO_FORCE[
            quality
        ]

        print("\n" + "=" * 80)
        print(
            f"Q{quality} / EF-LIC force_ind={force_ind}"
        )
        print("=" * 80)

        eflic.prepare_inference_(
            force_ind=force_ind
        )

        bpp_values = []

        stats = {
            route: {
                "baseline_raw_objects":
                    baseline_raw_route_totals[route],
                "baseline_objects":
                    baseline_route_totals[route],
                "baseline_excluded_objects":
                    baseline_excluded_route_totals[route],
                "compressed_raw_detections": 0,
                "compressed_detections": 0,
                "compressed_excluded_detections": 0,
                "matched_objects": 0,
                "missed_objects": 0,
                "same_top1_objects": 0,
                "top1_changed_objects": 0,
                "top3_exact_same_objects": 0,
                "top3_exact_changed_objects": 0,
                "top3_overlap_count_sum": 0,
                "top3_overlap_error_sum": 0.0,
                "extra_compressed_objects": 0,
                "yolos_count_abs_error": 0,
                "match_ious": [],
            }
            for route in ROUTE_CLASSES
        }

        for image_index, path in enumerate(
            images,
            start=1,
        ):
            reference_rgb, _ = (
                load_reference_image(path)
            )

            x = rgb_uint8_to_eflic_tensor(
                reference_rgb,
                device,
            )

            # -------------------------
            # EF-LIC inference
            # -------------------------
            with torch.inference_mode():
                indices = eflic.compress(
                    x,
                    force_ind=force_ind,
                )

                payload = pack_indices(
                    eflic,
                    indices,
                )

                x_hat = eflic.decompress(
                    indices,
                    force_ind=force_ind,
                )

            reconstructed_bgr = (
                eflic_reconstruction_to_bgr(
                    x_hat
                )
            )

            frame_height, frame_width = (
                reference_rgb.shape[:2]
            )
            if (frame_width, frame_height) != (TARGET_WIDTH, TARGET_HEIGHT):
                raise RuntimeError("EF-LIC received a non-768x512 source frame")
            if reconstructed_bgr.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
                raise RuntimeError(
                    f"EF-LIC reconstruction shape {reconstructed_bgr.shape[:2]} "
                    "does not match 768x512 reference"
                )

            bpp = (
                len(payload)
                * 8.0
                / float(
                    frame_height
                    * frame_width
                )
            )

            bpp_values.append(
                bpp
            )

            # -------------------------
            # Compressed YOLOS path
            # -------------------------
            compressed_raw_detections = (
                detect_targets(
                    reconstructed_bgr,
                    yolos,
                    route_ids,
                    device,
                    args.threshold,
                    args.nms_iou,
                )
            )

            (
                compressed_detections,
                compressed_excluded_detections,
            ) = apply_bbox_size_gate(
                compressed_raw_detections,
                args.min_box_side,
            )

            baseline_entry = (
                baseline_cache[
                    str(path)
                ]
            )

            baseline_detections = (
                baseline_entry[
                    "detections"
                ]
            )

            baseline_predictions = (
                baseline_entry[
                    "predictions"
                ]
            )

            # Same-route greedy highest-IoU matching.
            matches, unmatched_compressed = (
                match_detections(
                    baseline_detections,
                    compressed_detections,
                    args.match_iou,
                )
            )

            baseline_counts = route_counts(
                baseline_detections
            )

            compressed_raw_counts = route_counts(
                compressed_raw_detections
            )

            compressed_counts = route_counts(
                compressed_detections
            )

            compressed_excluded_counts = route_counts(
                compressed_excluded_detections
            )

            # Route-level diagnostics for the bbox-gated pipeline.
            for route in ROUTE_CLASSES:
                stats[route][
                    "compressed_raw_detections"
                ] += compressed_raw_counts[route]

                stats[route][
                    "compressed_detections"
                ] += compressed_counts[route]

                stats[route][
                    "compressed_excluded_detections"
                ] += compressed_excluded_counts[route]

                # Count NMAE for the ELIGIBLE detector outputs only.
                stats[route][
                    "yolos_count_abs_error"
                ] += abs(
                    compressed_counts[route]
                    - baseline_counts[route]
                )

            # Extra compressed detections are NOT baseline objects.
            for ci in unmatched_compressed:
                route = (
                    compressed_detections[
                        ci
                    ]["route"]
                )

                stats[route][
                    "extra_compressed_objects"
                ] += 1

            # One row / one outcome for every eligible baseline object.
            #
            # Top-1:
            #   baseline Top-1 == compressed Top-1 -> stable
            #
            # Top-3:
            #   compare baseline Top-3 SET vs compressed Top-3 SET.
            #   overlap_count = |B_top3 ∩ C_top3|
            #   overlap_fraction = overlap_count / 3
            #   overlap_error = 1 - overlap_fraction
            #
            # A missed detection contributes 100% error to BOTH E2E metrics.
            for bi, base_det in enumerate(
                baseline_detections
            ):
                route = base_det["route"]
                base_pred = (
                    baseline_predictions[bi]
                )

                base_top3 = list(
                    base_pred["top3_classes"]
                )

                row = {
                    "image": str(path),
                    "quality": f"Q{quality}",
                    "force_ind": force_ind,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "bpp": bpp,
                    "route": route,
                    "baseline_detection_score":
                        base_det["score"],
                    "baseline_bbox":
                        bbox_to_string(
                            base_det["bbox"]
                        ),
                    "baseline_bbox_width":
                        bbox_size(base_det["bbox"])[0],
                    "baseline_bbox_height":
                        bbox_size(base_det["bbox"])[1],
                    "min_box_side_threshold":
                        args.min_box_side,
                    "baseline_top1":
                        base_pred["top1_class"],
                    "baseline_top3":
                        "|".join(base_top3),
                    "matched": 0,
                    "match_iou": "",
                    "compressed_detection_score": "",
                    "compressed_bbox": "",
                    "compressed_bbox_width": "",
                    "compressed_bbox_height": "",
                    "compressed_top1": "",
                    "compressed_top3": "",
                    "top1_same": 0,
                    "top1_error_contribution": 1.0,
                    "top3_overlap_count": 0,
                    "top3_overlap_fraction": 0.0,
                    "top3_overlap_error_contribution": 1.0,
                    "top3_exact_same": 0,
                    "outcome": "missed_detection",
                }

                if bi not in matches:
                    stats[route][
                        "missed_objects"
                    ] += 1

                    per_object_rows.append(
                        row
                    )

                    continue

                ci, match_iou = (
                    matches[bi]
                )

                comp_det = (
                    compressed_detections[
                        ci
                    ]
                )

                comp_pred = classify_detection(
                    reconstructed_bgr,
                    comp_det,
                    classifiers[route],
                    device,
                )

                if comp_pred is None:
                    raise RuntimeError(
                        "Compressed classifier crop failed for "
                        f"{path}, route={route}, bbox={comp_det['bbox']}"
                    )

                stats[route][
                    "matched_objects"
                ] += 1

                stats[route][
                    "match_ious"
                ].append(
                    match_iou
                )

                # -------------------------
                # Top-1 comparison
                # -------------------------
                top1_same = int(
                    base_pred["top1_class"]
                    == comp_pred["top1_class"]
                )

                if top1_same:
                    stats[route][
                        "same_top1_objects"
                    ] += 1
                else:
                    stats[route][
                        "top1_changed_objects"
                    ] += 1

                # -------------------------
                # Top-3 SET comparison
                # -------------------------
                comp_top3 = list(
                    comp_pred["top3_classes"]
                )

                base_top3_set = set(
                    base_top3
                )

                comp_top3_set = set(
                    comp_top3
                )

                top3_overlap_count = len(
                    base_top3_set
                    & comp_top3_set
                )

                # All routed classifiers have >= 3 classes, so k=3.
                top3_overlap_fraction = (
                    top3_overlap_count
                    / 3.0
                )

                top3_overlap_error = (
                    1.0
                    - top3_overlap_fraction
                )

                top3_exact_same = int(
                    base_top3_set
                    == comp_top3_set
                )

                stats[route][
                    "top3_overlap_count_sum"
                ] += top3_overlap_count

                stats[route][
                    "top3_overlap_error_sum"
                ] += top3_overlap_error

                if top3_exact_same:
                    stats[route][
                        "top3_exact_same_objects"
                    ] += 1
                else:
                    stats[route][
                        "top3_exact_changed_objects"
                    ] += 1

                if top1_same:
                    top1_outcome = "top1_same"
                else:
                    top1_outcome = "top1_changed"

                if top3_exact_same:
                    top3_outcome = "top3_exact_same"
                else:
                    top3_outcome = (
                        f"top3_overlap_{top3_overlap_count}_of_3"
                    )

                row.update({
                    "matched": 1,
                    "match_iou":
                        match_iou,
                    "compressed_detection_score":
                        comp_det["score"],
                    "compressed_bbox":
                        bbox_to_string(
                            comp_det["bbox"]
                        ),
                    "compressed_bbox_width":
                        bbox_size(comp_det["bbox"])[0],
                    "compressed_bbox_height":
                        bbox_size(comp_det["bbox"])[1],
                    "compressed_top1":
                        comp_pred["top1_class"],
                    "compressed_top3":
                        "|".join(comp_top3),
                    "top1_same":
                        top1_same,
                    "top1_error_contribution":
                        float(1 - top1_same),
                    "top3_overlap_count":
                        top3_overlap_count,
                    "top3_overlap_fraction":
                        top3_overlap_fraction,
                    "top3_overlap_error_contribution":
                        top3_overlap_error,
                    "top3_exact_same":
                        top3_exact_same,
                    "outcome":
                        f"{top1_outcome};{top3_outcome}",
                })

                per_object_rows.append(
                    row
                )

            if (
                image_index == 1
                or image_index % 25 == 0
                or image_index == len(images)
            ):
                print(
                    f"[Q{quality} "
                    f"{image_index:03d}/{len(images):03d}] "
                    f"BPP={bpp:.5f} | "
                    f"baseline_eligible={len(baseline_detections)} "
                    f"compressed_raw={len(compressed_raw_detections)} "
                    f"compressed_eligible={len(compressed_detections)} "
                    f"matched={len(matches)} "
                    f"missed={len(baseline_detections)-len(matches)} "
                    f"extra={len(unmatched_compressed)}"
                )

        mean_bpp = safe_mean(
            bpp_values
        )

        quality_json = {}

        # ---------------------------------------------------------
        # Per-route summaries
        # ---------------------------------------------------------
        route_rows = {}

        for route in ROUTE_CLASSES:
            s = stats[route]

            baseline_objects = (
                s["baseline_objects"]
            )

            matched_objects = (
                s["matched_objects"]
            )

            missed_objects = (
                s["missed_objects"]
            )

            top1_changed_objects = (
                s["top1_changed_objects"]
            )

            same_top1_objects = (
                s["same_top1_objects"]
            )

            # -------------------------
            # Top-1 metrics
            # -------------------------
            top1_e2e_error_rate = safe_percent(
                missed_objects
                + top1_changed_objects,
                baseline_objects,
            )

            top1_e2e_retention = safe_percent(
                same_top1_objects,
                baseline_objects,
            )

            detection_loss_rate = safe_percent(
                missed_objects,
                baseline_objects,
            )

            top1_change_on_matched = safe_percent(
                top1_changed_objects,
                matched_objects,
            )

            top1_agreement_on_matched = safe_percent(
                same_top1_objects,
                matched_objects,
            )

            # -------------------------
            # Top-3 set-overlap metrics
            # -------------------------
            #
            # Each matched object contributes:
            #   0.0 error if 3/3 labels overlap
            #   1/3 error if 2/3 overlap
            #   2/3 error if 1/3 overlap
            #   1.0 error if 0/3 overlap
            #
            # Each missed baseline object contributes 1.0 error.
            top3_overlap_error_on_matched = (
                None
                if matched_objects <= 0
                else (
                    s["top3_overlap_error_sum"]
                    / matched_objects
                    * 100.0
                )
            )

            top3_overlap_retention_on_matched = (
                None
                if matched_objects <= 0
                else (
                    (
                        matched_objects
                        - s["top3_overlap_error_sum"]
                    )
                    / matched_objects
                    * 100.0
                )
            )

            top3_exact_change_on_matched = safe_percent(
                s["top3_exact_changed_objects"],
                matched_objects,
            )

            top3_exact_agreement_on_matched = safe_percent(
                s["top3_exact_same_objects"],
                matched_objects,
            )

            top3_e2e_error_rate = (
                None
                if baseline_objects <= 0
                else (
                    (
                        missed_objects
                        + s["top3_overlap_error_sum"]
                    )
                    / baseline_objects
                    * 100.0
                )
            )

            top3_e2e_retention = (
                None
                if baseline_objects <= 0
                else (
                    (
                        matched_objects
                        - s["top3_overlap_error_sum"]
                    )
                    / baseline_objects
                    * 100.0
                )
            )

            mean_top3_overlap_count = (
                None
                if matched_objects <= 0
                else (
                    s["top3_overlap_count_sum"]
                    / matched_objects
                )
            )

            extra_detection_rate = safe_percent(
                s[
                    "extra_compressed_objects"
                ],
                baseline_objects,
            )

            eligible_coverage = safe_percent(
                baseline_objects,
                s["baseline_raw_objects"],
            )

            yolos_count_nmae_percent = (
                None
                if baseline_objects <= 0
                else (
                    s["yolos_count_abs_error"]
                    / baseline_objects
                    * 100.0
                )
            )

            row = {
                "quality": f"Q{quality}",
                "force_ind": force_ind,
                "route": route,
                "aggregation": "per_route",
                "mean_bpp": mean_bpp,
                "min_box_side_px":
                    args.min_box_side,
                "baseline_raw_objects":
                    s["baseline_raw_objects"],
                "baseline_objects":
                    baseline_objects,
                "baseline_excluded_objects":
                    s["baseline_excluded_objects"],
                "baseline_eligible_coverage_percent":
                    eligible_coverage,
                "compressed_raw_detections":
                    s["compressed_raw_detections"],
                "compressed_detections":
                    s["compressed_detections"],
                "compressed_excluded_detections":
                    s["compressed_excluded_detections"],
                "matched_objects":
                    matched_objects,
                "missed_objects":
                    missed_objects,

                # Top-1 counts / rates
                "same_top1_objects":
                    same_top1_objects,
                "top1_changed_objects":
                    top1_changed_objects,
                "top1_change_rate_on_matched_percent":
                    top1_change_on_matched,
                "top1_agreement_on_matched_percent":
                    top1_agreement_on_matched,
                "end_to_end_top1_error_rate_percent":
                    top1_e2e_error_rate,
                "end_to_end_top1_retention_percent":
                    top1_e2e_retention,

                # Top-3 counts / rates
                "top3_exact_same_objects":
                    s["top3_exact_same_objects"],
                "top3_exact_changed_objects":
                    s["top3_exact_changed_objects"],
                "top3_exact_change_rate_on_matched_percent":
                    top3_exact_change_on_matched,
                "top3_exact_agreement_on_matched_percent":
                    top3_exact_agreement_on_matched,
                "mean_top3_overlap_count_of_3":
                    mean_top3_overlap_count,
                "top3_overlap_error_rate_on_matched_percent":
                    top3_overlap_error_on_matched,
                "top3_overlap_retention_on_matched_percent":
                    top3_overlap_retention_on_matched,
                "end_to_end_top3_overlap_error_rate_percent":
                    top3_e2e_error_rate,
                "end_to_end_top3_overlap_retention_percent":
                    top3_e2e_retention,

                # Shared diagnostics
                "extra_compressed_objects":
                    s["extra_compressed_objects"],
                "mean_match_iou":
                    safe_mean(
                        s["match_ious"]
                    ),
                "detection_loss_rate_percent":
                    detection_loss_rate,
                "extra_detection_rate_percent":
                    extra_detection_rate,
                "yolos_count_nmae_percent":
                    yolos_count_nmae_percent,
            }

            route_rows[
                route
            ] = row

            summary_rows.append(
                row
            )

            quality_json[
                route
            ] = row

        # ---------------------------------------------------------
        # MACRO summary.
        #
        # Every combined error/rate below is the simple arithmetic mean
        # of valid car, motorcycle, and bicycle PER-ROUTE rates.
        # NO micro/object-count-weighted ALL error is reported.
        # ---------------------------------------------------------
        def macro_metric(field_name: str) -> Optional[float]:
            values = [
                route_rows[route][field_name]
                for route in ROUTE_CLASSES
                if route_rows[route][field_name]
                is not None
            ]

            return safe_mean(values)

        macro_row = {
            "quality": f"Q{quality}",
            "force_ind": force_ind,
            "route": "MACRO",
            "aggregation": "macro_equal_route_weight",
            "mean_bpp": mean_bpp,
            "min_box_side_px":
                args.min_box_side,

            # Counts shown only as diagnostics.
            "baseline_raw_objects": sum(
                s["baseline_raw_objects"]
                for s in stats.values()
            ),
            "baseline_objects": sum(
                s["baseline_objects"]
                for s in stats.values()
            ),
            "baseline_excluded_objects": sum(
                s["baseline_excluded_objects"]
                for s in stats.values()
            ),
            "compressed_raw_detections": sum(
                s["compressed_raw_detections"]
                for s in stats.values()
            ),
            "compressed_detections": sum(
                s["compressed_detections"]
                for s in stats.values()
            ),
            "compressed_excluded_detections": sum(
                s["compressed_excluded_detections"]
                for s in stats.values()
            ),
            "matched_objects": sum(
                s["matched_objects"]
                for s in stats.values()
            ),
            "missed_objects": sum(
                s["missed_objects"]
                for s in stats.values()
            ),
            "same_top1_objects": sum(
                s["same_top1_objects"]
                for s in stats.values()
            ),
            "top1_changed_objects": sum(
                s["top1_changed_objects"]
                for s in stats.values()
            ),
            "top3_exact_same_objects": sum(
                s["top3_exact_same_objects"]
                for s in stats.values()
            ),
            "top3_exact_changed_objects": sum(
                s["top3_exact_changed_objects"]
                for s in stats.values()
            ),
            "extra_compressed_objects": sum(
                s["extra_compressed_objects"]
                for s in stats.values()
            ),

            # Macro coverage/shared diagnostics.
            "baseline_eligible_coverage_percent":
                macro_metric(
                    "baseline_eligible_coverage_percent"
                ),
            "mean_match_iou":
                macro_metric(
                    "mean_match_iou"
                ),
            "detection_loss_rate_percent":
                macro_metric(
                    "detection_loss_rate_percent"
                ),
            "extra_detection_rate_percent":
                macro_metric(
                    "extra_detection_rate_percent"
                ),
            "yolos_count_nmae_percent":
                macro_metric(
                    "yolos_count_nmae_percent"
                ),

            # MACRO Top-1 metrics.
            "top1_change_rate_on_matched_percent":
                macro_metric(
                    "top1_change_rate_on_matched_percent"
                ),
            "top1_agreement_on_matched_percent":
                macro_metric(
                    "top1_agreement_on_matched_percent"
                ),
            "end_to_end_top1_error_rate_percent":
                macro_metric(
                    "end_to_end_top1_error_rate_percent"
                ),
            "end_to_end_top1_retention_percent":
                macro_metric(
                    "end_to_end_top1_retention_percent"
                ),

            # MACRO Top-3 metrics.
            "top3_exact_change_rate_on_matched_percent":
                macro_metric(
                    "top3_exact_change_rate_on_matched_percent"
                ),
            "top3_exact_agreement_on_matched_percent":
                macro_metric(
                    "top3_exact_agreement_on_matched_percent"
                ),
            "mean_top3_overlap_count_of_3":
                macro_metric(
                    "mean_top3_overlap_count_of_3"
                ),
            "top3_overlap_error_rate_on_matched_percent":
                macro_metric(
                    "top3_overlap_error_rate_on_matched_percent"
                ),
            "top3_overlap_retention_on_matched_percent":
                macro_metric(
                    "top3_overlap_retention_on_matched_percent"
                ),
            "end_to_end_top3_overlap_error_rate_percent":
                macro_metric(
                    "end_to_end_top3_overlap_error_rate_percent"
                ),
            "end_to_end_top3_overlap_retention_percent":
                macro_metric(
                    "end_to_end_top3_overlap_retention_percent"
                ),
        }

        summary_rows.append(
            macro_row
        )

        quality_json[
            "MACRO"
        ] = macro_row

        json_summary[
            f"Q{quality}"
        ] = quality_json

        # ---------------------------------------------------------
        # Console summary
        # ---------------------------------------------------------

        print(
            f"\nQ{quality} BBOX-GATED TOP-1 + TOP-3 END-TO-END SUMMARY"
        )

        for route in ROUTE_CLASSES:
            r = quality_json[
                route
            ]

            print(
                f"  {route:10s} "
                f"eligible={r['baseline_objects']}/"
                f"{r['baseline_raw_objects']} "
                f"({fmt_optional(r['baseline_eligible_coverage_percent'], 2)}%) | "
                f"miss={r['missed_objects']} "
                f"({fmt_optional(r['detection_loss_rate_percent'], 2)}%)"
            )

            print(
                f"    Top1: change="
                f"{fmt_optional(r['top1_change_rate_on_matched_percent'], 2)}% matched | "
                f"E2E="
                f"{fmt_optional(r['end_to_end_top1_error_rate_percent'], 2)}%"
            )

            print(
                f"    Top3: overlap-error="
                f"{fmt_optional(r['top3_overlap_error_rate_on_matched_percent'], 2)}% matched | "
                f"mean-overlap="
                f"{fmt_optional(r['mean_top3_overlap_count_of_3'], 3)}/3 | "
                f"exact-set-change="
                f"{fmt_optional(r['top3_exact_change_rate_on_matched_percent'], 2)}% matched | "
                f"E2E-overlap-error="
                f"{fmt_optional(r['end_to_end_top3_overlap_error_rate_percent'], 2)}%"
            )

            print(
                f"    YOLOS-NMAE="
                f"{fmt_optional(r['yolos_count_nmae_percent'], 2)}%"
            )

        m = quality_json[
            "MACRO"
        ]

        print(
            "  MACRO"
        )

        print(
            f"    coverage="
            f"{fmt_optional(m['baseline_eligible_coverage_percent'], 2)}% | "
            f"Detection-loss="
            f"{fmt_optional(m['detection_loss_rate_percent'], 2)}% | "
            f"YOLOS-NMAE="
            f"{fmt_optional(m['yolos_count_nmae_percent'], 2)}%"
        )

        print(
            f"    Top1: change="
            f"{fmt_optional(m['top1_change_rate_on_matched_percent'], 2)}% matched | "
            f"E2E="
            f"{fmt_optional(m['end_to_end_top1_error_rate_percent'], 2)}%"
        )

        print(
            f"    Top3: overlap-error="
            f"{fmt_optional(m['top3_overlap_error_rate_on_matched_percent'], 2)}% matched | "
            f"mean-overlap="
            f"{fmt_optional(m['mean_top3_overlap_count_of_3'], 3)}/3 | "
            f"exact-set-change="
            f"{fmt_optional(m['top3_exact_change_rate_on_matched_percent'], 2)}% matched | "
            f"E2E-overlap-error="
            f"{fmt_optional(m['end_to_end_top3_overlap_error_rate_percent'], 2)}%"
        )

        # Save incrementally after every quality.
        save_csv(
            args.per_object_csv,
            per_object_rows,
        )

        save_csv(
            args.summary_csv,
            summary_rows,
        )

        args.out_json.write_text(
            json.dumps(
                {
                    "experiment": (
                        "Preprocessed 768x512 Cityscapes bbox-gated object-matched "
                        "EF-LIC -> YOLOS -> trained classifier Top-1 + Top-3 "
                        "macro end-to-end error"
                    ),
                    "spatial_preprocess": {
                        "input_width": 2048,
                        "input_height": 1024,
                        "center_crop_width": 1536,
                        "center_crop_height": 1024,
                        "resize_width": TARGET_WIDTH,
                        "resize_height": TARGET_HEIGHT,
                        "resize_interpolation": "bicubic_antialias",
                        "bbox_gate_min_width_height_px": args.min_box_side,
                        "bbox_coordinate_system": "preprocessed_768x512",
                        "bpp_denominator_pixels": TARGET_WIDTH * TARGET_HEIGHT,
                    },
                    "metric_definition": {
                        "bbox_gate": (
                            "The same minimum width AND minimum height gate is "
                            "applied independently to both uncompressed and "
                            "compressed YOLOS detections before matching and "
                            "classifier inference."
                        ),
                        "baseline_denominator": (
                            "Uncompressed YOLOS detections that pass the "
                            "bbox-size gate."
                        ),
                        "matching": (
                            "Eligible detections only; same YOLOS route class + "
                            "greedy highest IoU with "
                            f"IoU >= {args.match_iou}."
                        ),
                        "top1_per_route_e2e_error": (
                            "(missed eligible baseline objects + matched Top-1 "
                            "changes) / eligible baseline objects."
                        ),
                        "top3_comparison": (
                            "For each matched object, compare the unordered "
                            "baseline Top-3 label set with the unordered "
                            "compressed Top-3 label set."
                        ),
                        "top3_matched_overlap_error": (
                            "1 - |baseline Top-3 intersection compressed Top-3| / 3. "
                            "Thus 3/3 overlap=0 error, 2/3=1/3 error, "
                            "1/3=2/3 error, 0/3=1 error."
                        ),
                        "top3_per_route_e2e_overlap_error": (
                            "(missed eligible baseline objects + summed matched "
                            "Top-3 overlap errors) / eligible baseline objects. "
                            "A missed object contributes 1.0 error."
                        ),
                        "top3_exact_set_change": (
                            "Diagnostic binary rate among matched objects. "
                            "Changed if the baseline and compressed unordered "
                            "Top-3 sets are not exactly equal."
                        ),
                        "macro_aggregation": (
                            "Every combined error/rate is the simple arithmetic "
                            "mean of valid car, motorcycle, and bicycle per-route "
                            "rates. No object-count-weighted micro ALL error is used."
                        ),
                        "macro_yolos_count_nmae": (
                            "Simple arithmetic mean of valid per-route count "
                            "NMAEs computed on bbox-eligible YOLOS detections."
                        ),
                        "coverage": (
                            "Eligible baseline detections / raw baseline YOLOS "
                            "detections."
                        ),
                    },
                    "input": (
                        "Original 2048x1024 Cityscapes frame; center crop "
                        "1536x1024, bicubic antialiased resize 768x512 "
                        "before EF-LIC and both YOLOS branches."
                    ),
                    "qualities":
                        json_summary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        "Per-object results :",
        args.per_object_csv.resolve(),
    )

    print(
        "Summary CSV        :",
        args.summary_csv.resolve(),
    )

    print(
        "Summary JSON       :",
        args.out_json.resolve(),
    )


if __name__ == "__main__":
    main()
