#!/usr/bin/env python3
"""
EF‑LIC Cityscapes Benchmark (Human‑Friendly Version)

This script evaluates the EF‑LIC image compression model on the Cityscapes
validation dataset. It measures encoding/decoding latency, bits per pixel (BPP),
PSNR, and compression ratios for multiple rate points (force indices).

- BPP is derived from the packed bitstream (payload length).
- Compression ratio is defined as:
    Input FP32 tensor size (H × W × 3 × 4 bytes) / compressed tensor raw bytes
    (obtained via .cpu().numpy().tobytes() on z_inds and y_inds).

The code is organised for clarity: data loading, model inference, bit packing,
metric computation, and result aggregation each have their own module or function.
"""

import argparse
import csv
import json
import logging
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ----------------------------------------------------------------------
# Model import (adjust path if needed)
# ----------------------------------------------------------------------
from EF_LIC import model

# ----------------------------------------------------------------------
# Constants & Configuration
# ----------------------------------------------------------------------
FORCE_INDICES = range(5)          # Rate points to evaluate
TARGET_WIDTH = 768                # Output image width
TARGET_HEIGHT = 512               # Output image height

# Flag to only print the deep tensor-shape analysis once per run
_DIAGNOSTIC_DONE = False

# Cityscapes preprocessing:
# Original: 2048×1024 (2:1) → centre crop to 1536×1024 (3:2) →
# resize to 768×512 with bicubic antialiasing → map to [-1, 1].
PREPROCESS = transforms.Compose([
    transforms.CenterCrop((1024, 1536)),
    transforms.Resize(
        (TARGET_HEIGHT, TARGET_WIDTH),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    ),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t * 2.0 - 1.0),
])

# ----------------------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Data Containers
# ----------------------------------------------------------------------
@dataclass
class BenchmarkConfig:
    """All configurable parameters for the benchmark."""
    cityscapes_root: Path
    checkpoint_path: Path
    device: torch.device
    warmup: int
    repeats: int
    out_json: Path
    out_csv: Path


@dataclass
class ImageMetrics:
    """Metrics collected for a single image."""
    enc_times_ms: List[float]      # encode latencies (ms) per repeat
    dec_times_ms: List[float]      # decode latencies (ms) per repeat
    bpp: float                     # bits per pixel (from packed payload)
    compressed_tensor_bytes: int   # raw tensor bytes from .tobytes() (for CR)
    mse: float                     # mean squared error (in [0,1])
    psnr_db: float                 # PSNR in dB
    # [ADDED] Diagnostics
    input_batch: int
    output_batch: int


@dataclass
class RatePointResult:
    """Aggregated results for one force index (rate point)."""
    force_index: int
    bpp: float
    psnr_db: float
    input_size_bytes: int
    input_size_mib: float
    input_size_mib_per_image: float
    compressed_tensor_bytes_total: float
    compressed_tensor_mib_total: float
    compressed_tensor_mib_per_image: float
    payload_bytes_total: int
    payload_mib_total: float
    compression_ratio: float
    enc_stats: Dict[str, float]
    dec_stats: Dict[str, float]
    peak_gpu_mem_mb: Optional[float] = None
    input_batch: Optional[int] = None       # [ADDED]
    output_batch: Optional[int] = None      # [ADDED]
    batch_expansion_factor: Optional[float] = None  # [ADDED]


# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def list_cityscapes_images(root: Path) -> List[Path]:
    val_dir = root / "leftImg8bit" / "val"
    images = sorted(p for p in val_dir.rglob("*_leftImg8bit.png") if p.is_file())
    if not images:
        raise FileNotFoundError(f"No Cityscapes validation images found under {val_dir}")
    return images


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = PREPROCESS(img).unsqueeze(0).to(device, non_blocking=True)
    return tensor


def compute_mse(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    pred = ((x_hat + 1.0) * 0.5).clamp(0, 1)
    target = ((x + 1.0) * 0.5).clamp(0, 1)
    return F.mse_loss(pred, target).item()


def psnr_from_mse(mse: float) -> float:
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def pack_indices(network, indices: Dict[str, torch.Tensor]) -> bytes:
    n_e = tuple(int(x) for x in network.n_e)
    bit_widths = [(n - 1).bit_length() for n in n_e]

    items = [
        (indices["z_inds"], bit_widths[-1]),
        (indices["y_inds"][0], bit_widths[0]),
        (indices["y_inds"][1], bit_widths[1]),
        (indices["y_inds"][2], bit_widths[2]),
        (indices["y_inds"][3], bit_widths[3]),
    ]

    def tensor_to_bits(tensor: torch.Tensor, bits: int) -> np.ndarray:
        values = (
            tensor.detach()
            .reshape(-1)
            .to("cpu", torch.long)
            .numpy()
            .astype(np.uint32, copy=False)
        )
        shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
        bits_array = ((values[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
        return bits_array

    raw_bits = np.concatenate([tensor_to_bits(tensor, bits) for tensor, bits in items])
    return np.packbits(raw_bits, bitorder="big").tobytes()


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def summarize_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    stats = {
        "median": round(statistics.median(values), 3),
        "mean": round(statistics.mean(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }
    if len(values) >= 10:
        q = statistics.quantiles(values, n=10)
        stats["p10"] = round(q[0], 3)
        stats["p90"] = round(q[8], 3)
    return stats


# ----------------------------------------------------------------------
# [ADDED] Diagnostic: batch expansion analysis
# ----------------------------------------------------------------------
def analyze_batch_expansion(
    input_tensor: torch.Tensor,
    indices: Dict[str, torch.Tensor],
    force_ind: int,
) -> None:
    """
    Print a one-time analysis of how the input batch dimension relates to the
    encoder output batch dimension. Helps identify whether a batch size of 2
    in the output is intentional (e.g., parallel streams in the entropy model)
    or a bug in the model's compress() method.
    """
    global _DIAGNOSTIC_DONE
    if _DIAGNOSTIC_DONE:
        return
    _DIAGNOSTIC_DONE = True

    in_batch = input_tensor.shape[0]
    z_batch = indices["z_inds"].shape[0]
    y_batches = [yt.shape[0] for yt in indices["y_inds"]]

    logger.info("=" * 70)
    logger.info("[DIAGNOSTIC] Batch Dimension Analysis")
    logger.info("=" * 70)
    logger.info(f"  Input batch dimension     : {in_batch}")
    logger.info(f"  z_inds batch dimension    : {z_batch}")
    logger.info(f"  y_inds[*] batch dimensions: {y_batches}")

    if in_batch != z_batch or any(b != in_batch for b in y_batches):
        expansion = z_batch / in_batch if in_batch > 0 else float("nan")
        logger.warning(
            f"  ⚠ Batch MISMATCH: input has batch={in_batch}, but encoder "
            f"outputs have batch={z_batch}. Expansion factor = {expansion:.2f}×"
        )
        logger.warning(
            "    Possible explanations:"
        )
        logger.warning(
            "      1. The model intentionally duplicates the batch to create two "
            "parallel streams (e.g., mean+scale, or two entropy models)."
        )
        logger.warning(
            "      2. There is a bug in the model's compress() that incorrectly "
            "expands the batch (e.g., torch.cat([x, x], dim=0))."
        )
        logger.warning(
            "    Action: Inspect the model's compress() method. Look for calls to "
            "torch.cat, torch.stack, .repeat(), or .expand() with dim=0."
        )
        logger.warning(
            f"    Impact on metrics: If unintentional, the raw tensor byte count "
            f"is inflated by {expansion:.2f}×, making the compression ratio "
            f"appear {expansion:.2f}× WORSE than the true value."
        )
    else:
        logger.info(
            f"  ✓ Batch dimensions match (input={in_batch}, output={z_batch}). "
            f"No expansion detected."
        )

    # Report the per-tensor bit-width assignment (helps spot anomalies)
    try:
        n_e = tuple(int(x) for x in model().n_e)  # type: ignore
    except Exception:
        n_e = None
    if n_e is not None:
        bit_widths = [(n - 1).bit_length() for n in n_e]
        logger.info(f"  Model n_e                : {n_e}")
        logger.info(f"  Derived bit widths       : {bit_widths}")

    logger.info("=" * 70)


# ----------------------------------------------------------------------
# Core Benchmark Functions
# ----------------------------------------------------------------------
@torch.inference_mode()
def benchmark_image(
    network: torch.nn.Module,
    image_tensor: torch.Tensor,
    force_ind: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> ImageMetrics:
    """
    Benchmark encoding and decoding for a single image.
    """
    batch_size, _, height, width = image_tensor.shape

    logger.info(
        f"Input Tensor -> Shape: {image_tensor.shape}, "
        f"Dtype: {image_tensor.dtype}, Device: {image_tensor.device}"
    )

    # Warm-up runs (excluded from timing)
    for _ in range(warmup):
        indices = network.compress(image_tensor, force_ind=force_ind)
        network.decompress(indices, force_ind=force_ind)

    synchronize_device(device)

    enc_times = []
    dec_times = []
    payload = None
    x_hat = None
    compressed_tensor_bytes = 0
    output_batch = batch_size  # default; will be updated on first rep

    for rep in range(repeats):
        # ---- Encode (TIMED) ----
        synchronize_device(device)
        t0 = time.perf_counter()
        indices = network.compress(image_tensor, force_ind=force_ind)
        synchronize_device(device)
        enc_times.append((time.perf_counter() - t0) * 1000.0)

        # ---- Pack payload (NOT TIMED, for BPP) ----
        payload = pack_indices(network, indices)

        # ---- Decode (TIMED) ----
        synchronize_device(device)
        t1 = time.perf_counter()
        x_hat = network.decompress(indices, force_ind=force_ind)
        synchronize_device(device)
        dec_times.append((time.perf_counter() - t1) * 1000.0)

        # ---- Serialize output tensors (NOT TIMED) ----
        if rep == 0:
            # Run the diagnostic once per process
            analyze_batch_expansion(image_tensor, indices, force_ind)

            z_bytes = indices['z_inds'].cpu().numpy().tobytes()
            y_bytes_list = [yt.cpu().numpy().tobytes() for yt in indices["y_inds"]]
            compressed_tensor_bytes = len(z_bytes) + sum(len(yb) for yb in y_bytes_list)

            output_batch = indices["z_inds"].shape[0]

            logger.info(f"Encoder Output for force_ind={force_ind}:")
            logger.info(
                f"  z_inds -> Shape: {indices['z_inds'].shape}, "
                f"Dtype: {indices['z_inds'].dtype}"
            )
            for i, yt in enumerate(indices["y_inds"]):
                logger.info(
                    f"  y_inds[{i}] -> Shape: {yt.shape}, "
                    f"Dtype: {yt.dtype}"
                )
            logger.info(
                f"Compressed Tensor Raw Size (via .tobytes()): {compressed_tensor_bytes} bytes"
            )
            logger.info(
                f"Packed payload size (for BPP): {len(payload)} bytes ({len(payload)*8} bits)"
            )

    if payload is None or x_hat is None:
        raise RuntimeError("No repetitions completed successfully")

    bpp_from_payload = (len(payload) * 8.0) / float(batch_size * height * width)
    mse = compute_mse(x_hat, image_tensor)

    return ImageMetrics(
        enc_times_ms=enc_times,
        dec_times_ms=dec_times,
        bpp=bpp_from_payload,
        compressed_tensor_bytes=compressed_tensor_bytes,
        mse=mse,
        psnr_db=psnr_from_mse(mse),
        input_batch=batch_size,
        output_batch=output_batch,
    )


def run_benchmark_for_force_index(
    network: torch.nn.Module,
    force_ind: int,
    image_paths: List[Path],
    config: BenchmarkConfig,
) -> RatePointResult:
    logger.info(f"========== force_ind={force_ind} ==========")
    network.prepare_inference_(force_ind=force_ind)

    if config.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    enc_medians = []
    dec_medians = []
    bpp_values = []
    tensor_bytes_values = []
    payload_bytes_values = []
    mse_values = []
    input_batch = None
    output_batch = None

    for img_path in tqdm(image_paths, desc=f"force_ind {force_ind}", unit="img"):
        image_tensor = load_image(img_path, config.device)
        _, _, h, w = image_tensor.shape

        metrics = benchmark_image(
            network,
            image_tensor,
            force_ind,
            config.device,
            config.warmup,
            config.repeats,
        )

        enc_median = statistics.median(metrics.enc_times_ms)
        dec_median = statistics.median(metrics.dec_times_ms)
        enc_medians.append(enc_median)
        dec_medians.append(dec_median)
        bpp_values.append(metrics.bpp)
        tensor_bytes_values.append(metrics.compressed_tensor_bytes)
        mse_values.append(metrics.mse)

        payload_bytes = (metrics.bpp * h * w) / 8.0
        payload_bytes_values.append(payload_bytes)

        # Record batch dimensions (first image is enough)
        if input_batch is None:
            input_batch = metrics.input_batch
            output_batch = metrics.output_batch

        logger.info(
            f"{img_path.parent.name}/{img_path.name} | "
            f"input={(h*w*3*4)/(1024**2):.4f}MiB (FP32) "
            f"enc={enc_median:7.2f}ms dec={dec_median:7.2f}ms "
            f"PSNR={metrics.psnr_db:6.2f}dB "
            f"BPP={metrics.bpp:.4f} (payload) "
            f"comp_tensor={(metrics.compressed_tensor_bytes)/(1024**2):.4f}MiB "
            f"CR={(h*w*3*4) / metrics.compressed_tensor_bytes:.2f}x"
        )

    # --- Aggregate ---
    num_images = len(image_paths)
    total_pixels = num_images * TARGET_HEIGHT * TARGET_WIDTH

    # [FIXED] Input baseline: FP32 tensor = H * W * 3 channels * 4 bytes
    total_input_bytes = total_pixels * 3 * 4

    total_tensor_bytes = sum(tensor_bytes_values)
    total_payload_bytes = sum(payload_bytes_values)

    avg_bpp = float(np.mean(bpp_values))
    avg_mse = float(np.mean(mse_values))
    dataset_psnr = psnr_from_mse(avg_mse)

    # [FIXED] Compression Ratio now correctly uses 3-channel FP32 input size
    dataset_cr = (
        total_input_bytes / total_tensor_bytes
        if total_tensor_bytes > 0
        else float("inf")
    )

    peak_mb = None
    if config.device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1e6

    enc_stats = summarize_stats(enc_medians)
    dec_stats = summarize_stats(dec_medians)

    batch_expansion = None
    if input_batch is not None and input_batch > 0:
        batch_expansion = output_batch / input_batch

    result = RatePointResult(
        force_index=force_ind,
        bpp=round(avg_bpp, 5),
        psnr_db=round(dataset_psnr, 3),
        input_size_bytes=total_input_bytes,
        input_size_mib=round(total_input_bytes / (1024 ** 2), 4),
        input_size_mib_per_image=round(
            (TARGET_HEIGHT * TARGET_WIDTH * 3 * 4) / (1024 ** 2), 4
        ),
        compressed_tensor_bytes_total=round(total_tensor_bytes, 3),
        compressed_tensor_mib_total=round(total_tensor_bytes / (1024 ** 2), 6),
        compressed_tensor_mib_per_image=round(
            (total_tensor_bytes / num_images) / (1024 ** 2), 6
        ),
        payload_bytes_total=round(total_payload_bytes, 3),
        payload_mib_total=round(total_payload_bytes / (1024 ** 2), 6),
        compression_ratio=round(dataset_cr, 3),
        enc_stats=enc_stats,
        dec_stats=dec_stats,
        peak_gpu_mem_mb=round(peak_mb, 1) if peak_mb is not None else None,
        input_batch=input_batch,
        output_batch=output_batch,
        batch_expansion_factor=(
            round(batch_expansion, 3) if batch_expansion is not None else None
        ),
    )

    logger.info(
        f"AVG force_ind={force_ind}: "
        f"BPP={result.bpp:.4f} (payload) "
        f"CR={result.compression_ratio:.2f}x (FP32 / raw tensors) "
        f"PSNR={result.psnr_db:.2f}dB "
        f"enc={enc_stats['median']:.2f}ms "
        f"dec={dec_stats['median']:.2f}ms "
        f"mem={result.peak_gpu_mem_mb}MB "
        f"input={result.input_size_mib_per_image:.4f}MiB (FP32) "
        f"comp_tensor={result.compressed_tensor_mib_per_image:.6f}MiB "
        f"batch_in={input_batch} batch_out={output_batch} "
        f"(expansion={batch_expansion})"
    )

    return result


# ----------------------------------------------------------------------
# Main Benchmark Runner
# ----------------------------------------------------------------------
def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    network = model().to(config.device).eval()
    try:
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
        state_dict = (
            checkpoint.get("state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        network.load_state_dict(state_dict, strict=True)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

    image_paths = list_cityscapes_images(config.cityscapes_root)
    num_images = len(image_paths)
    logger.info(f"Found {num_images} images on {config.device}")

    results = {
        "codec": "EF-LIC",
        "dataset": "Cityscapes validation",
        "cityscapes_root": str(config.cityscapes_root),
        "device": str(config.device),
        "num_images": num_images,
        "warmup": config.warmup,
        "repeats": config.repeats,
        "preprocess_crop": {"height": 1024, "width": 1536, "type": "center"},
        "preprocess_resolution": {"width": TARGET_WIDTH, "height": TARGET_HEIGHT},
        "preprocess_interpolation": "bicubic antialias",
        "bpp_definition": "bits per pixel = (packed_payload_bytes * 8) / (H * W)",
        "compression_ratio_definition": (
            "FP32 tensor bytes (H*W*3*4) / raw compressed tensor bytes from "
            ".numpy().tobytes() of z_inds and y_inds"
        ),
        "rate_points": {},
    }

    csv_rows = []

    for force_ind in FORCE_INDICES:
        rate_result = run_benchmark_for_force_index(
            network, force_ind, image_paths, config
        )

        rate_dict = {
            "bpp": rate_result.bpp,
            "psnr_dB": rate_result.psnr_db,
            "input_size_bytes": rate_result.input_size_bytes,
            "input_size_MiB": rate_result.input_size_mib,
            "input_size_MiB_per_image": rate_result.input_size_mib_per_image,
            "compressed_tensor_bytes_total": rate_result.compressed_tensor_bytes_total,
            "compressed_tensor_MiB_total": rate_result.compressed_tensor_mib_total,
            "compressed_tensor_MiB_per_image": rate_result.compressed_tensor_mib_per_image,
            "payload_bytes_total": rate_result.payload_bytes_total,
            "payload_MiB_total": rate_result.payload_mib_total,
            "compression_ratio": rate_result.compression_ratio,
            "enc_ms": rate_result.enc_stats,
            "dec_ms": rate_result.dec_stats,
            "peak_gpu_mem_MB": rate_result.peak_gpu_mem_mb,
            "input_batch": rate_result.input_batch,
            "output_batch": rate_result.output_batch,
            "batch_expansion_factor": rate_result.batch_expansion_factor,
        }
        results["rate_points"][f"force_ind_{force_ind}"] = rate_dict

        row = {
            "force_ind": rate_result.force_index,
            "bpp": rate_result.bpp,
            "psnr_dB": rate_result.psnr_db,
            "input_size_bytes": rate_result.input_size_bytes,
            "input_size_MiB": rate_result.input_size_mib,
            "input_size_MiB_per_image": rate_result.input_size_mib_per_image,
            "compressed_tensor_bytes_total": rate_result.compressed_tensor_bytes_total,
            "compressed_tensor_MiB_total": rate_result.compressed_tensor_mib_total,
            "compressed_tensor_MiB_per_image": rate_result.compressed_tensor_mib_per_image,
            "payload_bytes_total": rate_result.payload_bytes_total,
            "payload_MiB_total": rate_result.payload_mib_total,
            "compression_ratio": rate_result.compression_ratio,
            "enc_median_ms": rate_result.enc_stats.get("median"),
            "enc_mean_ms": rate_result.enc_stats.get("mean"),
            "enc_min_ms": rate_result.enc_stats.get("min"),
            "enc_max_ms": rate_result.enc_stats.get("max"),
            "enc_p10_ms": rate_result.enc_stats.get("p10", ""),
            "enc_p90_ms": rate_result.enc_stats.get("p90", ""),
            "dec_median_ms": rate_result.dec_stats.get("median"),
            "dec_mean_ms": rate_result.dec_stats.get("mean"),
            "dec_min_ms": rate_result.dec_stats.get("min"),
            "dec_max_ms": rate_result.dec_stats.get("max"),
            "dec_p10_ms": rate_result.dec_stats.get("p10", ""),
            "dec_p90_ms": rate_result.dec_stats.get("p90", ""),
            "peak_gpu_mem_MB": rate_result.peak_gpu_mem_mb,
            "input_batch": rate_result.input_batch,
            "output_batch": rate_result.output_batch,
            "batch_expansion_factor": rate_result.batch_expansion_factor,
        }
        csv_rows.append(row)

        config.out_json.write_text(json.dumps(results, indent=2))

    save_csv(config.out_csv, csv_rows)
    config.out_json.write_text(json.dumps(results, indent=2))

    return results


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = rows[0].keys()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------
# Command-line Interface
# ----------------------------------------------------------------------
def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="EF-LIC Cityscapes Benchmark")
    parser.add_argument(
        "--cityscapes-root",
        type=Path,
        required=True,
        help="Cityscapes root containing leftImg8bit/val",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=Path("ckpt/checkpoint.pth.tar"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use (e.g., 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warm-up runs per image",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of measured repetitions per image",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("results_eflic_cityscapes.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results_eflic_cityscapes.csv"),
        help="Output CSV file",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    return BenchmarkConfig(
        cityscapes_root=args.cityscapes_root,
        checkpoint_path=args.ckpt_path,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
        out_json=args.out_json,
        out_csv=args.out_csv,
    )


def main() -> None:
    config = parse_args()
    logger.info(f"Running benchmark with config: {config}")
    try:
        run_benchmark(config)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()