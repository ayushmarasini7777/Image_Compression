#!/usr/bin/env python3
"""
compression_kodak_benchmark.py
------------------------------------------------------------------------------
Apple-to-apple Kodak benchmark for:
  * TCM (jmliu206/LIC_TCM)
  * bmshj2018-hyperprior (CompressAI)
  * cheng2020-anchor (optional fallback)

The benchmark follows the same aggregation protocol used by the EF-LIC
benchmark:
  * identical sorted Kodak image set
  * warmup passes per image and quality
  * CUDA-synchronized timing
  * median over repetitions for each image
  * dataset statistics over the per-image medians
  * dataset BPP calculated from total coded bits / total original pixels
  * file-size compression ratio calculated as:
        total original encoded-file bytes / total codec-bitstream bytes
  * raw-RGB compression ratio also retained separately
  * PSNR calculated from the mean per-image MSE
  * optional LPIPS and DISTS

Examples
--------
Run TCM q4 on Kodak:

    python3 compression_kodak_benchmark.py \
        --kodak-dir kodak \
        --codecs tcm \
        --qualities q4 \
        --checkpoint tcm:q4=LIC_TCM/checkpoints/tcm_q4.pth.tar \
        --warmup 3 --repeats 10 \
        --out results_tcm_kodak.json

Run bmshj2018-hyperprior q4 on Kodak:

    python3 compression_kodak_benchmark.py \
        --kodak-dir kodak \
        --codecs bmshj2018-hyperprior \
        --qualities q4 \
        --checkpoint bmshj2018-hyperprior:q4=compressai_weights/bmshj_q4.pth.tar \
        --warmup 3 --repeats 10 \
        --out results_bmshj_kodak.json

Run both codecs in one invocation:

    python3 compression_kodak_benchmark.py \
        --kodak-dir kodak \
        --codecs tcm bmshj2018-hyperprior \
        --qualities q4 \
        --checkpoint tcm:q4=LIC_TCM/checkpoints/tcm_q4.pth.tar \
        --checkpoint bmshj2018-hyperprior:q4=compressai_weights/bmshj_q4.pth.tar \
        --warmup 3 --repeats 10 \
        --out results_tcm_bmshj_kodak.json

Important
---------
Quality labels are model-specific. TCM q4 and bmshj q4 are not guaranteed to
have the same BPP or reconstruction quality. Compare codecs at matched BPP,
PSNR, LPIPS, or DISTS rather than assuming equal q labels are equal quality.
------------------------------------------------------------------------------
"""

# --- compressai import guard (must run before compressai is imported) --------
import sys as _sys
import types as _types


class _DummyMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Dummy

    def __call__(cls, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return super().__call__(*args, **kwargs)


class _Dummy(metaclass=_DummyMeta):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return _Dummy()

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Dummy


class _LenientModule(_types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        setattr(self, name, _Dummy)
        return _Dummy


class _TorchGeometricStubFinder:
    _prefix = "torch_geometric"

    def find_module(self, fullname, path=None):
        return self if self._match(fullname) else None

    def find_spec(self, fullname, path=None, target=None):
        if self._match(fullname):
            import importlib.machinery as _m
            return _m.ModuleSpec(fullname, self)
        return None

    def _match(self, fullname):
        return fullname == self._prefix or fullname.startswith(self._prefix + ".")

    def create_module(self, spec):
        module = _LenientModule(spec.name)
        module.__file__ = None
        module.__path__ = []
        module.__all__ = []
        return module

    def exec_module(self, module):
        return None

    def load_module(self, fullname):
        if fullname in _sys.modules:
            return _sys.modules[fullname]
        module = self.create_module(_types.SimpleNamespace(name=fullname))
        _sys.modules[fullname] = module
        return module


try:
    import torch_geometric  # noqa: F401
except ImportError:
    _sys.meta_path.insert(0, _TorchGeometricStubFinder())
# -----------------------------------------------------------------------------

import argparse
import json
import math
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor


# Register convenient defaults here if desired. Command-line --checkpoint
# overrides are preferred because they avoid editing this file.
TCM_CKPTS: Dict[str, str] = {
    "q1": "LIC_TCM/checkpoints/tcm_n64_lambda0.0025.pth.tar",
    "q2": "LIC_TCM/checkpoints/tcm_n64_lambda0.0035.pth.tar",
    "q3": "LIC_TCM/checkpoints/tcm_n64_lambda0.0067.pth.tar",
    "q4": "LIC_TCM/checkpoints/tcm_n64_lambda0.013.pth.tar",
    "q5": "LIC_TCM/checkpoints/tcm_n64_lambda0.025.pth.tar",
    "q6": "LIC_TCM/checkpoints/tcm_n64_lambda0.05.pth.tar",
}

BMSHJ_CKPTS: Dict[str, str] = {
    "q1": "compressai_weights/bmshj2018-hyperprior-1-7eb97409.pth.tar",
    "q2": "compressai_weights/bmshj2018-hyperprior-2-93677231.pth.tar",
    "q3": "compressai_weights/bmshj2018-hyperprior-3-6d87be32.pth.tar",
    "q4": "compressai_weights/bmshj2018-hyperprior-4-de1b779c.pth.tar",
    "q5": "compressai_weights/bmshj2018-hyperprior-5-f8b614e1.pth.tar",
    "q6": "compressai_weights/bmshj2018-hyperprior-6-1ab9c41e.pth.tar",
}

PAD_MULTIPLE = 64
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_checkpoint_overrides(items: Iterable[str]) -> Dict[Tuple[str, str], str]:
    """Parse repeated CODEC:QUALITY=PATH checkpoint specifications."""
    overrides: Dict[Tuple[str, str], str] = {}
    for item in items:
        if "=" not in item or ":" not in item.split("=", 1)[0]:
            raise ValueError(
                f"Invalid --checkpoint value '{item}'. Expected CODEC:QUALITY=PATH"
            )
        left, path = item.split("=", 1)
        codec, quality = left.split(":", 1)
        codec = codec.strip()
        quality = quality.strip()
        path = path.strip()
        if not codec or not quality or not path:
            raise ValueError(
                f"Invalid --checkpoint value '{item}'. Expected CODEC:QUALITY=PATH"
            )
        overrides[(codec, quality)] = path
    return overrides


def resolve_checkpoint(
    codec: str,
    quality: str,
    overrides: Dict[Tuple[str, str], str],
) -> Path:
    override = overrides.get((codec, quality))
    if override:
        path = Path(override)
    elif codec == "tcm":
        configured = TCM_CKPTS.get(quality)
        if not configured:
            raise ValueError(
                f"No TCM checkpoint configured for {quality}. Pass "
                f"--checkpoint tcm:{quality}=PATH or fill TCM_CKPTS."
            )
        path = Path(configured)
    elif codec == "bmshj2018-hyperprior":
        configured = BMSHJ_CKPTS.get(quality)
        if not configured:
            raise ValueError(
                f"No bmshj2018 checkpoint configured for {quality}. Pass "
                f"--checkpoint bmshj2018-hyperprior:{quality}=PATH or fill "
                "BMSHJ_CKPTS."
            )
        path = Path(configured)
    else:
        raise ValueError(f"Codec {codec} does not use a local checkpoint here.")

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")
    return path


def load_codec(
    name: str,
    quality: str,
    device: str,
    checkpoint_overrides: Dict[Tuple[str, str], str],
    tcm_repo: str = "./LIC_TCM",
    N: int = 64,
):
    """Load one codec/quality pair and build entropy CDF tables."""
    if name == "tcm":
        repo = Path(tcm_repo).resolve()
        if not repo.exists():
            raise FileNotFoundError(f"TCM repository not found: {repo}")
        _sys.path.insert(0, str(repo))
        from models import TCM

        net = TCM(
            config=[2, 2, 2, 2, 2, 2],
            head_dim=[8, 16, 32, 32, 16, 8],
            drop_path_rate=0.0,
            N=N,
            M=320,
        )
        ckpt_path = resolve_checkpoint(name, quality, checkpoint_overrides)
        checkpoint = torch.load(ckpt_path, map_location=device)
        state = checkpoint.get("state_dict", checkpoint)
        net.load_state_dict(state)

    elif name == "bmshj2018-hyperprior":
        from compressai.models import ScaleHyperprior
        from compressai.zoo.pretrained import load_pretrained

        ckpt_path = resolve_checkpoint(name, quality, checkpoint_overrides)
        state = torch.load(ckpt_path, map_location=device)
        state = load_pretrained(state)
        net = ScaleHyperprior.from_state_dict(state)

    elif name == "cheng2020":
        from compressai.zoo import cheng2020_anchor

        quality_int = int(quality[1:] if quality.lower().startswith("q") else quality)
        net = cheng2020_anchor(quality=quality_int, pretrained=True)

    else:
        raise ValueError(f"Unknown codec: {name}")

    net = net.to(device).eval()
    net.update(force=True)
    return net


def list_images(root: Path) -> List[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"Kodak directory not found: {root.resolve()}")
    images = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No supported images found in {root.resolve()}")
    return images


def padded_size(h: int, w: int, multiple: int = PAD_MULTIPLE) -> Tuple[int, int]:
    ph = (h + multiple - 1) // multiple * multiple
    pw = (w + multiple - 1) // multiple * multiple
    return ph, pw


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int = PAD_MULTIPLE,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, h, w = x.shape
    ph, pw = padded_size(h, w, multiple)
    if ph == h and pw == w:
        return x, (h, w)
    return F.pad(x, (0, pw - w, 0, ph - h), mode="replicate"), (h, w)


def bitstream_bytes(output: dict) -> int:
    return sum(len(stream) for group in output["strings"] for stream in group)


def summarize(values: List[float]) -> Optional[dict]:
    if not values:
        return None
    result = {
        "median": round(statistics.median(values), 3),
        "mean": round(statistics.mean(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }
    if len(values) >= 10:
        deciles = statistics.quantiles(values, n=10)
        result["p10"] = round(deciles[0], 3)
        result["p90"] = round(deciles[8], 3)
    return result


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, received {tuple(x.shape)}")
        x = x[0]
    array = (
        x.detach()
        .clamp(0, 1)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def save_reconstruction(
    x_hat: torch.Tensor,
    source_path: Path,
    output_root: Path,
    codec: str,
    quality: str,
) -> str:
    output_dir = output_root / codec / quality / "reconstructed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_path.stem}_reconstructed.png"
    tensor_to_pil(x_hat).save(output_path)
    return str(output_path)


@torch.inference_mode()
def measure_image(
    net,
    image_tensor: torch.Tensor,
    device: str,
    warmup: int,
    repeats: int,
    fp16: bool,
    lpips_fn=None,
    dists_fn=None,
    keep_reconstruction: bool = False,
) -> dict:
    """Measure one image and return one JSON-serializable result dictionary."""
    x_pad, (h, w) = pad_to_multiple(image_tensor.unsqueeze(0).to(device))
    ph, pw = x_pad.shape[-2:]
    cuda = str(device).startswith("cuda")

    def amp():
        if fp16 and cuda:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    if cuda:
        torch.cuda.reset_peak_memory_stats()

    # Warm up both encode and decode, matching the EF-LIC harness structure.
    for _ in range(warmup):
        output = net.compress(x_pad)
        _ = net.decompress(output["strings"], output["shape"])
    if cuda:
        torch.cuda.synchronize()

    # GPU-only analysis-transform timing. This is supplemental; the comparable
    # compression overhead is full net.compress() below.
    transform_ms: List[float] = []
    if hasattr(net, "g_a") and cuda:
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with amp():
                _ = net.g_a(x_pad)
            end.record()
            torch.cuda.synchronize()
            transform_ms.append(float(start.elapsed_time(end)))

    encode_ms: List[float] = []
    outputs: List[dict] = []
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        output = net.compress(x_pad)  # always FP32: entropy coding is bitrate-sensitive
        if cuda:
            torch.cuda.synchronize()
        encode_ms.append((time.perf_counter() - start_time) * 1000.0)
        outputs.append(output)

    output = outputs[-1]
    nbytes = bitstream_bytes(output)
    encode_peak_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if cuda else None
    )

    # Decode the same final bitstream repeatedly. Compression time is not part of
    # decode timing.
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    decode_ms: List[float] = []
    reconstruction = None
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        decoded = net.decompress(output["strings"], output["shape"])
        if cuda:
            torch.cuda.synchronize()
        decode_ms.append((time.perf_counter() - start_time) * 1000.0)
        reconstruction = decoded["x_hat"]

    decode_peak_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if cuda else None
    )

    if reconstruction is None:
        raise RuntimeError("No decompression result was produced.")

    x_hat = reconstruction.clamp(0, 1)[..., :h, :w]
    x_orig = image_tensor.unsqueeze(0).to(device)[..., :h, :w]

    mse = torch.mean((x_hat - x_orig) ** 2).item()
    psnr_db = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)

    ms_ssim_value = None
    if min(h, w) >= 161:
        try:
            from pytorch_msssim import ms_ssim
            ms_ssim_value = ms_ssim(x_hat, x_orig, data_range=1.0).item()
        except Exception:
            pass

    lpips_value = None
    dists_value = None
    if lpips_fn is not None:
        lpips_value = lpips_fn(
            x_hat * 2.0 - 1.0,
            x_orig * 2.0 - 1.0,
        ).mean().item()
        dists_value = dists_fn(
            x_hat,
            x_orig,
            require_grad=False,
        ).detach().mean().item()

    result = {
        "status": "ok",
        "precision": "fp16" if fp16 else "fp32",
        "orig_hw": [h, w],
        "padded_hw": [ph, pw],
        "pixels": int(h * w),
        "bitstream_bytes": int(nbytes),
        "raw_rgb_bytes": int(h * w * 3),
        "raw_rgb_compression_ratio": round((h * w * 3) / nbytes, 3),
        "bpp": round(8.0 * nbytes / (h * w), 5),
        "mse": mse,
        "psnr_dB": round(psnr_db, 3) if math.isfinite(psnr_db) else psnr_db,
        "ms_ssim": round(ms_ssim_value, 5) if ms_ssim_value is not None else None,
        "lpips": round(lpips_value, 5) if lpips_value is not None else None,
        "dists": round(dists_value, 5) if dists_value is not None else None,
        "gpu_transform_ms": summarize(transform_ms),
        "enc_ms": summarize(encode_ms),
        "dec_ms": summarize(decode_ms),
        "encode_peak_gpu_mem_MB": (
            round(encode_peak_mb, 1) if encode_peak_mb is not None else None
        ),
        "decode_peak_gpu_mem_MB": (
            round(decode_peak_mb, 1) if decode_peak_mb is not None else None
        ),
    }

    if keep_reconstruction:
        result["_x_hat"] = x_hat.detach().cpu()

    return result


def aggregate_dataset(image_results: List[dict]) -> dict:
    valid = [result for result in image_results if result.get("status") == "ok"]
    if not valid:
        raise RuntimeError("No valid image results were available for aggregation.")

    total_pixels = sum(result["pixels"] for result in valid)
    total_bitstream_bytes = sum(result["bitstream_bytes"] for result in valid)
    total_raw_rgb_bytes = sum(result["raw_rgb_bytes"] for result in valid)
    total_input_file_bytes = sum(result["input_file_bytes"] for result in valid)

    mean_mse = float(np.mean([result["mse"] for result in valid]))
    dataset_psnr = (
        float("inf") if mean_mse == 0 else -10.0 * math.log10(mean_mse)
    )

    def medians(field: str) -> List[float]:
        return [result[field]["median"] for result in valid if result.get(field)]

    def optional_mean(field: str) -> Optional[float]:
        values = [result[field] for result in valid if result.get(field) is not None]
        return float(np.mean(values)) if values else None

    encode_peaks = [
        result["encode_peak_gpu_mem_MB"]
        for result in valid
        if result.get("encode_peak_gpu_mem_MB") is not None
    ]
    decode_peaks = [
        result["decode_peak_gpu_mem_MB"]
        for result in valid
        if result.get("decode_peak_gpu_mem_MB") is not None
    ]

    return {
        "num_images": len(valid),
        "total_pixels": int(total_pixels),
        "input_size_bytes": int(total_input_file_bytes),
        "compressed_size_bytes": int(total_bitstream_bytes),
        "input_size_MB": round(total_input_file_bytes / (1024 ** 2), 4),
        "compressed_size_MB": round(total_bitstream_bytes / (1024 ** 2), 4),
        # User-requested definition: encoded input-file bytes / codec bytes.
        "compression_ratio": round(
            total_input_file_bytes / total_bitstream_bytes,
            3,
        ),
        # Retained because papers often report raw RGB / compressed bytes.
        "raw_rgb_compression_ratio": round(
            total_raw_rgb_bytes / total_bitstream_bytes,
            3,
        ),
        "bpp": round(8.0 * total_bitstream_bytes / total_pixels, 5),
        "psnr_dB": round(dataset_psnr, 3) if math.isfinite(dataset_psnr) else dataset_psnr,
        "ms_ssim": (
            round(optional_mean("ms_ssim"), 5)
            if optional_mean("ms_ssim") is not None
            else None
        ),
        "lpips": (
            round(optional_mean("lpips"), 5)
            if optional_mean("lpips") is not None
            else None
        ),
        "dists": (
            round(optional_mean("dists"), 5)
            if optional_mean("dists") is not None
            else None
        ),
        "gpu_transform_ms": summarize(medians("gpu_transform_ms")),
        "enc_ms": summarize(medians("enc_ms")),
        "dec_ms": summarize(medians("dec_ms")),
        "encode_peak_gpu_mem_MB": round(max(encode_peaks), 1) if encode_peaks else None,
        "decode_peak_gpu_mem_MB": round(max(decode_peaks), 1) if decode_peaks else None,
    }


def benchmark_codec_quality(
    codec: str,
    quality: str,
    images: List[Path],
    device: str,
    checkpoint_overrides: Dict[Tuple[str, str], str],
    tcm_repo: str,
    N: int,
    warmup: int,
    repeats: int,
    fp16: bool,
    lpips_fn,
    dists_fn,
    save_reconstructions: bool,
    save_dir: Path,
) -> dict:
    print(f"\n===== codec={codec} quality={quality} =====")
    net = load_codec(
        codec,
        quality,
        device,
        checkpoint_overrides,
        tcm_repo=tcm_repo,
        N=N,
    )

    image_results: List[dict] = []

    for index, path in enumerate(images, 1):
        image = Image.open(path).convert("RGB")
        tensor = to_tensor(image)

        try:
            result = measure_image(
                net=net,
                image_tensor=tensor,
                device=device,
                warmup=warmup,
                repeats=repeats,
                fp16=fp16,
                lpips_fn=lpips_fn,
                dists_fn=dists_fn,
                keep_reconstruction=save_reconstructions,
            )
        except RuntimeError as error:
            if "out of memory" in str(error).lower() and str(device).startswith("cuda"):
                torch.cuda.empty_cache()
                result = {
                    "status": "OOM",
                    "error": str(error),
                    "input_file_bytes": int(path.stat().st_size),
                }
                image_results.append(result)
                print(f"[{index:02d}/{len(images):02d}] {path.name}: OOM")
                continue
            raise

        result["image"] = path.name
        result["input_file_bytes"] = int(path.stat().st_size)
        result["file_compression_ratio"] = round(
            result["input_file_bytes"] / result["bitstream_bytes"],
            3,
        )

        reconstruction = result.pop("_x_hat", None)
        if reconstruction is not None:
            result["reconstructed_path"] = save_reconstruction(
                reconstruction,
                source_path=path,
                output_root=save_dir,
                codec=codec,
                quality=quality,
            )

        image_results.append(result)

        print(
            f"[{index:02d}/{len(images):02d}] {path.name:14s} "
            f"BPP={result['bpp']:.4f} "
            f"fileCR={result['file_compression_ratio']:.2f}x "
            f"rawCR={result['raw_rgb_compression_ratio']:.2f}x "
            f"PSNR={result['psnr_dB']:.2f}dB "
            f"enc={result['enc_ms']['median']:.2f}ms "
            f"dec={result['dec_ms']['median']:.2f}ms"
        )

    summary = aggregate_dataset(image_results)

    print(
        f"---- DATASET {codec}/{quality}: "
        f"BPP={summary['bpp']:.5f} "
        f"fileCR={summary['compression_ratio']:.3f}x "
        f"rawCR={summary['raw_rgb_compression_ratio']:.3f}x "
        f"PSNR={summary['psnr_dB']:.3f}dB "
        f"enc={summary['enc_ms']['median']:.3f}ms "
        f"dec={summary['dec_ms']['median']:.3f}ms ----"
    )

    del net
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "summary": summary,
        "images": image_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kodak-dir",
        type=Path,
        default=Path("kodak"),
        help="Folder containing the Kodak images.",
    )
    parser.add_argument(
        "--codecs",
        nargs="+",
        default=["tcm", "bmshj2018-hyperprior"],
        choices=["tcm", "bmshj2018-hyperprior", "cheng2020"],
        help="One or more codecs to benchmark.",
    )
    parser.add_argument(
        "--qualities",
        nargs="+",
        default=["q4"],
        help=(
            "Quality labels to run for every selected codec. For cheng2020, "
            "q1 and 1 are both accepted."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="CODEC:QUALITY=PATH",
        help=(
            "Local checkpoint override. Repeat for each codec/quality, for "
            "example --checkpoint tcm:q4=/path/tcm.pth.tar"
        ),
    )
    parser.add_argument(
        "--tcm-repo",
        default="./LIC_TCM",
        help="Path to the LIC_TCM repository clone.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=64,
        help="TCM channels; must match the selected TCM checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip LPIPS and DISTS; PSNR and MS-SSIM are still computed.",
    )
    parser.add_argument(
        "--save-reconstructions",
        action="store_true",
        help="Save reconstructed PNG images for visual comparison.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("kodak_reconstructions"),
    )
    parser.add_argument(
        "--out",
        default="compression_kodak_results.json",
    )
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    checkpoint_overrides = parse_checkpoint_overrides(args.checkpoint)
    images = list_images(args.kodak_dir)
    device = args.device if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.benchmark = True

    lpips_fn = None
    dists_fn = None
    if not args.no_quality:
        import lpips
        import DISTS_pytorch as dists

        lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
        dists_fn = dists.DISTS().to(device).eval()

    print(
        f"Device: {device} | Kodak images: {len(images)} | "
        f"warmup={args.warmup} repeats={args.repeats}"
    )
    print(f"Codecs: {', '.join(args.codecs)}")
    print(f"Qualities: {', '.join(args.qualities)}")

    results = {
        "benchmark": "Kodak",
        "kodak_dir": str(args.kodak_dir),
        "num_images": len(images),
        "image_files": [path.name for path in images],
        "device": str(device),
        "precision": "fp16-transform" if args.fp16 else "fp32",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "compression_ratio_definition": (
            "total original encoded image-file bytes / total codec bitstream bytes"
        ),
        "raw_rgb_compression_ratio_definition": (
            "total H*W*3 raw RGB bytes / total codec bitstream bytes"
        ),
        "codecs": {},
    }

    for codec in args.codecs:
        codec_results = {
            "rate_points": {},
        }
        for quality in args.qualities:
            try:
                codec_results["rate_points"][quality] = benchmark_codec_quality(
                    codec=codec,
                    quality=quality,
                    images=images,
                    device=device,
                    checkpoint_overrides=checkpoint_overrides,
                    tcm_repo=args.tcm_repo,
                    N=args.N,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    fp16=args.fp16,
                    lpips_fn=lpips_fn,
                    dists_fn=dists_fn,
                    save_reconstructions=args.save_reconstructions,
                    save_dir=args.save_dir,
                )
            except (ValueError, FileNotFoundError) as error:
                codec_results["rate_points"][quality] = {
                    "status": "skipped",
                    "error": str(error),
                }
                print(f"SKIPPED {codec}/{quality}: {error}")

        results["codecs"][codec] = codec_results

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


# -----------------------------------------------------------------------------
# Self-test for pure helpers
# -----------------------------------------------------------------------------
def _selftest():
    assert padded_size(100, 100) == (128, 128)
    assert padded_size(90, 140) == (128, 192)
    assert padded_size(256, 256) == (256, 256)
    assert padded_size(1080, 1920) == (1088, 1920)

    parsed = parse_checkpoint_overrides([
        "tcm:q4=/tmp/tcm.pth.tar",
        "bmshj2018-hyperprior:q4=/tmp/bmshj.pth.tar",
    ])
    assert parsed[("tcm", "q4")] == "/tmp/tcm.pth.tar"
    assert parsed[("bmshj2018-hyperprior", "q4")] == "/tmp/bmshj.pth.tar"

    values = summarize([1.0, 2.0, 3.0])
    assert values["median"] == 2.0
    assert values["mean"] == 2.0
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in _sys.argv:
        _selftest()
    else:
        main()
