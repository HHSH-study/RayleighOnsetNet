"""Run frozen RayleighOnsetNet inference on 100 Hz three-component velocity.

The numerical pipeline follows the original Discussion-case inference helper.
Input validation, portable paths, CPU selection, and safe checkpoint loading are
added; model architecture, EMA weights, normalization, and picking are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from rayleigh_onsetnet_model import CFG, RayleighOnsetNet, set_seed, soft_argmax_1d

ROOT = Path(__file__).resolve().parent
FS = 100
WINDOW_SAMPLES = 4000
TEMPERATURE = 0.7


def read_velocity(path: Path) -> np.ndarray:
    """Read dt/npts header and H1, H2, UD columns without signal processing."""
    with path.open(encoding="utf-8-sig") as handle:
        fields = handle.readline().split()
        if len(fields) != 2:
            raise ValueError(f"{path.name}: first line must be 'dt_s npts'.")
        dt, count = map(float, fields)
        if not np.isfinite(dt) or not np.isclose(dt, 1 / FS, rtol=1e-6, atol=1e-10):
            raise ValueError(f"{path.name}: expected dt=0.01 s (100 Hz), received {dt}.")
        if not np.isfinite(count) or count != int(count) or count < 2:
            raise ValueError(f"{path.name}: npts must be an integer >= 2.")
        data = np.loadtxt(handle, ndmin=2)
    if data.shape != (int(count), 3):
        raise ValueError(f"{path.name}: header expects {int(count)} rows and 3 columns, got {data.shape}.")
    if not np.isfinite(data).all():
        raise ValueError(f"{path.name}: non-finite waveform values.")
    ud = data[:, 2].astype(np.float32)
    if not np.isfinite(ud).all() or not np.isfinite(ud.std()) or ud.std() == 0:
        raise ValueError(f"{path.name}: UD component is constant or outside the supported numerical range.")
    return (ud - ud.mean()) / (ud.std() + 1e-8)


def load_network(path: Path, device: torch.device) -> RayleighOnsetNet:
    # weights_only avoids arbitrary pickle execution. Use the supplied files.
    obj = torch.load(path, map_location=device, weights_only=True)
    cfg = CFG()
    expected = {"fs": 100, "p_min": 2.0, "p_max": 10.0, "num_scales": 48,
                "temperature": 0.7, "refine_window_sec": 40.0}
    for key, value in expected.items():
        if obj.get("cfg", {}).get(key) != value:
            raise ValueError(f"{path.name}: incompatible checkpoint configuration: {key}.")
    network = RayleighOnsetNet(cfg).to(device)
    network.load_state_dict(obj["model"], strict=True)
    ema = obj.get("ema")
    names = {name for name, _ in network.named_parameters()}
    if not isinstance(ema, dict) or set(ema) != names:
        raise ValueError(f"{path.name}: missing or incompatible EMA inference weights.")
    with torch.no_grad():
        for name, param in network.named_parameters():
            if param.shape != ema[name].shape:
                raise ValueError(f"{path.name}: incompatible EMA tensor {name}.")
            param.copy_(ema[name])
    return network.eval()


@torch.no_grad()
def predict_one(waveform: np.ndarray, coarse, refine, device) -> tuple[float, float]:
    wave = torch.tensor(waveform, dtype=torch.float32, device=device).view(1, 1, -1)
    coarse_index = soft_argmax_1d(coarse(wave), temperature=TEMPERATURE).item()
    start = int(round(coarse_index - WINDOW_SAMPLES / 2))
    end = start + WINDOW_SAMPLES
    record_len = wave.shape[-1]
    segment = wave[0, 0, max(0, start):min(record_len, end)]
    if start < 0:
        segment = torch.cat([torch.zeros(-start, device=device), segment])
    if end > record_len:
        segment = torch.cat([segment, torch.zeros(end - record_len, device=device)])
    if len(segment) > WINDOW_SAMPLES:
        segment = segment[:WINDOW_SAMPLES]
    elif len(segment) < WINDOW_SAMPLES:
        segment = torch.cat([segment, torch.zeros(WINDOW_SAMPLES - len(segment), device=device)])
    local_index = soft_argmax_1d(refine(segment.view(1, 1, WINDOW_SAMPLES)),
                                 temperature=TEMPERATURE).item()
    return coarse_index / FS, (start + local_index) / FS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="One .v file or a directory containing .v files.")
    parser.add_argument("--output", required=True, type=Path, help="Destination predictions CSV (must not exist).")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads; default: 4.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists():
        parser.error("Output already exists; choose another filename to preserve earlier results.")
    paths = sorted(args.input.glob("*.v")) if args.input.is_dir() else [args.input]
    if not paths or any(not p.is_file() or p.suffix.lower() != ".v" for p in paths):
        parser.error("Input must contain one or more .v files.")
    set_seed(42)
    torch.set_num_threads(args.threads)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu.")
    device = torch.device(device_name)
    coarse = load_network(ROOT / "checkpoints/coarse_best.pth", device)
    refine = load_network(ROOT / "checkpoints/refine_best.pth", device)
    rows = []
    for number, path in enumerate(paths, 1):
        waveform = read_velocity(path)
        tc, tr = predict_one(waveform, coarse, refine, device)
        if not np.isfinite([tc, tr]).all():
            raise RuntimeError(f"{path.name}: model returned non-finite times.")
        # Preserve the model output. Flag, rather than clip, boundary estimates.
        status = "ok" if 0 <= tr <= (len(waveform) - 1) / FS else "outside_record"
        rows.append({"filename": path.name, "npts": len(waveform), "sampling_rate_hz": FS,
                     "coarse_s": tc, "refined_s": tr, "status": status})
        print(f"[{number}/{len(paths)}] {path.name}: coarse={tc:.3f} s, refined={tr:.3f} s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"records": len(rows), "device": str(device),
                      "output": str(args.output), "numpy": np.__version__,
                      "torch": torch.__version__}, ensure_ascii=False))


if __name__ == "__main__":
    main()
