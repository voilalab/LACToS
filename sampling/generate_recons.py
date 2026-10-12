"""Sample DPS reconstructions of the FFHQ test images with the guided-diffusion FFHQ model.

    python sampling/generate_recons.py --recon_config configs/recon/adm_dps_ffhq_faces.yaml [--num_seeds N --samples ID ...]

The YAML holds the sampler (`additive_dps`: y = x + sigma n, DPS reverse process) and its
hyper-parameters, the checkpoint, the image directory, the sample ids and the output root.
CLI flags override the YAML. Output layout (read by evaluate.py):
    <out_root>/samples_<id>/adm_dps_whole_0_4/inpainting/{recon/<seed>_0_00000.png, label/0_00000.png, input/0_00000.png}
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ood.samplers import build_sampler, SAMPLERS
from ood.samplers.io import (
    image_to_tensor, load_adm_unet, prepare_sample_dirs,
    save_image_tensor,
)


def _parse_sampler_params(flat_kv_list):
    """Convert `--sampler_param k=v` repeats to a dict; cast numerics."""
    out: Dict[str, Any] = {}
    for kv in flat_kv_list or []:
        if "=" not in kv:
            raise ValueError(f"--sampler_param expects KEY=VAL, got {kv!r}")
        k, v = kv.split("=", 1)
        # Cast to int/float if possible; fall back to string.
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out


def _load_recon_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_config(args) -> Dict[str, Any]:
    """Merge YAML config + CLI overrides into one run config dict."""
    if args.recon_config:
        cfg = _load_recon_config(Path(args.recon_config))
    else:
        cfg = {
            "sampler": {"name": args.sampler, "params": _parse_sampler_params(args.sampler_param)},
            "ckpt": args.ckpt, "image_dir": args.image_dir,
            "samples": args.samples, "num_seeds": args.num_seeds,
            "out_root": args.out_root, "device": args.device,
            "test_origin_override": args.test_origin,
            "bottom_suffix": args.bottom_suffix,
        }

    # CLI always wins for these, when provided
    for k, cli_val in [
        ("ckpt", args.ckpt), ("image_dir", args.image_dir),
        ("num_seeds", args.num_seeds), ("out_root", args.out_root),
        ("device", args.device), ("bottom_suffix", args.bottom_suffix),
        ("batch_seeds", args.batch_seeds),
    ]:
        if cli_val is not None:
            cfg[k] = cli_val
    if args.samples:
        cfg["samples"] = args.samples
    if args.sampler:
        cfg.setdefault("sampler", {})["name"] = args.sampler
    if args.sampler_param:
        cfg.setdefault("sampler", {}).setdefault("params", {}).update(
            _parse_sampler_params(args.sampler_param)
        )
    if args.test_origin:
        cfg["test_origin_override"] = args.test_origin

    # Defaults (robust against explicit None values from YAML/CLI).
    cfg["num_seeds"] = cfg.get("num_seeds") or 20
    cfg["device"] = cfg.get("device") or "cuda:0"
    cfg["bottom_suffix"] = cfg.get("bottom_suffix") or "4"
    return cfg


def run(cfg: Dict[str, Any]) -> None:
    device = cfg["device"] if torch.cuda.is_available() else "cpu"

    print(f"[sampler]      {cfg['sampler']['name']}")
    print(f"[ckpt]         {cfg['ckpt']}")
    print(f"[image_dir]    {cfg['image_dir']}")
    print(f"[out_root]     {cfg['out_root']}")
    print(f"[num_seeds]    {cfg['num_seeds']}")

    # ADM / guided-diffusion checkpoint (the DPS2022 FFHQ model); `model_arch` names the preset in ood/samplers/io.py
    unet = load_adm_unet(cfg["ckpt"], device=device, arch=cfg.get("model_arch"))
    sampler_cfg = {**cfg["sampler"].get("params", {}), "device": device}
    sampler = build_sampler(cfg["sampler"]["name"], unet, sampler_cfg)
    meta = sampler.metadata

    test_origin = cfg.get("test_origin_override") or meta.test_origin
    print(f"[test_origin]  {test_origin}")
    print(f"[eff_sigma]    {meta.effective_sigma:.4f}")
    print(f"[hparams]      {meta.hyperparams}")

    out_root = Path(cfg["out_root"])
    image_dir = Path(cfg["image_dir"])
    samples = [str(s) for s in cfg["samples"]]
    # one process per GPU: each takes every num_shards-th image; the metadata below lists the whole set
    work = samples[int(cfg.get("shard", 0))::int(cfg.get("num_shards", 1))]
    print(f"[shard]        {cfg.get('shard', 0)}/{cfg.get('num_shards', 1)}: {len(work)} of {len(samples)} images")

    for sid in work:
        img_path = image_dir / f"{sid}.png"
        if not img_path.exists():
            print(f"MISSING: {img_path}")
            continue

        sample_name = f"samples_{sid}"
        dirs = prepare_sample_dirs(
            out_root, sample_name, test_origin, bottom_suffix=cfg["bottom_suffix"],
        )

        # Check if all seeds already exist
        num_seeds = int(cfg["num_seeds"])
        # Opt-in: also keep the unclamped float32 reconstruction. An 8-bit PNG
        # reorders DDAD's near-zero feature distances enough to cost it ~0.03 AP
        # (measured on shapes_L1: 0.629 float vs 0.600 PNG over 50 images).
        save_float = bool(cfg.get("save_float", False))
        float_dir = dirs["recon"].parent / "recon_float"
        done = lambda seed: (dirs["recon"] / f"{seed}_0_00000.png").exists() and (
            not save_float or (float_dir / f"{seed}_0_00000.npy").exists())
        existing_recons = [done(seed) for seed in range(num_seeds)]
        
        if all(existing_recons):
            print(f"[{sample_name}] Already fully sampled ({num_seeds} seeds). Skipping.")
            continue

        x = image_to_tensor(img_path, device=device)

        # label + input are the original image; some scorers read these.
        save_image_tensor(x[0], dirs["label"] / "0_00000.png")
        save_image_tensor(x[0], dirs["input"] / "0_00000.png")

        print(f"[{sample_name}]  N={num_seeds}")
        # Seeds are run in batches of `batch_seeds` (default: all missing seeds
        # at once) through Sampler.sample_batch; row k of a batch is the same
        # reconstruction `sample(x, seed_k)` would give (per-seed generators),
        # so partially finished images are completed consistently.
        batch_seeds = int(cfg.get("batch_seeds") or num_seeds)
        todo = [seed for seed in range(num_seeds) if not done(seed)]
        for i in range(0, len(todo), batch_seeds):
            chunk = todo[i:i + batch_seeds]
            recons = sampler.sample_batch(x, chunk)
            for seed, recon in zip(chunk, recons):
                save_image_tensor(recon, dirs["recon"] / f"{seed}_0_00000.png")
                if save_float:
                    float_dir.mkdir(parents=True, exist_ok=True)
                    np.save(float_dir / f"{seed}_0_00000.npy",
                            recon.detach().float().cpu().numpy().transpose(1, 2, 0))
            print(f"  seeds {chunk[0]}..{chunk[-1]} done ({i + len(chunk)}/{len(todo)} missing)")

    # Write σ + full metadata for downstream scorers and reproducibility.
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "sigma.txt", "w") as f:
        f.write(f"{meta.effective_sigma:.6f}\n")
    with open(out_root / "recon_metadata.json", "w") as f:
        json.dump({
            "sampler": meta.name,
            "test_origin": test_origin,
            "effective_sigma": meta.effective_sigma,
            "hyperparams": meta.hyperparams,
            "ckpt": cfg["ckpt"],
            "image_dir": cfg["image_dir"],
            "samples": samples,
            "num_seeds": cfg["num_seeds"],
            "bottom_suffix": cfg["bottom_suffix"],
        }, f, indent=2)
    print(f"\nWrote sigma.txt ({meta.effective_sigma:.4f}) and recon_metadata.json under {out_root}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recon_config", type=str, default=None,
                   help="YAML config driving the run (preferred). CLI flags override.")
    # CLI overrides / flat-mode fields
    p.add_argument("--sampler", choices=list(SAMPLERS.keys()), default=None)
    p.add_argument("--sampler_param", action="append", default=None,
                   help="Repeatable KEY=VAL override into sampler.params.")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--image_dir", type=str, default=None)
    p.add_argument("--samples", nargs="+", default=None)
    p.add_argument("--num_seeds", type=int, default=None)
    p.add_argument("--batch_seeds", type=int, default=None, help="seeds per batched sampler call (default: all)")
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--test_origin", type=str, default=None,
                   help="Override sampler's default output dirname segment.")
    p.add_argument("--bottom_suffix", type=str, default=None)
    p.add_argument("--shard", type=int, default=0, help="process images shard::num_shards of the sample list")
    p.add_argument("--num_shards", type=int, default=1, help="number of parallel runs sharing the sample list (one per GPU)")
    args = p.parse_args()

    cfg = _resolve_config(args)

    # Validate minimal required fields
    missing = [k for k in ("ckpt", "image_dir", "samples", "sampler") if not cfg.get(k)]
    if missing:
        p.error(f"missing required config fields: {missing}")

    if not 0 <= args.shard < args.num_shards:
        p.error("--shard must satisfy 0 <= shard < num_shards")
    cfg["shard"], cfg["num_shards"] = args.shard, args.num_shards
    run(cfg)


if __name__ == "__main__":
    main()
