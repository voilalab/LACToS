"""Evaluate OOD detection — category/subcategory aware.

Directory convention:
    {results_root}_{category}_{subcategory}/
        samples_000/ ...
        samples_001/ ...

GT mask convention (tried in order):
    {gt_root}/{category}/{subcategory}/{sample_id}_mask.png
    {gt_root}/{subcategory}/{sample_id}_mask.png

Usage:
    # Single subcategory, specific samples
    python evaluate.py --config configs/experiment_ddad_native.yaml \\
        --category cable --subcategory missing_wire \\
        --sample_names samples_000 samples_001

    # All subcategories, auto-discover all samples (cap at 10 each for speed)
    python evaluate.py --config configs/experiment_ddad_native.yaml \\
        --category cable --all_subcategories \\
        --results_root ./results_patches_ddad_native \\
        --max_samples 10

    # AE mode
    python evaluate.py --config configs/experiment_ddad_native.yaml \\
        --category cable --subcategory missing_wire \\
        --autoencoder_path /models/ae_cable_3.pth --n_pca 3
"""

import argparse, json, os, re, subprocess, sys
from copy import deepcopy
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ood.data import (load_gt_mask, load_label_image, load_reconstructions,
                      load_superpixel_mask, parse_sample_id)
from ood.superpixels import recursive_subdivide
from ood.embeddings import ResNetPixelEmbedder, embed_and_project
from ood.scoring import compute_delta_map_gpu
from ood.scoring_local_gaussian import compute_delta_map_local_gaussian
from ood.metrics import evaluate_delta_map
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── PixelAutoEncoder ─────────────────────────────────────────────────────────
class PixelAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim=1417, latent_dim=5):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim,512,1,bias=False),torch.nn.BatchNorm2d(512),torch.nn.ReLU(True),
            torch.nn.Conv2d(512,256,1,bias=False),torch.nn.BatchNorm2d(256),torch.nn.ReLU(True),
            torch.nn.Conv2d(256,64,1,bias=False),torch.nn.BatchNorm2d(64),torch.nn.ReLU(True),
            torch.nn.Conv2d(64,latent_dim,1),torch.nn.Tanh())
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(latent_dim,64,1,bias=False),torch.nn.BatchNorm2d(64),torch.nn.ReLU(True),
            torch.nn.Conv2d(64,256,1,bias=False),torch.nn.BatchNorm2d(256),torch.nn.ReLU(True),
            torch.nn.Conv2d(256,512,1,bias=False),torch.nn.BatchNorm2d(512),torch.nn.ReLU(True),
            torch.nn.Conv2d(512,input_dim,1))
    def forward(self,x):
        z=self.encoder(x); return self.decoder(z),z

def load_autoencoder(ae_path, embedder, latent_dim, device):
    with torch.no_grad():
        embed_dim = embedder(torch.zeros(1,3,256,256).to(device)).shape[1]
    ae = PixelAutoEncoder(input_dim=embed_dim, latent_dim=latent_dim).to(device)
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()
    print(f"  AE loaded: {ae_path}  (embed_dim={embed_dim}, latent_dim={latent_dim})")
    return ae

_FOLD_CHUNK = 64


def _fold_plan(ae, embedder):
    """Per-layer slices of the AE's first 1x1 conv for the folded encoder, or None if unsupported."""
    enc0 = ae.encoder[0]
    if not (isinstance(enc0, torch.nn.Conv2d) and enc0.kernel_size == (1, 1)
            and enc0.bias is None and enc0.groups == 1):
        return None
    if embedder.use_patch_context and embedder.patchify_size % 2 == 0:
        return None
    with torch.no_grad():
        probe = torch.zeros(1, 3, 256, 256, device=enc0.weight.device)
        feats = embedder.extractor(probe)
        chans = []
        for ln in embedder.layers:
            f = feats[ln]
            if embedder.proj_dim_per_layer is not None:
                f = embedder.proj[ln](f)
            chans.append(int(f.shape[1]))
    if sum(chans) != enc0.weight.shape[1]:
        return None
    W, off, plan = enc0.weight.data, 0, []
    for c in chans:
        plan.append(W[:, off:off + c].contiguous())
        off += c
    return plan


def _encode_folded(ae, embedder, x, plan):
    """ae.encoder(embedder(x)) with the first 1x1 conv applied per layer at native resolution.

    Mathematically identical to the direct path, but a different float32 evaluation order:
    the two paths differ by ~1e-6 in the latents, which is enough to move a few latent
    values across histogram-bin edges. The CHAOS row of the paper was computed with
    THIS path (akheirandish3's package); FFHQ with the direct path. `--ae_encode` selects.
    """
    if x.min() < 0:
        x = (x + 1) / 2.0
    if embedder.use_imagenet_norm:
        x = (x - embedder.mean) / embedder.std
    feats = embedder.extractor(x)
    H, W_ = x.shape[-2:]
    out_hw = ((embedder.out_size, embedder.out_size)
              if embedder.out_size is not None else (H, W_))
    acc = None
    for li, ln in enumerate(embedder.layers):
        f = feats[ln]
        if embedder.use_patch_context:
            f = F.avg_pool2d(f, embedder.patchify_size, stride=1,
                             padding=(embedder.patchify_size - 1) // 2,
                             count_include_pad=True)
        if embedder.proj_dim_per_layer is not None:
            f = embedder.proj[ln](f)
        proj = F.interpolate(F.conv2d(f, plan[li]), size=out_hw,
                             mode="bilinear", align_corners=False)
        sq = torch.zeros(f.shape[0], 1, *out_hw, device=f.device, dtype=torch.float32)
        for c0 in range(0, f.shape[1], _FOLD_CHUNK):
            u = F.interpolate(f[:, c0:c0 + _FOLD_CHUNK], size=out_hw,
                              mode="bilinear", align_corners=False)
            sq += u.float().pow(2).sum(1, keepdim=True)
        n = sq.clamp_min(1e-24).sqrt()      # == F.normalize's denominator (eps=1e-12)
        acc = proj / n if acc is None else acc + proj / n
    return ae.encoder[1:](acc)


def embed_with_ae(ae, embedder, label_image, images_recon_all, batch_size, device, ae_encode="direct"):
    """ae_encode: 'direct' = ae(embedder(x)) (FFHQ row of the paper); 'folded' = _encode_folded (CHAOS row)."""
    from ood.embeddings import to_tensor01
    if ae_encode == "folded":
        plan = _fold_plan(ae, embedder)
        if plan is None:
            raise ValueError("--ae_encode folded needs an AE whose first layer is a bias-free 1x1 conv")
        encode = lambda t: _encode_folded(ae, embedder, t, plan)
    elif ae_encode == "direct":
        encode = lambda t: ae(embedder(t))[1]
    else:
        raise ValueError(f"unknown ae_encode {ae_encode!r}")
    with torch.no_grad():
        label_z = encode(to_tensor01(label_image, device=str(device)))
    label_pca_map = label_z.squeeze(0).permute(1,2,0).cpu().numpy()
    x_all = torch.from_numpy(images_recon_all).float().permute(0,3,1,2)
    if x_all.max() > 1.0: x_all = x_all / 255.0
    pca_list = []
    with torch.no_grad():
        for i in range(0, x_all.shape[0], batch_size):
            bz = encode(x_all[i:i+batch_size].to(device))
            pca_list.append(bz.cpu())
    return {"label_pca_map": label_pca_map,
            "pca_feats_recon": torch.cat(pca_list,0).permute(0,2,3,1).numpy()}


# ── Category / subcategory path helpers ──────────────────────────────────────

def resolve_results_dir(cfg, category, subcategory):
    d = cfg["data"]
    if "results_root" in d:
        suffix = d.get("results_suffix", "")
        return f"{d['results_root']}_{category}_{subcategory}{suffix}"
    return d["results_dir"]

def resolve_gt_mask_template(cfg, category, subcategory):
    """Returns a path template containing {sample} for use with load_gt_mask."""
    d = cfg["data"]
    if "gt_root" in d:
        root = d["gt_root"]
        for cand in [
            os.path.join(root, category, subcategory, "{sample}_mask.png"),
            os.path.join(root, subcategory, "{sample}_mask.png"),
            os.path.join(root, "{sample}_mask.png"),
        ]:
            parent = os.path.dirname(cand.replace("{sample}", "PLACEHOLDER"))
            if os.path.isdir(parent):
                return cand
        return os.path.join(root, category, subcategory, "{sample}_mask.png")
    return d["gt_mask"]["path"]

def discover_subcategories(results_root, category, suffix=""):
    prefix  = f"{results_root}_{category}_"
    matches = sorted(glob(f"{prefix}*{suffix}"))
    out = []
    for m in matches:
        if not os.path.isdir(m): continue
        subcat = m[len(prefix):]
        if suffix and subcat.endswith(suffix):
            subcat = subcat[:-len(suffix)]
        if subcat:
            out.append(subcat)
    return out

def discover_samples(results_dir):
    if not os.path.isdir(results_dir):
        return []
    return sorted(e for e in os.listdir(results_dir)
                  if os.path.isdir(os.path.join(results_dir, e)) and e.startswith("samples_"))

def resolve_figures_dir(cfg, category, sample_name):
    d = cfg["data"]
    if "figures_root" in d:
        return os.path.join(d["figures_root"], category, sample_name)
    return os.path.join(d["figures_dir"], sample_name)


# ── Config ───────────────────────────────────────────────────────────────────
def load_config(path):
    with open(path) as f: return yaml.safe_load(f)


# ── Single-sample evaluation ─────────────────────────────────────────────────
def _run_single_sample(cfg):
    data_cfg    = cfg["data"]
    sample_name = data_cfg["sample_name"]
    sample_id   = parse_sample_id(sample_name)
    device      = torch.device(cfg["embeddings"].get("device","cuda"))
    score_cfg   = cfg["scoring"]
    figures_dir = data_cfg.get("_figures_dir",
                               os.path.join(data_cfg["figures_dir"], sample_name))

    print(f"\n  [{sample_name}]")

    recon_all   = load_reconstructions(results_dir=data_cfg["results_dir"],
                      sample_name=sample_name, test_origin=data_cfg["test_origin"],
                      num_patches=cfg["sampling"].get("num_patches",24),
                      bottom_suffix=data_cfg["bottom_suffix"])
    label_image = load_label_image(results_dir=data_cfg["results_dir"],
                      sample_name=sample_name, test_origin=data_cfg["test_origin"],
                      bottom_suffix=data_cfg["bottom_suffix"])
    mask_path = os.path.join(figures_dir, "mask.png")
    max_recon = cfg.get("sampling", {}).get("max_reconstructions", None)
    if max_recon is not None:
        n_total = len(recon_all)
        if n_total > max_recon:
            idx = np.sort(np.random.choice(n_total, max_recon, replace=False))
            recon_all = recon_all[idx]
            print(f"  Reconstructions: using {max_recon}/{n_total} (subsampled)")
        else:
            print(f"  Reconstructions: using all {n_total} (max_reconstructions={max_recon} not reached)")
    if not os.path.exists(mask_path):
        inp = os.path.join(data_cfg.get("image_dir", ""), f"{sample_id}.png")

        print(f"  Superpixel mask missing: {mask_path}")
        print(f"  Trying to generate from image: {inp}")

        if not os.path.exists(inp):
            raise FileNotFoundError(
                f"Input image for superpixel generation not found: {inp}\n"
                f"Set cfg['data']['image_dir'] correctly for this category/subcategory."
            )

        os.makedirs(figures_dir, exist_ok=True)
        subprocess.run(
            [sys.executable, "super_pixel_generation.py",
             f"--input_image={inp}", f"--output_dir={figures_dir}"],
            check=True,
        )

    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"super_pixel_generation.py finished but did not create: {mask_path}"
        )

    sp_mask     = load_superpixel_mask(figures_dir)
    gt_mask     = load_gt_mask(path_template=data_cfg["gt_mask"]["path"],
                               sample=sample_id,
                               downsample_factor=data_cfg["gt_mask"]["downsample_factor"])

    category = data_cfg.get("category", "cable")
    valid_mask = None

    if category == "CT":
        import re
        import PIL.Image as Image

        m = re.search(r"samples_(?:image_)?0*(\d+)", sample_name)
        if m is None:
            raise ValueError(f"Could not parse CT sample number from {sample_name}")

        s_int = str(int(m.group(1)))
        s_pad4 = f"{int(m.group(1)):04d}"

        # CT lesion + body masks: OOD_CT_MASK_ROOT/{masks_ood_med_med,masks_body_ood_default_large}/image_XXXX_mask.png
        _ct_root = os.environ.get("OOD_CT_MASK_ROOT", "/data2/akheirandish3/id_new_warped_images")
        mask_dirs = [
            os.path.join(_ct_root, "masks_ood_med_med"),
        ]

        candidates = []
        for md in mask_dirs:
            candidates.extend([
                os.path.join(md, f"image_{s_pad4}_mask.png"),
                os.path.join(md, f"image_{s_int}_mask.png"),
                os.path.join(md, f"{s_pad4}_mask.png"),
                os.path.join(md, f"{s_int}_mask.png"),
            ])

        gt_path = next((p for p in candidates if os.path.exists(p)), None)
        gray_scale = True
        if gt_path is None:
            print("Tried these CT GT mask paths:")
            for p in candidates:
                print("  ", p)
            raise FileNotFoundError(f"CT GT mask not found for {sample_name}")

        print(f"  CT GT mask: {gt_path}")
        mask = np.array(Image.open(gt_path).convert("L"))[::2, ::2]
        gt_mask = (mask > 0).astype(np.uint8)

        body_candidates = [
            os.path.join(_ct_root, "masks_body_ood_default_large", f"image_{s_pad4}_mask.png"),
            os.path.join(_ct_root, "masks_body_ood_default_large", f"image_{s_int}_mask.png"),
        ]
        body_p = next((p for p in body_candidates if os.path.exists(p)), None)
        if body_p is not None:
            body = np.array(Image.open(body_p).convert("L")) > 0
            valid_mask = body.astype(np.uint8)

    else:
        gt_mask = load_gt_mask(
            path_template=data_cfg["gt_mask"]["path"],
            sample=sample_id,
            downsample_factor=data_cfg["gt_mask"]["downsample_factor"],
        )
        gray_scale = False

    # Superpixel refinement
    sp_cfg = cfg["superpixels"]
    labels_fine, final_ids, parent_map = recursive_subdivide(
        img=label_image, labels=sp_mask,
        var_threshold=sp_cfg["var_threshold"], min_pixels=sp_cfg["min_pixels"],
        max_sub=sp_cfg["max_sub"], max_depth=sp_cfg["max_depth"],
        compactness=sp_cfg["compactness"], alpha_grad=sp_cfg["alpha_grad"],
        target_size=sp_cfg["target_size"])
    print(f"  {len(np.unique(sp_mask))} -> {len(np.unique(final_ids))} superpixels")
    _plot_superpixels(label_image, labels_fine, sp_mask,
                    save_path=os.path.join(figures_dir, f"{sample_name}_superpixels.png"))

    # Embed
    embed_cfg  = cfg["embeddings"]
    latent_dim = cfg["pca"]["n_components"]
    ae_path    = score_cfg.get("autoencoder_path","")
    embedder   = ResNetPixelEmbedder(
        resnet_name=embed_cfg["backbone"], layers=tuple(embed_cfg["layers"]),
        use_patch_context=embed_cfg["use_patch_context"],
        patchify_size=embed_cfg.get("patchify_size",3),
        proj_dim_per_layer=embed_cfg.get("proj_dim_per_layer")).to(device).eval()

    use_pca    = bool(score_cfg.get("use_pca", True))
    if not use_pca:
        # RGB-only ablation (--no_pca). The feature term enters the score
        # additively and compute_delta_map_gpu zeroes it, so the features are
        # never used; feed one constant channel instead of computing them.
        h, w = label_image.shape[:2]
        embed_result = {
            "pca_feats_recon": np.zeros((recon_all.shape[0], h, w, 1), np.float32),
            "label_pca_map":   np.zeros((h, w, 1), np.float32),
        }
        print("  RGB-only scoring: feature/autoencoder term disabled")
    elif ae_path:
        ae_model     = load_autoencoder(ae_path, embedder, latent_dim, device)
        embed_result = embed_with_ae(ae_model, embedder, label_image, recon_all,
                                     embed_cfg.get("batch_size",16), device,
                                     ae_encode=score_cfg.get("ae_encode", "direct"))
        del ae_model
    else:
        embed_result = embed_and_project(embedder=embedder, label_image=label_image,
            images_recon_all=recon_all, n_pca=latent_dim, device=str(device))
    del embedder; torch.cuda.empty_cache()

    # Score
    algorithm  = score_cfg.get("algorithm","typical_set")
    if algorithm == "local_gaussian":
        sigma_r = score_cfg.get("sigma_rohan")
        if sigma_r is None:
            sp = os.path.join(data_cfg["results_dir"],"sigma.txt")
            sigma_r = float(open(sp).read().strip()) if os.path.exists(sp) else None
        if sigma_r is None: raise ValueError("local_gaussian scorer needs --sigma_rohan")
        delta_map,_,labels_used = compute_delta_map_local_gaussian(
            labels_fine=sp_mask,
            parent_map={int(s):[int(s)] for s in np.unique(sp_mask)},
            images_recon_all=recon_all, label_image=label_image,
            sigma=sigma_r, n_realizations=score_cfg.get("n_realizations",1000),
            min_pixels=score_cfg["min_pixels"], metric=score_cfg.get("lg_metric","typicality_unsigned"))
    else:
        delta_map,_,labels_used = compute_delta_map_gpu(
            labels_fine=labels_fine, images_recon_all=recon_all,
            pca_feats_recon=embed_result["pca_feats_recon"],
            label_image=label_image, label_pca_map=embed_result["label_pca_map"],
            bins_rgb=score_cfg["bins_rgb"], bins_pca=score_cfg["bins_pca"],
            smooth_sigma=score_cfg["smooth_sigma"], min_pixels=score_cfg["min_pixels"],
            device=device, gray_scale=gray_scale, use_pca=use_pca)

    print(f"  Scored {len(labels_used)} SPs [{algorithm}]")

    # ── Shape reconciliation ──────────────────────────────────────────────
    if gt_mask.shape != delta_map.shape:
        from skimage.transform import resize as _resize
        print(f"  [WARN] Shape mismatch: delta_map={delta_map.shape} "
              f"gt_mask={gt_mask.shape} — resizing gt_mask to match delta_map.")
        gt_mask = _resize(
            gt_mask.astype(np.float32), output_shape=delta_map.shape,
            order=0, mode="edge", preserve_range=True, anti_aliasing=False,
        ).astype(np.uint8)

    if labels_fine.shape != delta_map.shape:
        print(f"  [WARN] labels_fine shape {labels_fine.shape} != "
              f"delta_map shape {delta_map.shape} — resizing labels_fine.")
        from skimage.transform import resize as _resize
        labels_fine = _resize(
            labels_fine.astype(np.float32), output_shape=delta_map.shape,
            order=0, mode="edge", preserve_range=True, anti_aliasing=False,
        ).astype(np.int32)

    if valid_mask is not None and valid_mask.shape != delta_map.shape:
        from skimage.transform import resize as _resize
        print(f"  [WARN] valid_mask shape mismatch, resizing to {delta_map.shape}.")
        valid_mask = _resize(
            valid_mask.astype(np.float32), output_shape=delta_map.shape,
            order=0, mode="edge", preserve_range=True, anti_aliasing=False,
        ).astype(np.uint8)

    eval_cfg = cfg["eval"]

    # ── Optional raw-map dump (--dump_maps) ───────────────────────────────
    # Schema is byte-compatible with the 2026-07-16 dumps under
    # results_eval/heatmaps/maps/adm_dps/: delta_map (float32, may contain
    # NaN), gt_mask (uint8), label_image (uint8), valid_mask. `valid_mask` is
    # written as an EMPTY array when there is no valid mask — that is the
    # existing convention (tools/uniform_metrics_from_maps.py indexes the key
    # unconditionally, and full_metric_table.py tests its shape), and keeping
    # it means new dumps are pooled on exactly the same terms as the old ones.
    # Written BEFORE the metric loop on purpose: on an anomaly-free image the
    # per-image metrics are meaningless, but the map is still wanted.
    dump_dir = eval_cfg.get("dump_maps_dir")
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        dump_path = os.path.join(dump_dir, f"{sample_name}.npz")
        np.savez_compressed(
            dump_path,
            delta_map=np.asarray(delta_map, dtype=np.float32),
            gt_mask=(gt_mask > 0).astype(np.uint8),
            label_image=np.asarray(label_image, dtype=np.uint8),
            valid_mask=(np.zeros(0, dtype=np.uint8) if valid_mask is None
                        else valid_mask.astype(np.uint8)),
        )
        print(f"  Dumped map: {dump_path}")

    # _save_heatmap(
    #     label_image=label_image,
    #     delta_map=delta_map,
    #     gt_mask=gt_mask,
    #     figures_dir=figures_dir,
    #     sample_name=sample_name,
    #     valid_mask=valid_mask,
    # )

    # Evaluate
    all_metrics = {}
    for sigma in eval_cfg["delta_smooth_sigmas"]:
        key    = f"sigma_{sigma}" if sigma else "raw"
        result = evaluate_delta_map(delta_map=delta_map, labels_fine=labels_fine,
            gt_mask_binary=gt_mask, anomaly_threshold=eval_cfg["sp_anomaly_threshold"],
            smooth_sigma=sigma, valid_mask=valid_mask)
        all_metrics[key] = result
        psnr_str = (f"{result['psnr']:.2f}" if np.isfinite(result.get('psnr', float('nan')))
                    else str(result.get('psnr', 'nan')))
        print(f"  [{key}] SP={result['sp_roc_auc']:.4f} Px={result['px_roc_auc']:.4f} "
              f"SP_AP={result['sp_ap']:.4f} Px_AP={result['px_ap']:.4f} "
              f"SNR={result['snr']:.2f} PSNR={psnr_str}")

    return all_metrics


# ── Subcategory runner ────────────────────────────────────────────────────────
def _run_subcategory(base_cfg, category, subcategory, sample_names, output_dir):
    print(f"\n{'═'*60}")
    print(f"  {category} / {subcategory}")
    print(f"{'═'*60}")

    cfg = deepcopy(base_cfg)
    results_dir = resolve_results_dir(base_cfg, category, subcategory)
    gt_template = resolve_gt_mask_template(base_cfg, category, subcategory)

    cfg["data"]["results_dir"]  = results_dir
    cfg["data"]["category"]     = category
    cfg["data"]["subcategory"]  = subcategory
    cfg["data"]["figures_dir"]  = os.path.join(base_cfg["data"]["figures_dir"], category, subcategory)
    if base_cfg["data"].get("image_dir_fixed"):
        print(f"  Image dir    : {cfg['data']['image_dir']} (--image_dir)")
    elif category != "CT":
        category_root = os.path.dirname(cfg["data"]["gt_root"])
        cfg["data"]["image_dir"] = os.path.join(category_root, "test", subcategory)
        print(f"  Image dir    : {cfg['data']['image_dir']}")
    if category != "CT":
        cfg["data"]["gt_mask"]["path"] = gt_template

    # Auto-discover samples; honour --max_samples for fast sweeps
    names = sample_names or discover_samples(results_dir)
    max_s = base_cfg.get("eval", {}).get("max_samples", None)
    if max_s is not None and len(names) > max_s:
        names = names[:max_s]
        print(f"  [max_samples={max_s}] processing first {max_s} samples for speed")

    if not names:
        print(f"  [WARN] No samples found in {results_dir}")
        return {}
    print(f"  Samples ({len(names)}): {names[:5]}{'...' if len(names)>5 else ''}")
    print(f"  Results dir : {results_dir}")
    print(f"  GT template : {gt_template}")

    results, failed = {}, []
    for sname in names:
        c = deepcopy(cfg)
        c["data"]["sample_name"] = sname
        c["data"]["_figures_dir"] = os.path.join(cfg["data"]["figures_dir"], sname)
        try:
            results[sname] = _run_single_sample(c)
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {sname}: {exc}")
            traceback.print_exc()
            failed.append(sname)

    # Average & save
    first_keys = list(next(iter(results.values())).keys()) if results else []
    averaged = {}
    if results:
        print(f"\n  AVERAGES — {subcategory} ({len(results)} samples):")
        for key in first_keys:
            vals = [results[s][key] for s in results if key in results[s]]
            averaged[key] = {
                "sp_roc_auc": float(np.mean([v["sp_roc_auc"] for v in vals])),
                "px_roc_auc": float(np.mean([v["px_roc_auc"] for v in vals])),
                "sp_ap":      float(np.mean([v["sp_ap"]       for v in vals])),
                "px_ap":      float(np.mean([v["px_ap"]       for v in vals])),
                "snr":        float(np.nanmean([v.get("snr", np.nan) for v in vals])),
                "psnr":       float(np.nanmean([v.get("psnr", np.nan) for v in vals])),
            }
            a = averaged[key]
            print(f"    [{key}] SP={a['sp_roc_auc']:.4f} Px={a['px_roc_auc']:.4f} "
                  f"SP_AP={a['sp_ap']:.4f} Px_AP={a['px_ap']:.4f} "
                  f"SNR={a['snr']:.2f} PSNR={_fmt_psnr(a['psnr'])}")

    os.makedirs(output_dir, exist_ok=True)
    ae_path = base_cfg["scoring"].get("autoencoder_path","")
    ae_tag  = f"ae_{os.path.splitext(os.path.basename(ae_path))[0]}_" if ae_path else ""
    jpath   = os.path.join(output_dir, f"{ae_tag}{category}_{subcategory}.json")
    with open(jpath,"w") as f:
        json.dump({"category":category,"subcategory":subcategory,
                   "results_dir":results_dir,"gt_template":gt_template,
                   "samples":list(results),"failed":failed,
                   "per_sample":{s:{k:{kk:vv for kk,vv in v.items() if kk!="curves"}
                                    for k,v in results[s].items()} for s in results},
                   "averaged":averaged}, f, indent=2, default=str)
    print(f"  Saved: {jpath}")
    return results


# ── Category summary ──────────────────────────────────────────────────────────
def _summarize_category(all_subcat_results, category, output_dir, base_cfg):
    print(f"\n{'═'*60}")
    print(f"CATEGORY SUMMARY: {category}  ({len(all_subcat_results)} subcategories)")
    print(f"{'═'*60}")
    rows = []
    for subcat, sample_results in all_subcat_results.items():
        if not sample_results: continue
        first_keys = list(next(iter(sample_results.values())).keys())
        for key in first_keys:
            vals = [sample_results[s][key] for s in sample_results if key in sample_results[s]]
            if not vals: continue
            rows.append({"subcategory":subcat,"eval_sigma":key,"n":len(sample_results),
                "sp_roc_auc":float(np.mean([v["sp_roc_auc"] for v in vals])),
                "px_roc_auc":float(np.mean([v["px_roc_auc"] for v in vals])),
                "sp_ap":float(np.mean([v["sp_ap"] for v in vals])),
                "px_ap":float(np.mean([v["px_ap"] for v in vals])),
                "psnr":float(np.nanmean([v.get("psnr", np.nan) for v in vals])),
                "snr":float(np.nanmean([v.get("snr", np.nan) for v in vals]))})
            r = rows[-1]
            print(f"  {subcat:<22} [{key}]  SP={r['sp_roc_auc']:.4f}  "
                  f"Px={r['px_roc_auc']:.4f}  SP_AP={r['sp_ap']:.4f}  "
                  f"Px_AP={r['px_ap']:.4f}  PSNR={_fmt_psnr(r['psnr'])} "
                  f"SNR={r['snr']:.2f} (n={r['n']})")
    if rows:
        overall_psnr = np.nanmean([float(r.get('psnr', np.nan)) for r in rows
                                   if np.isfinite(r.get('psnr', float('nan')))])
        print(f"\n  OVERALL  SP={np.mean([r['sp_roc_auc'] for r in rows]):.4f}  "
              f"Px={np.mean([r['px_roc_auc'] for r in rows]):.4f} "
              f"SNR={np.nanmean([float(r.get('snr', np.nan)) for r in rows]):.2f} "
              f"PSNR={_fmt_psnr(overall_psnr)}")
    ae_path = base_cfg["scoring"].get("autoencoder_path","")
    ae_tag  = f"ae_{os.path.splitext(os.path.basename(ae_path))[0]}_" if ae_path else ""
    jpath   = os.path.join(output_dir, f"{ae_tag}{category}_summary.json")
    with open(jpath,"w") as f:
        json.dump({"category":category,"rows":rows}, f, indent=2, default=str)
    print(f"Summary saved: {jpath}")

def _fmt_psnr(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return str(v)
    return f"{v:.2f}"


# ── Heatmap / figure saving ───────────────────────────────────────────────────
def _save_heatmap(
    label_image,
    delta_map,
    gt_mask,
    figures_dir,
    sample_name,
    valid_mask=None,
    save_npy=False,
    cmap="inferno",          # perceptually-uniform, prints well in greyscale
):
    import PIL.Image as _PIL
    from skimage.transform import resize

    os.makedirs(figures_dir, exist_ok=True)

    heat = np.asarray(delta_map, dtype=np.float32)
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Resize label image to match heatmap if needed ────────────────────────
    img = label_image
    if img.shape[:2] != heat.shape:
        img = resize(
            img,
            output_shape=heat.shape + (() if img.ndim == 2 else (img.shape[2],)),
            order=1, preserve_range=True, anti_aliasing=True,
        ).astype(label_image.dtype)

    # ── Save raw label image as PNG ──────────────────────────────────────────
    img_save = img.copy()
    if img_save.dtype != np.uint8:
        lo_, hi_ = img_save.min(), img_save.max()
        img_save = (((img_save - lo_) / (hi_ - lo_ + 1e-8)) * 255).astype(np.uint8)
    label_img_path = os.path.join(figures_dir, f"{sample_name}_label_image.png")
    mode = "L" if img_save.ndim == 2 else "RGB"
    _PIL.fromarray(img_save, mode=mode).save(label_img_path)
    print(f"  Saved label image    : {label_img_path}")

    # ── Robust percentile normalisation ─────────────────────────────────────
    region = heat[valid_mask.astype(bool)] if valid_mask is not None else heat.ravel()
    region = region[np.isfinite(region)]
    lo, hi = (np.percentile(region, [1, 99]) if len(region) > 0
              else (float(heat.min()), float(heat.max())))
    heat_norm = np.clip((heat - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    # ── Optional .npy ────────────────────────────────────────────────────────
    if save_npy:
        np.save(os.path.join(figures_dir, f"{sample_name}_delta_map.npy"), heat)

    # ── Heatmap-only PNG ─────────────────────────────────────────────────────
    heat_only_path = os.path.join(figures_dir, f"{sample_name}_heatmap_only.png")
    plt.imsave(heat_only_path, heat_norm, cmap=cmap)
    print(f"  Saved heatmap only   : {heat_only_path}")

    # ── Clean alpha-blended overlay ──────────────────────────────────────────
    cm_obj   = plt.get_cmap(cmap)
    heat_rgb = cm_obj(heat_norm)[..., :3]           # (H,W,3) float
    img_f    = (img.astype(np.float32) / 255.0 if img.max() > 1.0
                else img.astype(np.float32))
    if img_f.ndim == 2:
        img_f = np.repeat(img_f[..., None], 3, axis=2)
    alpha   = 0.55
    overlay = np.clip((1 - alpha) * img_f + alpha * heat_rgb, 0.0, 1.0)

    overlay_path = os.path.join(figures_dir, f"{sample_name}_overlay.png")
    _PIL.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)
    print(f"  Saved overlay PNG    : {overlay_path}")

    # ── NeurIPS composite figure: 4 panels, 200 dpi ──────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=200, facecolor="white")

    axes[0].imshow(img_f)
    axes[0].set_title("Input image", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(heat_norm, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Anomaly score", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(gt_mask, cmap="gray")
    axes[3].set_title("GT mask", fontsize=11)
    axes[3].axis("off")

    plt.tight_layout(pad=0.5)
    composite_path = os.path.join(figures_dir, f"{sample_name}_heatmap_overlay.png")
    plt.savefig(composite_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved composite fig  : {composite_path}")


def _plot_superpixels(
    img:          np.ndarray,
    labels_fine:  np.ndarray,
    labels_coarse: np.ndarray,
    save_path:    str = None,
    show:         bool = False,
) -> None:
    from skimage.segmentation import find_boundaries
    from skimage.util import img_as_float
    from skimage.color import label2rgb

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    def _overlay(ax, image, labels, title, color="yellow"):
        vis = img_as_float(image).copy()
        b   = find_boundaries(labels, mode="thick")
        if vis.ndim == 2:
            vis = np.repeat(vis[..., None], 3, axis=2)
        vis[b, :] = {"yellow":[1.0,1.0,0.0],"red":[1.0,0.2,0.2],"cyan":[0.0,1.0,1.0]}.get(color,[1.0,1.0,0.0])
        ax.imshow(vis)
        ax.set_title(f"{title}\n({len(np.unique(labels))} regions)")
        ax.axis("off")

    axes[0].imshow(img); axes[0].set_title("Label image"); axes[0].axis("off")
    _overlay(axes[1], img, labels_coarse, "Coarse SPs (mask.png)", color="cyan")
    _overlay(axes[2], img, labels_fine,   "Refined SPs",           color="yellow")

    colored = label2rgb(labels_fine, image=img, kind="avg", bg_label=-1)
    b_fine  = find_boundaries(labels_fine, mode="thick")
    colored[b_fine, :] = 0.0
    axes[3].imshow(colored)
    axes[3].set_title(f"SP mean colours\n({len(np.unique(labels_fine))} regions)")
    axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved superpixel plot: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",            required=True)
    p.add_argument("--category",          default=None)
    p.add_argument("--subcategory",       default=None)
    p.add_argument("--all_subcategories", action="store_true")
    p.add_argument("--results_root",      default=None)
    p.add_argument("--results_suffix",    default="")
    p.add_argument("--gt_root",           default=None)
    p.add_argument("--sample_name",       default=None)
    p.add_argument("--sample_names",      nargs="+", default=None)
    p.add_argument("--skip_sampling",     action="store_true")
    p.add_argument("--no_plots",          action="store_true")
    p.add_argument("--device",            default=None)
    p.add_argument("--output_dir",        default=None)
    p.add_argument("--backbone",          default=None)
    p.add_argument("--n_pca",             type=int,   default=None)
    p.add_argument("--bins_rgb",          type=int,   default=None)
    p.add_argument("--bins_pca",          type=int,   default=None)
    p.add_argument("--superpixel_target_size", type=int, default=None)
    p.add_argument("--smooth_sigma",      type=float, default=None)
    p.add_argument("--scorer",            default="typical_set",
                   choices=["typical_set","local_gaussian"])
    p.add_argument("--sigma_rohan",       type=float, default=None)
    p.add_argument("--autoencoder_path",  default=None)
    p.add_argument("--no_pca",            action="store_true",
                   help="RGB-only ablation: drop the ResNet/autoencoder feature "
                        "term from the typical-set score. Exact, not approximate: "
                        "the RGB and feature factors are additive.")
    p.add_argument("--max_reconstructions", type=int, default=None,
                   help="Cap reconstructions per sample (e.g. 8, 16).")
    p.add_argument("--max_samples",       type=int, default=None,
                   help="Process at most N samples per subcategory — useful for "
                        "fast 10-image sweeps (e.g. --max_samples 10).")
    p.add_argument("--dump_maps",         default=None, metavar="DIR",
                   help="Also save each sample's raw delta map as "
                        "DIR/<sample_name>.npz (delta_map, gt_mask, "
                        "label_image, valid_mask) for full_metric_table.py.")

    p.add_argument("--image_dir",         default=None,
                   help="Directory with the test images (for superpixel generation); "
                        "overrides the gt_root-derived default for every subcategory.")
    p.add_argument("--ae_encode",         default=None, choices=["direct", "folded"],
                   help="How the autoencoder latents are computed (see _encode_folded). "
                        "Paper: CHAOS 'folded', FFHQ 'direct'; default 'direct'.")
    p.add_argument("--figures_dir",       default=None,
                   help="Cache dir for superpixel masks / figures (default from config).")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("sampling",{})["enabled"] = False
    cfg.setdefault("scoring",{})["algorithm"] = args.scorer

    if args.device:                        cfg["embeddings"]["device"]        = args.device
    if args.output_dir:                    cfg["eval"]["output_dir"]          = args.output_dir
    if args.backbone:                      cfg["embeddings"]["backbone"]      = args.backbone
    if args.n_pca        is not None:      cfg["pca"]["n_components"]         = args.n_pca
    if args.bins_rgb     is not None:      cfg["scoring"]["bins_rgb"]         = args.bins_rgb
    if args.bins_pca     is not None:      cfg["scoring"]["bins_pca"]         = args.bins_pca
    if args.smooth_sigma is not None:      cfg["scoring"]["smooth_sigma"]     = args.smooth_sigma
    if args.sigma_rohan  is not None:      cfg["scoring"]["sigma_rohan"]      = args.sigma_rohan
    if args.autoencoder_path:              cfg["scoring"]["autoencoder_path"] = args.autoencoder_path
    if args.no_pca:                        cfg["scoring"]["use_pca"]          = False
    if args.results_root:                  cfg["data"]["results_root"]        = args.results_root
    if args.results_suffix:                cfg["data"]["results_suffix"]      = args.results_suffix
    if args.gt_root:                       cfg["data"]["gt_root"]             = args.gt_root
    if args.superpixel_target_size is not None:
        cfg.setdefault("superpixels", {})["target_size"] = args.superpixel_target_size
    if args.max_reconstructions is not None:
        cfg.setdefault("sampling", {})["max_reconstructions"] = args.max_reconstructions
    if args.max_samples is not None:                          # ← NEW
        cfg.setdefault("eval", {})["max_samples"] = args.max_samples
    if args.dump_maps:
        cfg.setdefault("eval", {})["dump_maps_dir"] = args.dump_maps
    if args.image_dir:
        cfg["data"]["image_dir"] = args.image_dir
        cfg["data"]["image_dir_fixed"] = True
    if args.figures_dir:
        cfg["data"]["figures_dir"] = args.figures_dir
    if args.ae_encode:
        cfg["scoring"]["ae_encode"] = args.ae_encode

    output_dir   = cfg["eval"]["output_dir"]
    category     = args.category or cfg["data"].get("category","cable")
    sample_names = args.sample_names or ([args.sample_name] if args.sample_name else None)

    if args.all_subcategories:
        rroot   = cfg["data"].get("results_root", cfg["data"].get("results_dir",""))
        rsuffix = cfg["data"].get("results_suffix", "")
        subcats = discover_subcategories(rroot, category, suffix=rsuffix)
        if not subcats:
            print(f"[ERROR] No subcategories found for {rroot}_{category}_*{rsuffix}")
            sys.exit(1)
        print(f"Discovered subcategories: {subcats}")
    elif args.subcategory:
        subcats = [args.subcategory]
    else:
        subcats = [None]

    all_subcat_results = {}
    for subcat in subcats:
        if subcat is None:
            names = sample_names or [cfg["data"].get("sample_name","samples_000")]
            for sname in names:
                c = deepcopy(cfg)
                c["data"]["sample_name"] = sname
                c["data"]["_figures_dir"] = os.path.join(cfg["data"]["figures_dir"], sname)
                _run_single_sample(c)
        else:
            all_subcat_results[subcat] = _run_subcategory(
                cfg, category, subcat, sample_names, output_dir)

    if len(all_subcat_results) > 1:
        _summarize_category(all_subcat_results, category, output_dir, cfg)

if __name__ == "__main__":
    main()
    