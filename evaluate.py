"""LACToS scorer: diffusion reconstructions -> superpixels -> typicality score -> per-image delta map.

Per test image: load the M reconstructions and the image, build / load the SLIC superpixel mask
(super_pixel_generation.py), refine it to the target size (ood.superpixels), embed image and
reconstructions with ResNet-101 features compressed by the per-dataset PixelAutoEncoder
(ood.embeddings + PixelAutoEncoder below), score every superpixel by (NLL - entropy) of the
image's pixels under the reconstructions' RGB and latent histograms (ood.scoring), and dump the
delta map. The per-dataset configs (configs/lactos_*.yaml) hold the paper's settings, so
`python evaluate.py --config configs/lactos_ffhq_faces.yaml` scores a set as in the paper; every
config value can be overridden on the command line. scripts/score_lactos.sh runs every set;
stage 3 (scripts/make_table1.py) pools the dumps into Table 1.

Reconstruction layout (ood.data): <results_dir>/<sample>/<test_origin>_0_<bottom_suffix>/inpainting/{recon/*.png, label/0_00000.png}
GT masks: <gt_root>/[<category>/][<subcategory>/]<sample_id>_mask.png, or data.gt_mask.path in the config.
CT: lesion + body masks are read from $OOD_CT_MASK_ROOT/{masks_ood_med_med,masks_body_ood_default_large}/image_XXXX_mask.png
and the histogram is grayscale.
"""

import argparse, json, os, re, subprocess, sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ood.data import (load_gt_mask, load_label_image, load_reconstructions,
                      load_superpixel_mask, parse_sample_id)
from ood.superpixels import recursive_subdivide
from ood.embeddings import ResNetPixelEmbedder
from ood.scoring import compute_delta_map_gpu


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
    THIS path (the lab's package copy of the scorer); FFHQ with the direct path. `--ae_encode` selects.
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


def discover_samples(results_dir):
    if not os.path.isdir(results_dir):
        return []
    return sorted(e for e in os.listdir(results_dir)
                  if os.path.isdir(os.path.join(results_dir, e)) and e.startswith("samples_"))


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
        _ct_root = os.environ.get("OOD_CT_MASK_ROOT", "assets/data/CT/masks")
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
        raise ValueError("--autoencoder_path is required unless --no_pca (RGB-only scoring) is given")
    del embedder; torch.cuda.empty_cache()

    # Score: typical-set deviation per superpixel (RGB term + feature term)
    delta_map,_,labels_used = compute_delta_map_gpu(
        labels_fine=labels_fine, images_recon_all=recon_all,
        pca_feats_recon=embed_result["pca_feats_recon"],
        label_image=label_image, label_pca_map=embed_result["label_pca_map"],
        bins_rgb=score_cfg["bins_rgb"], bins_pca=score_cfg["bins_pca"],
        smooth_sigma=score_cfg["smooth_sigma"], min_pixels=score_cfg["min_pixels"],
        device=device, gray_scale=gray_scale, use_pca=use_pca)

    print(f"  Scored {len(labels_used)} superpixels")

    # ── Shape reconciliation ──────────────────────────────────────────────
    if gt_mask.shape != delta_map.shape:
        from skimage.transform import resize as _resize
        print(f"  [WARN] Shape mismatch: delta_map={delta_map.shape} "
              f"gt_mask={gt_mask.shape} — resizing gt_mask to match delta_map.")
        gt_mask = _resize(
            gt_mask.astype(np.float32), output_shape=delta_map.shape,
            order=0, mode="edge", preserve_range=True, anti_aliasing=False,
        ).astype(np.uint8)

    if valid_mask is not None and valid_mask.shape != delta_map.shape:
        from skimage.transform import resize as _resize
        print(f"  [WARN] valid_mask shape mismatch, resizing to {delta_map.shape}.")
        valid_mask = _resize(
            valid_mask.astype(np.float32), output_shape=delta_map.shape,
            order=0, mode="edge", preserve_range=True, anti_aliasing=False,
        ).astype(np.uint8)

    # Per-image dump read by scripts/make_table1.py: delta_map (float32, NaN where a superpixel
    # could not be scored), gt_mask (uint8), label_image (uint8), valid_mask (uint8 or EMPTY when
    # every pixel is valid; full_metric_table.py accepts both).
    dump_dir = cfg["eval"].get("dump_maps_dir")
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
    return {"n_superpixels": int(len(labels_used)), "n_reconstructions": int(len(recon_all)),
            "dump": dump_path if dump_dir else None}


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
    if base_cfg["data"].get("image_dir"):          # from the config or --image_dir
        print(f"  Image dir    : {cfg['data']['image_dir']}")
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

    os.makedirs(output_dir, exist_ok=True)
    jpath = os.path.join(output_dir, f"{category}_{subcategory}.json")
    with open(jpath, "w") as f:
        json.dump({"category": category, "subcategory": subcategory, "results_dir": results_dir,
                   "gt_template": gt_template, "samples": results, "failed": failed}, f, indent=2, default=str)
    print(f"  {len(results)} samples scored, {len(failed)} failed -> {jpath}")
    return results


# ── Category summary ──────────────────────────────────────────────────────────


# ── Heatmap / figure saving ───────────────────────────────────────────────────


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="LACToS scoring: reconstructions -> superpixel typicality -> per-image delta-map dumps")
    p.add_argument("--config",            required=True, help="configs/lactos_*.yaml")
    p.add_argument("--category",          default=None, help="CT | xray | faces (default: data.category of the config; 'CT' selects the CT branch)")
    p.add_argument("--subcategory",       default=None, help="image set within the category, e.g. cta, ctn, xrsc, xrgd, random (default: data.subcategory of the config)")
    p.add_argument("--results_root",      default=None,
                   help="Reconstructions in the <root>_<category>_<subcategory> layout (CT, xray).")
    p.add_argument("--results_dir",       default=None,
                   help="Reconstruction directory of this one set (FFHQ); overrides results_root.")
    p.add_argument("--gt_root",           default=None,
                   help="Ground-truth root; <gt_root>/[<category>/][<subcategory>/]<sample>_mask.png is tried.")
    p.add_argument("--image_dir",         default=None,
                   help="Test images (for superpixel generation); default <dirname(gt_root)>/test/<subcategory>.")
    p.add_argument("--figures_dir",       default=None, help="Cache dir for superpixel masks (default from config).")
    p.add_argument("--autoencoder_path",  default=None, help="PixelAutoEncoder state dict (required unless --no_pca).")
    p.add_argument("--ae_encode",         default=None, choices=["direct", "folded"],
                   help="How the autoencoder latents are computed (see _encode_folded). Paper: CHAOS folded, FFHQ direct.")
    p.add_argument("--no_pca",            action="store_true", help="RGB-only scoring (feature term disabled).")
    p.add_argument("--backbone",          default=None, help="torchvision ResNet name (paper: resnet101)")
    p.add_argument("--n_pca",             type=int,   default=None, help="autoencoder latent size (paper: 3)")
    p.add_argument("--bins_rgb",          type=int,   default=None)
    p.add_argument("--bins_pca",          type=int,   default=None)
    p.add_argument("--smooth_sigma",      type=float, default=None, help="histogram smoothing sigma in bins")
    p.add_argument("--superpixel_target_size", type=int, default=None)
    p.add_argument("--dump_maps",         default=None, metavar="DIR",
                   help="Save each sample's delta map as DIR/<sample_name>.npz (delta_map, gt_mask, label_image, valid_mask).")
    p.add_argument("--output_dir",        default=None, help="where the per-set summary JSON goes")
    p.add_argument("--max_samples",       type=int, default=None, help="score only the first N samples (smoke tests)")
    p.add_argument("--device",            default=None)
    p.add_argument("--skip_sampling",     action="store_true", help="accepted for compatibility; sampling is never run here")
    p.add_argument("--no_plots",          action="store_true", help="accepted for compatibility; no plots are produced")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("sampling", {})["enabled"] = False
    category = args.category or cfg["data"].get("category")
    subcategory = args.subcategory or cfg["data"].get("subcategory")
    if not category or not subcategory:
        p.error("--category / --subcategory are needed when the config has no data.category / data.subcategory")
    if args.device:                        cfg["embeddings"]["device"]        = args.device
    if args.output_dir:                    cfg["eval"]["output_dir"]          = args.output_dir
    if args.backbone:                      cfg["embeddings"]["backbone"]      = args.backbone
    if args.n_pca        is not None:      cfg["pca"]["n_components"]         = args.n_pca
    if args.bins_rgb     is not None:      cfg["scoring"]["bins_rgb"]         = args.bins_rgb
    if args.bins_pca     is not None:      cfg["scoring"]["bins_pca"]         = args.bins_pca
    if args.smooth_sigma is not None:      cfg["scoring"]["smooth_sigma"]     = args.smooth_sigma
    if args.autoencoder_path:              cfg["scoring"]["autoencoder_path"] = args.autoencoder_path
    if args.no_pca:                        cfg["scoring"]["use_pca"]          = False
    if args.ae_encode:                     cfg["scoring"]["ae_encode"]        = args.ae_encode
    if args.results_root:                  cfg["data"]["results_root"]        = args.results_root
    if args.results_dir:
        cfg["data"]["results_dir"] = args.results_dir
        cfg["data"].pop("results_root", None)
    if args.gt_root:                       cfg["data"]["gt_root"]             = args.gt_root
    if args.image_dir:                     cfg["data"]["image_dir"]           = args.image_dir
    if args.figures_dir:                   cfg["data"]["figures_dir"]         = args.figures_dir
    if args.superpixel_target_size is not None:
        cfg.setdefault("superpixels", {})["target_size"] = args.superpixel_target_size
    if args.max_samples is not None:
        cfg.setdefault("eval", {})["max_samples"] = args.max_samples
    if args.dump_maps:
        cfg.setdefault("eval", {})["dump_maps_dir"] = args.dump_maps

    _run_subcategory(cfg, category, subcategory, None, cfg["eval"]["output_dir"])


if __name__ == "__main__":
    main()
