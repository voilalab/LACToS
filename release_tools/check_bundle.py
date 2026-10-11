#!/usr/bin/env python
"""Check an extracted LACToS Table-1 asset bundle: layout, sample/draw counts, SHA-256.

  python release_tools/check_bundle.py ASSETS_DIR [--no-sha]

Exit code 0 = complete bundle. Prints every missing or mismatching item.
"""
import argparse, hashlib, os, sys

EXPECTED_RECONS = {  # dir under recons/ : (n_samples, n_draws, sample prefix)
    "results_patches_ddad_native_CT_cta": (100, 5, "samples_image_0"),
    "results_patches_ddad_native_CT_ctn": (100, 5, "samples_image_6"),
    "results_patches_ddad_native_xray_xrsc": (72, 20, "samples_P"),
    "results_patches_ddad_native_xray_xrgd": (72, 20, "samples_N"),
    "results_patches_adm_dps_faces": (22, 20, "samples_"),
    "results_patches_adm_dps_moreface": (28, 20, "samples_"),
    "results_patches_adm_dps_faces_normal": (50, 20, "samples_1"),
}
EXPECTED_FILES = [
    "checkpoints/lactos/ae_CT.pth", "checkpoints/lactos/ae_faces.pth", "checkpoints/lactos/ae_xray.pth",
    "checkpoints/lactos/backbone_xray_ddad_lastlayer_feat4.pth",
    "checkpoints/diffusion/ct_model050000.pt", "checkpoints/diffusion/xray_model500000.pt", "checkpoints/diffusion/ffhq_10m.pt",
    "data/CT/grid_subset_selection.json", "data/xray/grid_subset_selection.json", "SHA256SUMS",
]
for b in ("ffhq", "ct", "xray"):
    EXPECTED_FILES += [f"checkpoints/baselines/{b}/patchcore/nnscorer_search_index.faiss",
                       f"checkpoints/baselines/{b}/patchcore/patchcore_params.pkl",
                       f"checkpoints/baselines/{b}/simplenet/ckpt.pth", f"checkpoints/baselines/{b}/glass/ckpt_best.pth",
                       f"checkpoints/baselines/{b}/supersimplenet/weights.pt", f"checkpoints/baselines/{b}/ddad/unet.pth",
                       f"checkpoints/baselines/{b}/efficientad/teacher_final.pth", f"checkpoints/baselines/{b}/efficientad/student_final.pth",
                       f"checkpoints/baselines/{b}/efficientad/autoencoder_final.pth", f"checkpoints/baselines/{b}/efficientad/norm_stats.pt"]
EXPECTED_DATA_COUNTS = {  # dir : (n_png, suffix)
    "data/CT/test/cta": 100, "data/CT/test/ctn": 100, "data/CT/ground_truth/cta": 100, "data/CT/ground_truth/ctn": 100,
    "data/CT/masks/masks_ood_med_med": 200, "data/CT/masks/masks_body_ood_default_large": 200, "data/CT/body_masks_baselines": 100,
    "data/xray/test/xrsc": 72, "data/xray/test/xrgd": 72, "data/xray/ground_truth/xrsc": 72, "data/xray/ground_truth/xrgd": 72,
    "data/ffhq/faces/test": 22, "data/ffhq/faces/ground_truth": 22,
    "data/ffhq/moreface/test": 28, "data/ffhq/moreface/ground_truth": 28,
    "data/ffhq/faces_normal/test": 50, "data/ffhq/faces_normal/ground_truth": 50,
}


def recon_problems(root):
    out = []
    for d, (n, draws, prefix) in EXPECTED_RECONS.items():
        base = os.path.join(root, "recons", d)
        if not os.path.isdir(base):
            out.append(f"missing recon dir {d}"); continue
        samples = sorted(s for s in os.listdir(base) if s.startswith("samples_"))
        if len(samples) != n:
            out.append(f"{d}: {len(samples)} samples, expected {n}")
        bad_prefix = [s for s in samples if not s.startswith(prefix)]
        if bad_prefix:
            out.append(f"{d}: unexpected sample names {bad_prefix[:3]}")
        for s in samples:
            sub = [x for x in os.listdir(os.path.join(base, s)) if "_0_4" in x]
            if len(sub) != 1:
                out.append(f"{d}/{s}: expected one <origin>_0_4 dir, found {sub}"); continue
            inp = os.path.join(base, s, sub[0], "inpainting")
            k = len([f for f in os.listdir(os.path.join(inp, "recon")) if f.endswith(".png")]) if os.path.isdir(os.path.join(inp, "recon")) else 0
            if k != draws:
                out.append(f"{d}/{s}: {k} recon pngs, expected {draws}")
            if not os.path.exists(os.path.join(inp, "label", "0_00000.png")):
                out.append(f"{d}/{s}: label/0_00000.png missing")
    return out


def data_problems(root):
    out = []
    for d, n in EXPECTED_DATA_COUNTS.items():
        p = os.path.join(root, d)
        k = len([f for f in os.listdir(p) if f.endswith(".png")]) if os.path.isdir(p) else -1
        if k != n:
            out.append(f"{d}: {k} png files, expected {n}")
    return out


def sha_problems(root):
    out = []
    path = os.path.join(root, "SHA256SUMS")
    if not os.path.exists(path):
        return ["SHA256SUMS missing"]
    listed = {line.rstrip("\n").split("  ", 1)[1] for line in open(path)}
    for d, _, files in os.walk(root):
        for f in files:
            rel = os.path.relpath(os.path.join(d, f), root)
            if rel not in listed and rel != "SHA256SUMS":
                out.append(f"file not listed in SHA256SUMS (stray?): {rel}")
    for d, dirs, _ in os.walk(root):
        for x in dirs:
            if os.path.islink(os.path.join(d, x)):
                out.append(f"unexpected symlink: {os.path.relpath(os.path.join(d, x), root)}")
    for line in open(path):
        h, rel = line.rstrip("\n").split("  ", 1)
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            out.append(f"listed file missing: {rel}"); continue
        if hashlib.sha256(open(p, "rb").read()).hexdigest() != h:
            out.append(f"sha256 mismatch: {rel}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("assets")
    ap.add_argument("--no-sha", action="store_true", help="skip the SHA-256 pass (slow on 10 GB)")
    a = ap.parse_args()
    root = a.assets
    problems = [f"missing file {f}" for f in EXPECTED_FILES if not os.path.exists(os.path.join(root, f))]
    problems += data_problems(root) + recon_problems(root)
    if not a.no_sha:
        problems += sha_problems(root)
    for p in problems:
        print("  ", p)
    if problems:
        print(f"BUNDLE INCOMPLETE: {len(problems)} problem(s) in {root}")
        sys.exit(1)
    print(f"bundle OK: {root}")


if __name__ == "__main__":
    main()
