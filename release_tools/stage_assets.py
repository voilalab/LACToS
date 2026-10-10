#!/usr/bin/env python
"""Stage the LACToS Table-1 asset bundle from the lab's scattered sources.

Run once, on the lab cluster, by a maintainer:
  python release_tools/stage_assets.py --out /usr/scratch/jhong392/workspace/LACToS_table1_assets

Every copy is a real file (no symlinks). Re-running skips files that already exist.
The resulting layout is documented in docs/BUNDLE.md and checked by release_tools/check_bundle.py.
Source locations are those recorded in docs/PROVENANCE.md (lab paths, 2026-10-10).
"""
import argparse, glob, hashlib, os, shutil

R = "/usr/scratch/jhong392/workspace/Statistical_OOD_detection"
A = "/usr/scratch/akheirandish3"
PKG = f"{A}/OOD_reproduction_package"
BUN = f"{A}/dps_ct_bundle/outputs"
H = f"{A}/handoff_CT_xray_20260924"
XR = f"{A}/xray_baselines/xray_checkpoints_by_baseline"


def cp(src, dst):
    if os.path.exists(dst):
        return
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src = os.path.realpath(src)
    if os.path.isdir(src):
        # regular files only: the lab image dirs contain `random -> .` convenience symlinks
        os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(src)):
            fp = os.path.join(src, f)
            if os.path.isfile(fp) and not os.path.islink(fp):
                shutil.copy2(fp, os.path.join(dst, f))
    else:
        shutil.copy2(src, dst)


CKPT = {  # dst under checkpoints/ : src
    "lactos/ae_CT.pth": f"{PKG}/models/pixel_autoencoder_CT.pth",
    "lactos/ae_faces.pth": f"{R}/checkpoints/faces/autoencoder/faces.pth",
    "lactos/ae_xray.pth": f"{PKG}/all_categories_embedder/models/xray_mine/last_layer.pth",
    "lactos/backbone_xray_ddad_lastlayer_feat4.pth": f"{PKG}/experiments/xray_finetune_mine/last_layer/checkpoints/feat4.pth",
    "diffusion/ct_model050000.pt": f"{A}/dps_ct_bundle/checkpoints/model050000.pt",
    "diffusion/xray_model500000.pt": f"{A}/Xray/model500000.pt",
    "diffusion/ffhq_10m.pt": f"{R}/checkpoints/faces/guided-diffusion/ffhq_10m.pt",
    "baselines/ffhq/patchcore/nnscorer_search_index.faiss": f"{R}/checkpoints/faces/patchcore_faces_5k/FacesAD_5k/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0/models/mvtec_faces/nnscorer_search_index.faiss",
    "baselines/ffhq/patchcore/patchcore_params.pkl": f"{R}/checkpoints/faces/patchcore_faces_5k/FacesAD_5k/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0/models/mvtec_faces/patchcore_params.pkl",
    "baselines/ffhq/simplenet/ckpt.pth": f"{R}/checkpoints/faces/simplenet_faces_70k/FacesAD_70k/simplenet_faces/run/models/0/mvtec_faces/ckpt.pth",
    "baselines/ffhq/glass/ckpt_best.pth": f"{R}/checkpoints/faces/GLASS/ckpt_best_29.pth",
    "baselines/ffhq/supersimplenet/weights.pt": f"{R}/checkpoints/faces/supersimplenet_faces_5k/superSimpleNet/checkpoints/mvtec/faces/1/weights.pt",
    "baselines/ffhq/ddad/unet.pth": f"{R}/checkpoints/faces/DDAD/2000",
    "baselines/ct/patchcore/nnscorer_search_index.faiss": f"{A}/mvtec_ct/nnscorer_search_index.faiss",
    "baselines/ct/patchcore/patchcore_params.pkl": f"{A}/mvtec_ct/patchcore_params.pkl",
    "baselines/ct/simplenet/ckpt.pth": f"{A}/baselines/simple_net/ckpt.pth",
    "baselines/ct/glass/ckpt_best.pth": f"{A}/ckpt_best_48.pth",
    "baselines/ct/supersimplenet/weights.pt": f"{A}/baselines/super_simple_net/weights.pt",
    "baselines/ct/ddad/unet.pth": f"{H}/ddad_baseline/checkpoints/CT/2000",
    "baselines/xray/patchcore/nnscorer_search_index.faiss": f"{XR}/patchcore/nnscorer_search_index.faiss",
    "baselines/xray/patchcore/patchcore_params.pkl": f"{XR}/patchcore/patchcore_params.pkl",
    "baselines/xray/simplenet/ckpt.pth": f"{XR}/simplenet/ckpt.pth",
    "baselines/xray/glass/ckpt_best.pth": f"{XR}/glass/ckpt_best_29.pth",
    "baselines/xray/supersimplenet/weights.pt": f"{XR}/supersimplenet/weights.pt",
    "baselines/xray/ddad/unet.pth": f"{H}/ddad_baseline/checkpoints/xray/500",
}
EAD = {"ffhq": f"{R}/checkpoints/faces/efficientad_faces/faces", "ct": f"{A}/baselines/effecient-AD", "xray": f"{XR}/efficientad"}
EAD_FILES = ("teacher_final.pth", "student_final.pth", "autoencoder_final.pth", "norm_stats.pt", "run_args.txt")

# raw DPS outputs: <src>/<stem>/{recon,label,input}/m<k>.png  ->  evaluate.py layout
RECON = {  # dst dir under recons/ : (src dir, n_draws, id remap)
    "results_patches_ddad_native_CT_cta": (f"{BUN}/CT_sigma0.1_zeta0.2_full", 5, None),
    "results_patches_ddad_native_CT_ctn": (f"{BUN}/ID_same_base_sigma0.1_zeta0.2", 5, lambda s: f"image_{int(s.split('_')[1]) + 6000}"),
    "results_patches_ddad_native_xray_xrsc": (f"{BUN}/XRAY_scissors_zeta0.5_sigma2.0_M20", 20, None),
    "results_patches_ddad_native_xray_xrgd": (f"{BUN}/XRAY_good_zeta0.5_sigma2.0_M20", 20, None),
}
FFHQ_RECON = {"faces": 22, "moreface": 28, "faces_normal": 50}  # results_patches_adm_dps_<set>, already in evaluate.py layout
# the faces set is the 22 ids of configs/recon/adm_dps_faces_images.yaml (5 and 98 were dropped on 2026-09-22;
# their reconstructions still sit in the lab tree and must not be shipped)
FACES_IDS = ["1", "2", "3", "4", "6", "7", "8", "9", "10", "40", "44", "49", "59", "61", "65", "80", "86", "92", "107", "117", "120", "129"]


def stage(out):
    for d, s in CKPT.items():
        cp(s, f"{out}/checkpoints/{d}")
    for b, d in EAD.items():
        for f in EAD_FILES:
            if os.path.exists(f"{d}/{f}"):
                cp(f"{d}/{f}", f"{out}/checkpoints/baselines/{b}/efficientad/{f}")
    print("checkpoints staged", flush=True)

    for dst, (src, n, remap) in RECON.items():
        stems = sorted(os.path.basename(p) for p in glob.glob(f"{src}/*") if os.path.isdir(p))
        for stem in stems:
            sid = remap(stem) if remap else stem
            base = f"{out}/recons/{dst}/samples_{sid}/ddad_native_0_4/inpainting"
            for k in range(n):
                cp(f"{src}/{stem}/recon/m{k}.png", f"{base}/recon/{k}_0_00000.png")
            cp(f"{src}/{stem}/label/m0.png", f"{base}/label/0_00000.png")
            cp(f"{src}/{stem}/input/m0.png", f"{base}/input/0_00000.png")
        print(f"{dst}: {len(stems)} samples x {n} draws", flush=True)
    for s, n in FFHQ_RECON.items():
        src = f"{R}/results_patches_adm_dps_{s}"
        dst = f"{out}/recons/results_patches_adm_dps_{s}"
        samples = sorted(d for d in os.listdir(src) if d.startswith("samples_"))
        if s == "faces":
            samples = [f"samples_{i}" for i in FACES_IDS]
            assert all(os.path.isdir(f"{src}/{d}") for d in samples), "faces recon missing"
        assert len(samples) == n, (s, len(samples))
        for d in samples:
            cp(f"{src}/{d}", f"{dst}/{d}")
        cp(f"{src}/recon_metadata.json", f"{dst}/recon_metadata.json")
        print(f"results_patches_adm_dps_{s}: {n} samples", flush=True)

    # CT: anomalous slices keep their ids; normal slices (same base slice, same id) get id 6000+i,
    # as in the paper runs, so GT / body masks / recons / dumps never collide.
    for i in range(100):
        a, nrm = f"image_{i:04d}", f"image_{6000 + i}"
        cp(f"{H}/CT/test/anomalous/{a}.png", f"{out}/data/CT/test/cta/{a}.png")
        cp(f"{H}/CT/test/normal/{a}.png", f"{out}/data/CT/test/ctn/{nrm}.png")
        cp(f"{H}/CT/ground_truth/anomalous/{a}_mask.png", f"{out}/data/CT/ground_truth/cta/{a}_mask.png")
        cp(f"{H}/CT/ground_truth/normal/{a}_mask.png", f"{out}/data/CT/ground_truth/ctn/{nrm}_mask.png")
        for m in (a, nrm):
            cp(f"{PKG}/data/id_new_warped_images/masks_ood_med_med/{m}_mask.png", f"{out}/data/CT/masks/masks_ood_med_med/{m}_mask.png")
            cp(f"{PKG}/data/id_new_warped_images/masks_body_ood_default_large/{m}_mask.png", f"{out}/data/CT/masks/masks_body_ood_default_large/{m}_mask.png")
    cp(f"{H}/CT/grid_subset_selection.json", f"{out}/data/CT/grid_subset_selection.json")
    for sub, hs in (("xrsc", "scissors"), ("xrgd", "good")):
        cp(f"{H}/xray/test/{hs}", f"{out}/data/xray/test/{sub}")
        cp(f"{H}/xray/ground_truth/{hs}", f"{out}/data/xray/ground_truth/{sub}")
    cp(f"{H}/xray/grid_subset_selection.json", f"{out}/data/xray/grid_subset_selection.json")
    cp(f"{R}/images/faces/test", f"{out}/data/ffhq/faces/test")
    cp(f"{R}/images/faces/ground_truth", f"{out}/data/ffhq/faces/ground_truth")
    cp(f"{R}/images/moreface_masked_subset_256_fullsized_masks_binary/test/random", f"{out}/data/ffhq/moreface/test")
    cp(f"{R}/images/moreface_masked_subset_256_fullsized_masks_binary/ground_truth/random", f"{out}/data/ffhq/moreface/ground_truth")
    cp(f"{R}/images/moreface_masked_subset_256_fullsized_masks_binary/README.txt", f"{out}/data/ffhq/moreface/README.txt")
    cp(f"{R}/images/faces_normal/test", f"{out}/data/ffhq/faces_normal/test")
    cp(f"{R}/images/faces_normal/ground_truth", f"{out}/data/ffhq/faces_normal/ground_truth")
    cp(f"{R}/images/faces_normal/SOURCE.json", f"{out}/data/ffhq/faces_normal/SOURCE.json")
    print("data staged", flush=True)


def checksums(out):
    lines = []
    for root, _, files in os.walk(out):
        for f in files:
            if f in ("SHA256SUMS", "MANIFEST.md"):
                continue
            p = os.path.join(root, f)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            lines.append((os.path.relpath(p, out), h))
    lines.sort()
    with open(f"{out}/SHA256SUMS", "w") as fh:
        for rel, h in lines:
            fh.write(f"{h}  {rel}\n")
    print(len(lines), "files hashed", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-sha", action="store_true")
    a = ap.parse_args()
    stage(a.out)
    if not a.no_sha:
        checksums(a.out)
