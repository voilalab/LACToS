"""DDAD reconstruction pass for one (bench, w, T, skip, eta) setting. Saves, per
image, the unsmoothed pixel distance i_d and feature distance f_d (256x256) so
that v (pixel/feature weight) and the spatial blur are applied post hoc:
  map = f_d + v * (max f_d / max i_d) * i_d      (DDAD/anomaly_map.py, minus blur)
Reuses the DDAD modules in baselines/ddad_lib/ verbatim (Reconstruction, pixel/feature distance,
domain_adaptation). Noise is seeded PER IMAGE (table1_common.image_seed) so settings are
comparable and runs are resumable. Env: ddad.
"""
import argparse, os, sys
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import table1_common as C
sys.path.insert(0, os.path.join(HERE, "ddad_lib"))
from reconstruction import Reconstruction            # noqa: E402
from anomaly_map import pixel_distance, feature_distance  # noqa: E402
from feature_extractor import domain_adaptation      # noqa: E402
from unet import UNetModel                           # noqa: E402

def unwrap(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "ema_state_dict", "ema"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]; break
    if isinstance(ckpt, dict) and next(iter(ckpt)).startswith("module."):
        ckpt = {k[7:]: v for k, v in ckpt.items()}
    return ckpt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=C.BENCHES)
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--T", type=int, default=250)
    ap.add_argument("--skip", type=int, default=25)
    ap.add_argument("--eta", type=float, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_root", default=C.OUT_ROOT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(HERE, "ddad_config.yaml"))
    cfg.model.device = a.device
    cfg.model.feature_extractor = C.DDAD_FE[a.bench]
    cfg.model.use_frozen_fe = True
    cfg.model.w = a.w; cfg.model.test_trajectoy_steps = a.T; cfg.model.skip = a.skip; cfg.model.eta = a.eta
    setting = C.ddad_setting(a.w, a.T, a.skip, a.eta)
    od = C.out_dir(a.out_root, a.bench, "ddad", setting); os.makedirs(od, exist_ok=True)
    items = C.list_images(a.bench)
    if a.limit:
        per = {}
        items = [it for it in items if per.setdefault(it[0], []).append(it) or len(per[it[0]]) <= a.limit]
    todo = [it for it in items if a.force or not os.path.exists(os.path.join(od, C.npz_name(it[0], it[1])))]
    print("ddad/%s/%s fe=%s: %d images, %d to do" % (a.bench, setting, cfg.model.feature_extractor,
                                                    len(items), len(todo)), flush=True)
    if not todo:
        return
    dev = torch.device(a.device)
    unet = UNetModel(cfg.data.image_size, 64, dropout=0.0, n_heads=4, in_channels=cfg.data.input_channel)
    unet.load_state_dict(unwrap(torch.load(C.CKPT["ddad"][a.bench], map_location="cpu")), strict=True)
    unet = unet.to(dev).eval()
    fe = domain_adaptation(unet, cfg, fine_tune=False).eval()
    recon = Reconstruction(unet, cfg)
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                             transforms.Lambda(lambda t: t * 2 - 1)])
    with torch.no_grad():
        for i, (s, sid, img, gt, valid) in enumerate(todo):
            x = tf(Image.open(img).convert("RGB")).unsqueeze(0).to(dev)
            torch.manual_seed(C.image_seed(s, sid))
            x0 = recon(x, x, cfg.model.w)[-1]
            i_d = pixel_distance(x0, x)[0, 0]
            f_d = feature_distance(x0, x, fe, cfg)[0, 0]
            np.savez_compressed(os.path.join(od, C.npz_name(s, sid)),
                                i_d=i_d.cpu().numpy().astype(np.float32),
                                f_d=f_d.cpu().numpy().astype(np.float32),
                                gt_mask=C.load_gt(gt), valid_mask=C.load_valid(valid))
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                print("  %d/%d" % (i + 1, len(todo)), flush=True)
    print("done ddad/%s/%s" % (a.bench, setting), flush=True)

if __name__ == "__main__":
    main()
