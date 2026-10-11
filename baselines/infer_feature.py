"""One inference pass per (method, bench) for the five feature-based baselines,
saving UNSMOOTHED 256x256 quantities so every inference-time knob is applied
post hoc (posthoc.py):
  patchcore      : nn_dist [h, w, 15] sorted NN distances at patch resolution
                   (k-NN mean over the first k reproduces upstream for any k <= 15)
  simplenet/glass/supersimplenet : map [256, 256] with the internal sigma=4 blur disabled
  efficientad    : map_st, map_ae [256, 256] (quantile-normalised, padded, upsampled)
Preprocessing per (method, bench) follows the runs behind the published numbers
(table1_common.PREP). Run under the method's own conda env (see scripts/30_run_baselines.sh).
"""
import argparse, os, pickle, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import table1_common as C

IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def make_prep(method, bench):
    resize, crop = C.PREP[method][bench]
    t = [transforms.Resize(resize)]
    if crop:
        t.append(transforms.CenterCrop(crop))
    t += [transforms.ToTensor(), transforms.Normalize(*IMNET)]
    tf = transforms.Compose(t)
    return lambda p: tf(Image.open(p).convert("RGB")).unsqueeze(0)

def to256(m):
    """[H, W] float tensor/array -> [256, 256] float32 by bilinear (align_corners=False)."""
    m = m.detach().float().cpu() if torch.is_tensor(m) else torch.as_tensor(np.asarray(m, np.float32))
    if m.shape != (C.SIZE, C.SIZE):
        m = F.interpolate(m[None, None], size=(C.SIZE, C.SIZE), mode="bilinear",
                          align_corners=False)[0, 0]
    return m.cpu().numpy().astype(np.float32)

# ------------------------------------------------------------------ loaders
def load_patchcore(bench, dev):
    sys.path.insert(0, C.UPSTREAM["patchcore"])
    import patchcore.patchcore, patchcore.common, patchcore.backbones
    path = C.CKPT["patchcore"][bench]
    params = pickle.load(open(os.path.join(path, "patchcore_params.pkl"), "rb"))
    print("patchcore params:", params, flush=True)
    params["backbone"] = patchcore.backbones.load(params["backbone.name"])
    params["backbone"].name = params.pop("backbone.name")
    # NB upstream load() takes `anomaly_score_num_nn` and swallows the pickle's
    # `anomaly_scorer_num_nn` in **kwargs, so the published runs used k=1.
    # We query the index for KMAX neighbours directly and sweep k post hoc.
    pc = patchcore.patchcore.PatchCore(dev)
    pc.load(**params, device=dev, nn_method=patchcore.common.FaissNN(False, 16))   # 16 CPU threads (published: 4; same exact search)
    pc.anomaly_scorer.load(path)
    assert tuple(params["input_shape"][1:]) == (C.input_size("patchcore", bench),) * 2
    # Exact k-NN search on the GPU instead of CPU FAISS (no faiss-gpu in this env):
    # the memory bank is an IndexFlatL2, so squared-L2 top-k via matmul is the same
    # search. Verified vs faiss on the CT bank: 99.99% identical neighbour ids
    # (ties at 1e-4), max relative distance diff 4e-7, 450x faster.
    import faiss
    idx = pc.anomaly_scorer.nn_method.search_index
    assert isinstance(idx, faiss.IndexFlatL2), type(idx)
    xb = faiss.rev_swig_ptr(idx.get_xb(), idx.ntotal * idx.d).reshape(idx.ntotal, idx.d).copy()
    X = torch.from_numpy(xb).to(dev); Xn = (X * X).sum(1)
    print("patchcore memory bank: %d x %d on %s" % (idx.ntotal, idx.d, dev), flush=True)

    def knn(feats, k):
        Q = torch.from_numpy(np.ascontiguousarray(feats, np.float32)).to(dev)
        d2 = (Q * Q).sum(1, keepdim=True) + Xn[None] - 2 * Q @ X.T
        d2, _ = torch.topk(d2, k, dim=1, largest=False)
        return d2.clamp_min(0).cpu().numpy()

    @torch.no_grad()
    def infer(x):
        feats, patch_shapes = pc._embed(x.to(dev), provide_patch_shapes=True)
        feats = np.asarray(feats)
        feats = pc.anomaly_scorer.feature_merger.merge([feats])
        dists = knn(feats, C.PATCHCORE_KMAX)                                   # [n_patches, KMAX], ascending
        h, w = patch_shapes[0]
        return {"nn_dist": dists.reshape(h, w, C.PATCHCORE_KMAX).astype(np.float32),
                "input_size": np.int32(C.input_size("patchcore", bench))}
    return infer

def _load_sn_like(method, bench, dev):
    sys.path.insert(0, C.UPSTREAM[method])
    import backbones
    S = C.input_size(method, bench)
    if method == "simplenet":
        import simplenet as mod
        net = mod.SimpleNet(dev)
        extra = dict(embedding_size=256)
    else:
        import glass as mod
        net = mod.GLASS(dev)
        extra = {}
    net.load(backbone=backbones.load("wideresnet50"), layers_to_extract_from=["layer2", "layer3"],
             device=dev, input_shape=(3, S, S), pretrain_embed_dimension=1536,
             target_embed_dimension=1536, patchsize=3, dsc_layers=2, dsc_hidden=1024,
             pre_proj=1, **extra)
    sd = torch.load(C.CKPT[method][bench], map_location=dev, weights_only=False)
    net.discriminator.load_state_dict(sd["discriminator"])
    net.pre_projection.load_state_dict(sd["pre_projection"])
    net.discriminator.eval(); net.pre_projection.eval()
    net.anomaly_segmentor.smoothing = 0        # disable internal sigma=4 blur (identity, verified)

    @torch.no_grad()
    def infer(x):
        out = net._predict(x.to(dev))
        return {"map": to256(out[1][0])}
    return infer

def load_supersimplenet(bench, dev):
    sys.path.insert(0, C.UPSTREAM["supersimplenet"])
    from model.supersimplenet import SuperSimpleNet
    S = C.input_size("supersimplenet", bench)
    net = SuperSimpleNet(image_size=(S, S), config={"adapt_cls_feat": False}).to(dev)
    sd = torch.load(C.CKPT["supersimplenet"][bench], map_location=dev, weights_only=False)
    res = net.load_state_dict(sd, strict=False)
    miss = [k for k in res.missing_keys if not k.startswith("feature_extractor")]
    print("ssn non-backbone missing=%d unexpected=%d" % (len(miss), len(res.unexpected_keys)), flush=True)
    assert not miss and not res.unexpected_keys, (miss[:5], res.unexpected_keys[:5])
    net.eval()
    net.anomaly_map_generator.blur = torch.nn.Identity()   # disable internal sigma=4 blur

    @torch.no_grad()
    def infer(x):
        amap, _ = net(x.to(dev))
        return {"map": to256(amap[0, 0])}
    return infer

def load_efficientad(bench, dev, pad=4):
    d = C.CKPT["efficientad"][bench]
    m = {n: torch.load("%s/%s_final.pth" % (d, n), map_location=dev, weights_only=False).eval().to(dev)
         for n in ("teacher", "student", "autoencoder")}
    s = torch.load(d + "/norm_stats.pt", map_location=dev, weights_only=False)
    s = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
    Cn = s["teacher_mean"].shape[1]
    print("efficientad out_ch=%d q_st=[%.4f,%.4f] q_ae=[%.4f,%.4f]" % (
        Cn, s["q_st_start"], s["q_st_end"], s["q_ae_start"], s["q_ae_end"]), flush=True)

    @torch.no_grad()
    def infer(x):
        x = x.to(dev)
        t = (m["teacher"](x) - s["teacher_mean"]) / s["teacher_std"]
        st = m["student"](x)
        map_st = torch.mean((t - st[:, :Cn]) ** 2, dim=1, keepdim=True)
        map_ae = torch.mean((m["autoencoder"](x) - st[:, Cn:]) ** 2, dim=1, keepdim=True)
        map_st = 0.1 * (map_st - s["q_st_start"]) / (s["q_st_end"] - s["q_st_start"])
        map_ae = 0.1 * (map_ae - s["q_ae_start"]) / (s["q_ae_end"] - s["q_ae_start"])
        up = lambda z: F.interpolate(F.pad(z, (pad,) * 4) if pad else z, size=(C.SIZE, C.SIZE),
                                     mode="bilinear", align_corners=False)[0, 0].cpu().numpy().astype(np.float32)
        return {"map_st": up(map_st), "map_ae": up(map_ae)}
    return infer

LOADERS = {"patchcore": load_patchcore,
           "simplenet": lambda b, d: _load_sn_like("simplenet", b, d),
           "glass": lambda b, d: _load_sn_like("glass", b, d),
           "supersimplenet": load_supersimplenet, "efficientad": load_efficientad}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=C.FEATURE_METHODS)
    ap.add_argument("--bench", required=True, choices=C.BENCHES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_root", default=C.OUT_ROOT)
    ap.add_argument("--limit", type=int, default=0, help="first N images per set (smoke test)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--nopad", action="store_true", help="EfficientAD: skip the 4-px pad (as the lab's CT/X-ray scripts); setting 'nopad'")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(a.device)
    items = C.list_images(a.bench)
    if a.limit:
        per = {}
        items = [it for it in items if per.setdefault(it[0], []).append(it) or len(per[it[0]]) <= a.limit]
    od = C.out_dir(a.out_root, a.bench, a.method, "nopad" if a.nopad else "default")
    os.makedirs(od, exist_ok=True)
    todo = [it for it in items if a.force or not os.path.exists(os.path.join(od, C.npz_name(it[0], it[1])))]
    print("%s/%s: %d images, %d to do -> %s" % (a.method, a.bench, len(items), len(todo), od), flush=True)
    if not todo:
        return
    infer = load_efficientad(a.bench, dev, pad=0) if a.nopad else LOADERS[a.method](a.bench, dev)
    prep = make_prep(a.method, a.bench)
    for i, (s, sid, img, gt, valid) in enumerate(todo):
        out = infer(prep(img))
        out["gt_mask"] = C.load_gt(gt); out["valid_mask"] = C.load_valid(valid)
        np.savez_compressed(os.path.join(od, C.npz_name(s, sid)), **out)
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print("  %d/%d" % (i + 1, len(todo)), flush=True)
    print("done %s/%s" % (a.method, a.bench), flush=True)

if __name__ == "__main__":
    main()
