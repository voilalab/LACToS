#!/usr/bin/env python
"""Convert raw sample_fast.py output into the layout evaluate.py reads.

  python sampling/wire_recons.py --src RAW_DIR --dst RECON_DIR --n_draws M [--id_offset 6000]

RAW_DIR/<stem>/{recon,label,input}/m<k>.png   (sample_fast.py, one dir per test image)
  -> RECON_DIR/samples_<stem[+offset]>/ddad_native_0_4/inpainting/{recon/<k>_0_00000.png, label/0_00000.png, input/0_00000.png}

--id_offset renames image_XXXX to image_(XXXX+offset); the CT normal slices use 6000 so that
their ids never collide with the anomalous slice of the same base image. Files are copied
(not linked) so the result can be moved or archived. Re-running skips existing files.
"""
import argparse, glob, os, shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n_draws", type=int, required=True)
    ap.add_argument("--id_offset", type=int, default=0)
    a = ap.parse_args()
    stems = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.src, "*")) if os.path.isdir(p))
    if not stems:
        raise SystemExit(f"no sample dirs under {a.src}")
    n_ok = 0
    for stem in stems:
        sid = stem
        if a.id_offset and stem.startswith("image_"):
            sid = "image_%d" % (int(stem.split("_")[1]) + a.id_offset)
        base = os.path.join(a.dst, f"samples_{sid}", "ddad_native_0_4", "inpainting")
        pairs = [(os.path.join(a.src, stem, "recon", f"m{k}.png"), os.path.join(base, "recon", f"{k}_0_00000.png")) for k in range(a.n_draws)]
        pairs += [(os.path.join(a.src, stem, "label", "m0.png"), os.path.join(base, "label", "0_00000.png")),
                  (os.path.join(a.src, stem, "input", "m0.png"), os.path.join(base, "input", "0_00000.png"))]
        missing = [s for s, _ in pairs if not os.path.exists(s)]
        if missing:
            print(f"  {stem}: {len(missing)} file(s) missing (e.g. {missing[0]}), skipped")
            continue
        for s, d in pairs:
            if not os.path.exists(d):
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(s, d)
        n_ok += 1
    print(f"wired {n_ok}/{len(stems)} samples x {a.n_draws} draws -> {a.dst}")


if __name__ == "__main__":
    main()
