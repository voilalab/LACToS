"""Seed python / numpy / torch (ABL_SEED, default 0), then run the target script as __main__.

  python tools/seeded_run.py evaluate.py --config ...

Keeps any random choice inside the scorer reproducible (nothing in the Table 1 runs
subsamples reconstructions, but the seed makes that true by construction).
"""
import os, random, runpy, sys
import numpy as np, torch
seed = int(os.environ.get("ABL_SEED", "0"))
np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
script = sys.argv[1]; sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.getcwd())
runpy.run_path(script, run_name="__main__")
