"""
Fast DPS sampler for the CT denoising setup (no mask -- measurement is x + sigma*n).

Replaces run_ct_reconstruction.sh's 16-jobs-per-image structure, which was pure
redundancy: with the refined_box mask returning all ones, the patch index changes
nothing about the inverse problem.

One process per GPU, each owning a shard of the (image, measurement) work list and
running many reconstructions in a single batched diffusion loop.
"""
from functools import partial
import os, sys, time, argparse, glob, hashlib
import yaml
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from util.img_utils import clear_color, mask_generator
from guided_diffusion.condition_methods import PosteriorSampling


class PosteriorSamplingPerSample(PosteriorSampling):
    """DPS step with a PER-SAMPLE residual norm.

    guided_diffusion/condition_methods.py calls torch.linalg.norm(difference)
    with no dim, which is a Frobenius norm over the WHOLE batch. The gradient
    for sample b is then d_b/||d_all|| instead of d_b/||d_b||, so the effective
    step size is zeta'/sqrt(B) -- it silently depends on batch size.

    Summing per-sample norms makes d(sum_b ||d_b||)/dx_a = d||d_a||/dx_a, which
    is the paper's rule and is batch-invariant (verified exact to 0 ulp).
    """
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        difference = measurement - self.operator.forward(x_0_hat, **kwargs)
        norms = torch.linalg.norm(difference.reshape(difference.shape[0], -1), dim=1)
        norm_grad = torch.autograd.grad(outputs=norms.sum(), inputs=x_prev)[0]
        x_t = x_t - norm_grad * self.scale
        return x_t, norms.mean()


class PosteriorSamplingSquared(PosteriorSampling):
    """Naive DPS step: fixed step size on the SQUARED residual norm.

    This is Algorithm 1 line 8 with a constant zeta_i, i.e. without the
    zeta_i = zeta'/||y - A(x_hat_0)|| normalization the paper's footnote
    recommends. Included to measure what that normalization buys.
    """
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        difference = measurement - self.operator.forward(x_0_hat, **kwargs)
        norm = torch.linalg.norm(difference) ** 2
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t = x_t - norm_grad * self.scale
        return x_t, norm


def load_yaml(p):
    with open(p) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_config', default='configs/ct_model_config.yaml')
    ap.add_argument('--diffusion_config', default='configs/diffusion_config.yaml')
    ap.add_argument('--task_config', default='configs/ct_inpainting_config.yaml')
    ap.add_argument('--model_path', default=None, help='override checkpoint path')
    ap.add_argument('--data_root', default='data/CT_images')
    ap.add_argument('--save_dir', default='outputs/CT_denoise_0.2')
    ap.add_argument('--num_measurements', type=int, default=4)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--scale', type=float, default=None,
                    help="override conditioning scale; paper's zeta' = scale/2 in normalized mode")
    ap.add_argument('--grad_norm', choices=['per_sample', 'batch'], default='per_sample',
                    help="per_sample = correct, batch-invariant (default); "
                         "batch = reproduce the original repo's batch-coupled norm")
    ap.add_argument('--grad_mode', choices=['norm', 'squared'], default='norm',
                    help="norm = paper's normalized step (zeta_i = zeta'/||y-A(x0)||, what the repo "
                         "already does); squared = naive fixed step on ||y-A(x0)||^2")
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=99)
    ap.add_argument('--images', default=None,
                    help='comma-separated image stems (e.g. image_0016,image_0023); overrides start/end')
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = load_yaml(args.model_config)
    if args.model_path:
        model_config['model_path'] = args.model_path
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    model = create_model(**model_config).to(device).eval()
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    cond_config = task_config['conditioning']
    if args.scale is not None:
        cond_config['params']['scale'] = args.scale
    if args.grad_mode == 'squared':
        cond_method = PosteriorSamplingSquared(operator=operator, noiser=noiser, **cond_config['params'])
    elif args.grad_norm == 'per_sample':
        cond_method = PosteriorSamplingPerSample(operator=operator, noiser=noiser, **cond_config['params'])
    else:
        cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    _sc = cond_config['params']['scale']
    if args.grad_mode == 'norm':
        _eff = _sc / 2 if args.grad_norm == 'per_sample' else _sc / 2 / (args.batch ** 0.5)
        log(f"conditioning: scale={_sc} grad_norm={args.grad_norm} -> effective zeta'={_eff:.4g}"
            + ('' if args.grad_norm == 'per_sample' else f' (batch-coupled, /sqrt({args.batch}))'))
    else:
        log(f'conditioning: un-normalized ||.||^2, scale={_sc}')
    sampler = create_sampler(**diffusion_config)

    # The measurement mask for this config is identically ones (verified); build it
    # once from the real generator rather than assuming, then reuse for every batch.
    probe = torch.zeros(1, 3, 256, 256, device=device)
    m1 = mask_generator(**measure_config['mask_opt'])(
        probe, box_coords=(0, 4, 0, 64), box_prob_multiplier=0.9, mask_path='data/mask.png')
    assert float((m1 == 0).float().sum()) == 0.0, \
        'mask is not all-ones -- this fast path assumes the noise-only setup'
    log(f'mask verified all-ones (noise-only, sigma={measure_config["noise"]["sigma"]})')

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    paths = sorted(glob.glob(os.path.join(args.data_root, '*.png')))
    if args.images:
        want = set(args.images.split(','))
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in want]
    else:
        def _idx(p):
            # image_0042.png -> 42; arbitrary names (e.g. P01809_OL.png) have no
            # index, so they bypass the start/end range filter entirely.
            try:
                return int(os.path.splitext(os.path.basename(p))[0].split('_')[1])
            except (IndexError, ValueError):
                return None
        paths = [p for p in paths
                 if _idx(p) is None or args.start <= _idx(p) <= args.end]

    # flat work list, sharded round-robin so every GPU gets a similar mix
    work = [(p, k) for p in paths for k in range(args.num_measurements)]
    work = [w for i, w in enumerate(work) if i % args.num_shards == args.shard]

    # resume: drop items already on disk
    def out_paths(path, k):
        stem = os.path.splitext(os.path.basename(path))[0]
        d = os.path.join(args.save_dir, stem)
        return d, os.path.join(d, 'recon', f'm{k}.png')

    todo = [(p, k) for (p, k) in work if not os.path.exists(out_paths(p, k)[1])]
    log(f'shard {args.shard}/{args.num_shards}: {len(todo)} of {len(work)} reconstructions to do '
        f'(batch={args.batch})')
    if not todo:
        return

    t_start = time.time()
    done = 0
    for bi in range(0, len(todo), args.batch):
        chunk = todo[bi:bi + args.batch]
        # deterministic per (image, measurement) so runs are reproducible
        imgs = []
        for p, k in chunk:
            imgs.append(tf(Image.open(p).convert('RGB')))
        ref = torch.stack(imgs).to(device)
        B = ref.shape[0]
        mask = m1[:, :1].repeat(B, 1, 1, 1)

        # Seed per (image, draw) rather than per batch, so a given sample is
        # reproducible regardless of batch size, shard count or resume point.
        noises, starts = [], []
        for p, k in chunk:
            stem = os.path.splitext(os.path.basename(p))[0]
            try:                      # image_0042 -> 42
                idx = int(stem.split('_')[1])
            except (IndexError, ValueError):
                idx = 0
            if not stem.startswith('image_'):
                # arbitrary filenames (e.g. 0213_33_0040): hash the stem so every
                # (image, draw) still gets a distinct, reproducible noise draw
                idx = int(hashlib.md5(stem.encode()).hexdigest()[:8], 16) % 1_000_000
            gg = torch.Generator(device=device).manual_seed(
                args.seed * 1_000_003 + idx * 1_009 + k)
            noises.append(torch.randn(ref.shape[1:], device=device, generator=gg))
            starts.append(torch.randn(ref.shape[1:], device=device, generator=gg))

        y = operator.forward(ref, mask=mask)
        y_n = y + noiser.sigma * torch.stack(noises)

        cond_fn = partial(cond_method.conditioning, mask=mask)
        x_start = torch.stack(starts).requires_grad_()
        sample = sampler.p_sample_loop(model=model, x_start=x_start, measurement=y_n,
                                       measurement_cond_fn=cond_fn, record=False, save_root=None)

        for b, (p, k) in enumerate(chunk):
            d, rp = out_paths(p, k)
            for sub in ('label', 'input', 'recon'):
                os.makedirs(os.path.join(d, sub), exist_ok=True)
            plt.imsave(os.path.join(d, 'label', f'm{k}.png'), clear_color(ref[b]))
            plt.imsave(os.path.join(d, 'input', f'm{k}.png'), clear_color(y_n[b]))
            plt.imsave(rp, clear_color(sample[b]))

        done += B
        el = time.time() - t_start
        log(f'shard {args.shard}: {done}/{len(todo)} done | {el/done:.1f}s per recon | '
            f'eta {(len(todo)-done)*el/done/60:.1f} min')

    log(f'shard {args.shard}: finished {done} reconstructions in {(time.time()-t_start)/60:.1f} min')


if __name__ == '__main__':
    main()
