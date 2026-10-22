import numpy as np
import pytest
from PIL import Image

from toy.score_from_recons import (
    MODES, TEST_ORIGIN, apply_mode, check_method_mode, load_recon_stack, reference_order,
)

# What ood/data.py:load_reconstructions (and so run_lactos.py) feeds LACToS:
# files sorted as STRINGS, "<seed>_0_00000.png".
LEXICOGRAPHIC_20 = [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def _bank_dir(root, sid, seg, sub):
    d = root / f"samples_{sid}" / f"{seg}_0_4" / "inpainting" / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_png_stack(root, sid, seg, arrs):
    d = _bank_dir(root, sid, seg, "recon")
    for seed, a in enumerate(arrs):
        Image.fromarray(a, mode="RGB").save(d / f"{seed}_0_00000.png")


def _write_float_stack(root, sid, seg, arrs):
    d = _bank_dir(root, sid, seg, "recon_float")
    for seed, a in enumerate(arrs):
        np.save(d / f"{seed}_0_00000.npy", a)


def test_modes_are_exactly_the_three_specified():
    assert MODES == ("single", "mean", "set")


def test_recon_dir_segment_matches_what_generate_recons_writes():
    """additive_dps writes under ddad_dps_whole, NOT additive_dps."""
    assert TEST_ORIGIN == "ddad_dps_whole"


def test_reference_order_is_the_filename_sort_run_lactos_uses():
    assert reference_order(20) == LEXICOGRAPHIC_20
    assert reference_order(2) == [0, 1]


def test_load_recon_stack_returns_seeds_by_identity(tmp_path):
    """stack[k] is seed k, so `single` is unambiguously seed 0."""
    _write_png_stack(tmp_path, "000", "ddad_dps_whole",
                     [np.full((8, 8, 3), i * 10, np.uint8) for i in range(20)])
    stack = load_recon_stack(tmp_path, "000", "ddad_dps_whole")
    assert stack.shape == (20, 8, 8, 3) and stack.dtype == np.uint8
    assert [int(stack[k].max()) for k in range(20)] == [i * 10 for i in range(20)]


def test_load_recon_stack_float_reads_npy_by_seed_identity(tmp_path):
    arrs = [np.full((8, 8, 3), i / 20.0 - 0.5, np.float32) for i in range(20)]
    _write_float_stack(tmp_path, "000", "ddad_dps_whole", arrs)
    stack = load_recon_stack(tmp_path, "000", "ddad_dps_whole", kind="float")
    assert stack.shape == (20, 8, 8, 3) and stack.dtype == np.float32
    assert np.allclose(stack[:, 0, 0, 0], [i / 20.0 - 0.5 for i in range(20)])


def test_load_recon_stack_raises_when_the_directory_is_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_recon_stack(tmp_path, "000", "ddad_dps_whole")
    with pytest.raises(FileNotFoundError):
        load_recon_stack(tmp_path, "000", "ddad_dps_whole", kind="float")


def test_single_mode_takes_seed_zero_only():
    stack = np.stack([np.full((4, 4, 3), i, np.uint8) for i in range(20)])
    out = apply_mode(stack, "single")
    assert out.shape == (1, 4, 4, 3)
    assert np.array_equal(out[0], stack[0])


def test_mean_is_the_rounded_pixel_mean_of_every_seed_for_uint8():
    stack = np.random.default_rng(0).integers(0, 256, size=(20, 4, 4, 3), dtype=np.uint8)
    out = apply_mode(stack, "mean")
    assert out.shape == (1, 4, 4, 3) and out.dtype == np.uint8
    assert np.array_equal(out[0], np.rint(stack.astype(np.float64).mean(axis=0)).astype(np.uint8))


def test_mean_does_not_overflow_uint8():
    """A naive uint8 sum wraps at 256; the mean of twenty 250s must be 250."""
    assert int(apply_mode(np.full((20, 4, 4, 3), 250, np.uint8), "mean").max()) == 250


def test_mean_on_a_float_stack_is_the_exact_float_mean():
    """No rounding and no uint8 cast: DDAD reads float reconstructions."""
    stack = np.random.default_rng(1).uniform(-1.02, 1.02, size=(20, 4, 4, 3)).astype(np.float32)
    out = apply_mode(stack, "mean")
    assert out.shape == (1, 4, 4, 3) and out.dtype == np.float32
    assert np.allclose(out[0], stack.mean(axis=0), atol=1e-6)


def test_set_feeds_every_seed_in_run_lactos_file_order():
    """LACToS maps depend on the order of the 20 reconstructions, so set20 must
    reproduce ood/data.py:load_reconstructions' order to match run_lactos.py."""
    stack = np.stack([np.full((4, 4, 3), i, np.uint8) for i in range(20)])
    out = apply_mode(stack, "set")
    assert out.shape == (20, 4, 4, 3)
    assert [int(v) for v in out[:, 0, 0, 0]] == LEXICOGRAPHIC_20


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        apply_mode(np.zeros((20, 4, 4, 3), np.uint8), "median")


def test_ddad_rejects_set20_but_accepts_the_other_modes():
    """DDAD's heat_map compares ONE reconstruction to the input."""
    with pytest.raises(ValueError, match="set"):
        check_method_mode("ddad", "set")
    for method, mode in (("ddad", "single"), ("ddad", "mean"),
                         ("lactos", "mean"), ("lactos", "set")):
        check_method_mode(method, mode)


def test_check_method_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        check_method_mode("lactos", "median20")


# ── first-K subsets of a larger bank ────────────────────────────────────────
def test_first_k_reads_only_seeds_below_k(tmp_path):
    """Seeds >= K may still be being written by another job; they must not be read."""
    _write_png_stack(tmp_path, "000", "ddad_dps_whole",
                     [np.full((8, 8, 3), i, np.uint8) for i in range(20)])
    d = tmp_path / "samples_000" / "ddad_dps_whole_0_4" / "inpainting" / "recon"
    (d / "25_0_00000.png").write_bytes(b"not a png -- half written")
    stack = load_recon_stack(tmp_path, "000", "ddad_dps_whole", first_k=5)
    assert stack.shape[0] == 5
    assert [int(v) for v in stack[:, 0, 0, 0]] == [0, 1, 2, 3, 4]


def test_first_k_requires_every_seed_below_k(tmp_path):
    _write_png_stack(tmp_path, "000", "ddad_dps_whole",
                     [np.full((8, 8, 3), i, np.uint8) for i in range(20)])
    with pytest.raises(FileNotFoundError, match="seeds 0..24"):
        load_recon_stack(tmp_path, "000", "ddad_dps_whole", first_k=25)


def test_first_k_works_for_float_banks_too(tmp_path):
    _write_float_stack(tmp_path, "000", "ddad_dps_whole",
                       [np.full((8, 8, 3), i / 40.0, np.float32) for i in range(40)])
    stack = load_recon_stack(tmp_path, "000", "ddad_dps_whole", kind="float", first_k=10)
    assert stack.shape[0] == 10 and np.allclose(stack[:, 0, 0, 0], [i / 40.0 for i in range(10)])


def test_set_mode_on_forty_seeds_uses_the_filename_order():
    stack = np.stack([np.full((4, 4, 3), i, np.uint8) for i in range(40)])
    out = apply_mode(stack, "set")
    assert [int(v) for v in out[:, 0, 0, 0]] == sorted(range(40), key=lambda s: f"{s}_0_00000.png")
    assert reference_order(10) == list(range(10))           # single digits: numeric == filename order
