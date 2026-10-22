import numpy as np
import pytest

from toy.gen_data import (load_config,
    ANCHORS, LEVELS, SEED_BASE, TARGET_AREA, make_inlier, make_ood, make_shape,
    sample_objects,
)

CFG = load_config()
SIZE, SS = int(CFG["image_size"]), int(CFG["supersample"])
CLASS_NAMES = {"circle": "Circle", "square": "Square", "triangle": "RegularPolygon",
               "ellipse": "Ellipse", "star": "Star"}


def _quadrant(cx, cy):
    """Nearest anchor index: Q1=0, Q2=1, Q3=2, Q4=3."""
    return int(np.argmin([(cx - ax) ** 2 + (cy - ay) ** 2 for ax, ay in ANCHORS]))


def test_anchors_are_quadrant_centres_with_q1_top_right():
    assert ANCHORS == [(192.0, 64.0), (64.0, 64.0), (64.0, 192.0), (192.0, 192.0)]


def test_levels_grow_by_exactly_one_class():
    assert LEVELS[1] == ["circle", "square"]
    assert LEVELS[2] == ["circle", "square", "triangle"]
    assert LEVELS[3] == ["circle", "square", "triangle", "ellipse"]
    assert LEVELS[4] == ["circle", "square", "triangle", "ellipse", "star"]


@pytest.mark.parametrize("kind", ["circle", "square", "triangle", "ellipse", "star"])
def test_every_class_has_the_same_area(kind):
    """Equal-area convention: size is never a cue for any class."""
    shp = make_shape(kind, np.random.default_rng(0), 128.0, 128.0, 200)
    assert shp.coverage(SIZE, SS).sum() == pytest.approx(TARGET_AREA, rel=0.01), kind


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_inlier_has_one_object_per_quadrant_and_a_circle_at_q1(level):
    for i in range(50):
        objs = sample_objects(np.random.default_rng(SEED_BASE[level] + i), CFG, LEVELS[level])
        assert len(objs) == 4
        assert [_quadrant(o.cx, o.cy) for o in objs] == [0, 1, 2, 3]
        assert type(objs[0]).__name__ == "Circle"


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_ood_places_a_square_at_q1_and_nowhere_else_is_forced(level):
    for i in range(20):
        _, _, meta = make_ood(SEED_BASE[level] + 900_000 + i, CFG, LEVELS[level])
        assert meta["anomaly"]["kind"] == "Square"
        assert _quadrant(meta["anomaly"]["cx"], meta["anomaly"]["cy"]) == 0
        assert len(meta["objects"]) == 3
        assert sorted(_quadrant(o["cx"], o["cy"]) for o in meta["objects"]) == [1, 2, 3]


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_generation_is_deterministic(level):
    a, _ = make_inlier(SEED_BASE[level] + 7, CFG, LEVELS[level])
    b, _ = make_inlier(SEED_BASE[level] + 7, CFG, LEVELS[level])
    assert np.array_equal(a, b)


def test_render_output_is_uint8_256_square():
    img, _ = make_inlier(SEED_BASE[1], CFG, LEVELS[1])
    assert img.dtype == np.uint8 and img.shape == (256, 256)   # _save_rgb stacks to 3


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_ground_truth_mask_is_binary_and_is_the_square_footprint(level):
    img, mask, meta = make_ood(SEED_BASE[level] + 900_001, CFG, LEVELS[level])
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    from toy.shapes import Square
    a = meta["anomaly"]
    cov = Square(a["cx"], a["cy"], a["side"], a["theta"], a["intensity"]).coverage(SIZE, SS)
    assert np.array_equal(mask, ((cov >= 0.5) * 255).astype(np.uint8))
    assert 0.015 < (mask > 0).mean() < 0.025
    # Drawn last, so never occluded: every FULLY covered pixel shows exactly the
    # square's intensity. Edge pixels (0.5 <= coverage < 1) are in the mask but
    # are anti-aliased blends with whatever lies beneath, so they are excluded.
    interior = cov == 1.0
    assert interior.sum() > 0.8 * (mask > 0).sum()
    assert np.all(img[interior] == a["intensity"])


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_all_shapes_lie_fully_inside_the_frame(level):
    for i in range(20):
        img, _ = make_inlier(SEED_BASE[level] + i, CFG, LEVELS[level])
        assert img[0, :].max() == 0 and img[-1, :].max() == 0
        assert img[:, 0].max() == 0 and img[:, -1].max() == 0


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_non_q1_anchors_use_exactly_the_level_vocabulary(level):
    seen = set()
    for i in range(300):
        for o in sample_objects(np.random.default_rng(SEED_BASE[level] + i), CFG, LEVELS[level])[1:]:
            seen.add(type(o).__name__)
    assert seen == {CLASS_NAMES[k] for k in LEVELS[level]}


def test_intensities_stay_in_the_configured_range():
    lo, hi = int(CFG["inlier"]["intensity_min"]), int(CFG["inlier"]["intensity_max"])
    vals = [o.intensity for i in range(200)
            for o in sample_objects(np.random.default_rng(i), CFG, LEVELS[4])]
    assert min(vals) >= lo and max(vals) <= hi


# ── p-sweep: L1 vocabulary with P(square) at Q2-Q4 below 0.5 ────────────────
from toy.gen_data import (P_SWEEP, SEED_BASE_P, make_inlier_p,  # noqa: E402
                          make_ood_plus, sample_objects_p)


@pytest.mark.parametrize("p", [0.4, 0.3, 0.2, 0.1])
def test_p_sweep_square_frequency_at_q2_q4_matches_p(p):
    kinds = []
    for i in range(1500):
        objs = sample_objects_p(np.random.default_rng(SEED_BASE_P + i), CFG, p)
        assert type(objs[0]).__name__ == "Circle"                     # Q1 stays circle-only
        kinds += [type(o).__name__ for o in objs[1:]]
    frac = kinds.count("Square") / len(kinds)
    assert set(kinds) <= {"Circle", "Square"}
    assert frac == pytest.approx(p, abs=0.02)


def test_p_sweep_datasets_are_nested_and_share_every_other_draw():
    """Same seed at two p: identical positions, intensities and rotations; the
    squares at the lower p are a subset of the squares at the higher p."""
    for i in range(200):
        lo = sample_objects_p(np.random.default_rng(SEED_BASE_P + i), CFG, 0.1)
        hi = sample_objects_p(np.random.default_rng(SEED_BASE_P + i), CFG, 0.4)
        for a, b in zip(lo, hi):
            assert abs(a.cx - b.cx) < 5.1 and abs(a.cy - b.cy) < 5.1   # only border clip can differ
            assert a.intensity == b.intensity
            if type(a).__name__ == "Square":
                assert type(b).__name__ == "Square"
                assert a.theta == b.theta


def test_p_sweep_inlier_render_is_deterministic():
    a, _ = make_inlier_p(SEED_BASE_P + 3, CFG, 0.2)
    b, _ = make_inlier_p(SEED_BASE_P + 3, CFG, 0.2)
    assert np.array_equal(a, b)


def test_ood_plus_has_square_at_q1_and_at_least_one_square_elsewhere():
    from toy.shapes import Square
    for i in range(50):
        img, mask, meta = make_ood_plus(P_SWEEP["test_seed"] + i, CFG)
        assert meta["anomaly"]["kind"] == "Square"
        assert _quadrant(meta["anomaly"]["cx"], meta["anomaly"]["cy"]) == 0
        others = [o["kind"] for o in meta["objects"]]
        assert len(others) == 3 and "Square" in others and set(others) <= {"Circle", "Square"}
        a = meta["anomaly"]
        cov = Square(a["cx"], a["cy"], a["side"], a["theta"], a["intensity"]).coverage(SIZE, SS)
        assert np.array_equal(mask, ((cov >= 0.5) * 255).astype(np.uint8))


def test_existing_level_datasets_are_unchanged_by_the_p_sweep_code():
    """The default path must draw exactly as before, or quad_L1..L4 stop being reproducible."""
    from pathlib import Path
    from PIL import Image
    from toy.gen_data import DEFAULT_DATA_ROOT
    f = DEFAULT_DATA_ROOT / "quad_L1/train/good/00000.png"
    if not f.exists():
        pytest.skip("quad_L1 not generated")
    img, _ = make_inlier(SEED_BASE[1] + 0, CFG, LEVELS[1])
    assert np.array_equal(np.array(Image.open(f))[..., 0], img)
