import numpy as np
import pytest

from toy.shapes import (
    Circle, Ellipse, RegularPolygon, Square, circumradius_for_area, render,
)


def test_circle_coverage_area_matches_pi_r_squared():
    """Anti-aliased coverage must integrate to the analytic area."""
    c = Circle(cx=128.0, cy=128.0, r=20.0, intensity=255)
    cov = c.coverage(size=256, supersample=4)
    assert cov.shape == (256, 256)
    assert cov.min() >= 0.0 and cov.max() <= 1.0
    assert cov.sum() == pytest.approx(np.pi * 20.0**2, rel=0.01)


def test_ellipse_coverage_area_matches_pi_a_b():
    e = Ellipse(cx=128.0, cy=128.0, a=30.0, b=12.0, theta=0.0, intensity=200)
    cov = e.coverage(size=256, supersample=4)
    assert cov.sum() == pytest.approx(np.pi * 30.0 * 12.0, rel=0.01)


def test_ellipse_rotation_preserves_area():
    straight = Ellipse(128.0, 128.0, 30.0, 12.0, 0.0, 200).coverage(256, 4)
    turned = Ellipse(128.0, 128.0, 30.0, 12.0, np.pi / 3, 200).coverage(256, 4)
    assert turned.sum() == pytest.approx(straight.sum(), rel=0.02)


def test_ellipse_rotation_changes_orientation():
    straight = Ellipse(128.0, 128.0, 30.0, 12.0, 0.0, 200).coverage(256, 4)
    turned = Ellipse(128.0, 128.0, 30.0, 12.0, np.pi / 2, 200).coverage(256, 4)
    # theta=0 is wide (a along x); theta=pi/2 is tall.
    assert (straight > 0.5).sum(axis=0).max() < (straight > 0.5).sum(axis=1).max()
    assert (turned > 0.5).sum(axis=0).max() > (turned > 0.5).sum(axis=1).max()


def test_render_is_uint8_and_black_background():
    img = render([Circle(50.0, 50.0, 20.0, 200)], size=256, supersample=4)
    assert img.dtype == np.uint8
    assert img.shape == (256, 256)
    assert img[200, 200] == 0
    assert img[50, 50] == 200


def test_render_uses_painters_algorithm():
    """A later shape overwrites an earlier one where they overlap."""
    first = Circle(128.0, 128.0, 20.0, 100)
    second = Circle(128.0, 128.0, 20.0, 250)
    img = render([first, second], size=256, supersample=4)
    assert img[128, 128] == 250


def test_render_empty_is_all_zero():
    assert render([], size=256, supersample=4).max() == 0


def test_square_coverage_area_matches_side_squared():
    sq = Square(cx=128.0, cy=128.0, side=34.0, theta=0.0, intensity=200)
    cov = sq.coverage(size=256, supersample=4)
    assert cov.shape == (256, 256)
    assert cov.min() >= 0.0 and cov.max() <= 1.0
    assert cov.sum() == pytest.approx(34.0 ** 2, rel=0.01)


def test_square_rotation_preserves_area():
    straight = Square(128.0, 128.0, 34.0, 0.0, 200).coverage(256, 4)
    turned = Square(128.0, 128.0, 34.0, np.pi / 5, 200).coverage(256, 4)
    assert turned.sum() == pytest.approx(straight.sum(), rel=0.02)


def test_square_at_45_degrees_is_a_diamond():
    """A square rotated 45 deg has its corners on the axes, so its axis-aligned
    bounding extent grows by sqrt(2) while its area is unchanged."""
    straight = Square(128.0, 128.0, 34.0, 0.0, 200).coverage(256, 4) > 0.5
    turned = Square(128.0, 128.0, 34.0, np.pi / 4, 200).coverage(256, 4) > 0.5
    assert turned.sum(axis=1).max() == pytest.approx(
        straight.sum(axis=1).max() * np.sqrt(2), rel=0.06)


def test_square_equal_area_to_the_benchmark_ellipse():
    """Side 34 is chosen so the square's area matches the ellipse (30x12),
    keeping the anomalous pixel budget - and hence AP's base rate - constant."""
    sq = Square(128.0, 128.0, 34.0, 0.0, 200).coverage(256, 4).sum()
    el = Ellipse(128.0, 128.0, 30.0, 12.0, 0.0, 200).coverage(256, 4).sum()
    assert sq == pytest.approx(el, rel=0.03)


def test_square_circumradius():
    assert Square(0.0, 0.0, 34.0, 0.0, 1).circumradius == pytest.approx(24.042, abs=1e-3)


@pytest.mark.parametrize("n_sides", [3, 4, 5, 6, 8])
def test_regular_polygon_area_matches_the_target(n_sides):
    """Every inlier shape class must occupy the same area as the r=20 circle,
    so adding classes changes shape-space complexity and nothing else."""
    target = np.pi * 20.0 ** 2
    R = circumradius_for_area(target, n_sides)
    poly = RegularPolygon(128.0, 128.0, R, n_sides, 0.0, 200)
    assert poly.area == pytest.approx(target, rel=1e-9)
    assert poly.coverage(256, 4).sum() == pytest.approx(target, rel=0.02)


def test_regular_polygon_many_sides_approaches_a_circle():
    target = np.pi * 20.0 ** 2
    R = circumradius_for_area(target, 64)
    poly = RegularPolygon(128.0, 128.0, R, 64, 0.0, 255).coverage(256, 4)
    circ = Circle(128.0, 128.0, 20.0, 255).coverage(256, 4)
    assert np.abs(poly - circ).sum() / circ.sum() < 0.02


def test_regular_polygon_rotation_preserves_area():
    R = circumradius_for_area(np.pi * 400, 3)
    a = RegularPolygon(128.0, 128.0, R, 3, 0.0, 200).coverage(256, 4).sum()
    b = RegularPolygon(128.0, 128.0, R, 3, 0.7, 200).coverage(256, 4).sum()
    assert b == pytest.approx(a, rel=0.02)


def test_triangle_is_not_a_circle():
    """Sanity: equal area must not mean equal shape, or the axis is vacuous."""
    R = circumradius_for_area(np.pi * 400, 3)
    tri = RegularPolygon(128.0, 128.0, R, 3, 0.0, 255).coverage(256, 4) > 0.5
    circ = Circle(128.0, 128.0, 20.0, 255).coverage(256, 4) > 0.5
    iou = (tri & circ).sum() / (tri | circ).sum()
    assert iou < 0.75
