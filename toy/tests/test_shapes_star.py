import numpy as np
import pytest

from toy.shapes import Star, circumradius_for_star_area

TARGET_AREA = np.pi * 400.0          # equal-area convention: the r=20 circle


def test_circumradius_for_star_area_inverts_the_area_formula():
    R = circumradius_for_star_area(TARGET_AREA)
    analytic = 5.0 * R * (R * 0.382) * np.sin(np.pi / 5)
    assert analytic == pytest.approx(TARGET_AREA, rel=1e-9)
    assert R == pytest.approx(33.4564, abs=1e-3)


@pytest.mark.parametrize("theta", [0.0, 0.3, 0.7, 1.2])
def test_rendered_area_matches_the_analytic_area(theta):
    """Coverage sums to the analytic area regardless of rotation."""
    R = circumradius_for_star_area(TARGET_AREA)
    star = Star(cx=128.0, cy=128.0, R=R, ratio=0.382, theta=theta, intensity=255)
    cov = star.coverage(size=256, supersample=4)
    assert cov.sum() == pytest.approx(TARGET_AREA, rel=0.01)
    assert star.area == pytest.approx(TARGET_AREA, rel=1e-9)


def test_boundary_radius_alternates_between_outer_and_inner():
    """The 10 polygon vertices sit alternately at R and ratio*R."""
    R = circumradius_for_star_area(TARGET_AREA)
    ratio = 0.382
    star = Star(cx=0.0, cy=0.0, R=R, ratio=ratio, theta=0.0, intensity=255)
    for k in range(10):
        ang = k * np.pi / 5
        lo, hi = 0.0, 60.0
        for _ in range(60):                       # bisect the boundary along this ray
            mid = 0.5 * (lo + hi)
            if star._inside(np.array(mid * np.cos(ang)), np.array(mid * np.sin(ang))):
                lo = mid
            else:
                hi = mid
        expected = R if k % 2 == 0 else R * ratio
        assert lo == pytest.approx(expected, abs=1e-3)


def test_coverage_is_a_fraction_and_anti_aliased():
    R = circumradius_for_star_area(TARGET_AREA)
    cov = Star(128.0, 128.0, R, 0.382, 0.0, 255).coverage(256, 4)
    assert cov.shape == (256, 256)
    assert cov.min() >= 0.0 and cov.max() <= 1.0
    assert np.any((cov > 0.0) & (cov < 1.0)), "no partial coverage: edges are not anti-aliased"


def test_circumradius_property_is_the_outer_radius():
    star = Star(10.0, 10.0, 33.0, 0.382, 0.0, 128)
    assert star.circumradius == 33.0
