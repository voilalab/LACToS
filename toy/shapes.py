"""Shape primitives and the supersampled renderer for the synthetic benchmark.

Shapes report *coverage* (the fraction of each output pixel they occupy) rather
than a hard binary mask, so edges are anti-aliased. Hard-aliased edges are a
high-frequency artifact that a diffusion model wastes capacity on and that
ImageNet ResNet features respond to strongly — neither is what this benchmark
is trying to measure.
"""
from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np


def _downsample(fine: np.ndarray, size: int, supersample: int) -> np.ndarray:
    """Box-average a (size*s, size*s) array down to (size, size)."""
    s = supersample
    return fine.reshape(size, s, size, s).mean(axis=(1, 3))


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float
    intensity: int

    def coverage(self, size: int, supersample: int) -> np.ndarray:
        s = supersample
        n = size * s
        # +0.5 puts the sample at the centre of each fine pixel.
        ys = (np.arange(n, dtype=np.float64) + 0.5) / s
        xs = (np.arange(n, dtype=np.float64) + 0.5) / s
        dy = ys[:, None] - self.cy
        dx = xs[None, :] - self.cx
        inside = (dx * dx + dy * dy) <= (self.r * self.r)
        return _downsample(inside.astype(np.float64), size, s)


@dataclass(frozen=True)
class Ellipse:
    cx: float
    cy: float
    a: float          # semi-axis along the rotated x direction
    b: float          # semi-axis along the rotated y direction
    theta: float      # rotation in radians
    intensity: int

    def coverage(self, size: int, supersample: int) -> np.ndarray:
        s = supersample
        n = size * s
        ys = (np.arange(n, dtype=np.float64) + 0.5) / s
        xs = (np.arange(n, dtype=np.float64) + 0.5) / s
        dy = ys[:, None] - self.cy
        dx = xs[None, :] - self.cx
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        u = dx * cos_t + dy * sin_t
        v = -dx * sin_t + dy * cos_t
        inside = (u / self.a) ** 2 + (v / self.b) ** 2 <= 1.0
        return _downsample(inside.astype(np.float64), size, s)



@dataclass(frozen=True)
class Square:
    cx: float
    cy: float
    side: float
    theta: float      # rotation in radians
    intensity: int

    @property
    def circumradius(self) -> float:
        """Radius of the circumscribed circle — used for non-overlap placement."""
        return self.side * np.sqrt(2.0) / 2.0

    def coverage(self, size: int, supersample: int) -> np.ndarray:
        s = supersample
        n = size * s
        ys = (np.arange(n, dtype=np.float64) + 0.5) / s
        xs = (np.arange(n, dtype=np.float64) + 0.5) / s
        dy = ys[:, None] - self.cy
        dx = xs[None, :] - self.cx
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        u = dx * cos_t + dy * sin_t
        v = -dx * sin_t + dy * cos_t
        h = self.side / 2.0
        inside = (np.abs(u) <= h) & (np.abs(v) <= h)
        return _downsample(inside.astype(np.float64), size, s)



def circumradius_for_area(area: float, n_sides: int) -> float:
    """Circumradius of a regular `n_sides`-gon with the given area.

    Used to build inlier shape classes that all occupy the SAME area as the
    radius-20 circle, so that adding shape classes changes the distribution's
    shape-space complexity without touching occupancy or object size.
    """
    return float(np.sqrt(2.0 * area / (n_sides * np.sin(2.0 * np.pi / n_sides))))


@dataclass(frozen=True)
class RegularPolygon:
    cx: float
    cy: float
    R: float          # circumradius
    n_sides: int
    theta: float      # rotation in radians
    intensity: int

    @property
    def circumradius(self) -> float:
        return self.R

    @property
    def area(self) -> float:
        return 0.5 * self.n_sides * self.R ** 2 * float(np.sin(2.0 * np.pi / self.n_sides))

    def coverage(self, size: int, supersample: int) -> np.ndarray:
        s = supersample
        n = size * s
        ys = (np.arange(n, dtype=np.float64) + 0.5) / s
        xs = (np.arange(n, dtype=np.float64) + 0.5) / s
        dy = ys[:, None] - self.cy
        dx = xs[None, :] - self.cx
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        u = dx * cos_t + dy * sin_t
        v = -dx * sin_t + dy * cos_t
        r = np.hypot(u, v)
        # Boundary radius of a regular n-gon at polar angle phi:
        #   rho(phi) = R cos(pi/n) / cos((phi mod 2pi/n) - pi/n)
        step = 2.0 * np.pi / self.n_sides
        phi = np.mod(np.arctan2(v, u), step) - step / 2.0
        rho = self.R * np.cos(np.pi / self.n_sides) / np.cos(phi)
        return _downsample((r <= rho).astype(np.float64), size, s)


def circumradius_for_star_area(area: float, ratio: float = 0.382,
                               n_points: int = 5) -> float:
    """Outer radius of an `n_points`-pointed star with the given area.

    The star is the 2*n_points-gon whose vertices alternate between radius R
    and radius ratio*R, so its area is n_points * R * (ratio*R) * sin(pi/n).
    `ratio=0.382` is the pentagram value (1/phi^2), which reads as a "star"
    rather than a blunt decagon.
    """
    return float(np.sqrt(area / (n_points * ratio * np.sin(np.pi / n_points))))


@dataclass(frozen=True)
class Star:
    cx: float
    cy: float
    R: float          # outer (circum) radius
    ratio: float      # inner/outer radius ratio
    theta: float      # rotation in radians
    intensity: int
    n_points: int = 5

    @property
    def circumradius(self) -> float:
        return self.R

    @property
    def area(self) -> float:
        return float(self.n_points * self.R * (self.ratio * self.R)
                     * np.sin(np.pi / self.n_points))

    def _inside(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Point-in-star for offsets from the centre.

        Uses the star's 2*n_points-fold reflection symmetry: fold the polar
        angle into one half-period [0, gamma], where angle 0 points at an inner
        vertex and angle gamma at an outer one, then test the single straight
        edge joining them. The line through A=(r, 0) and B=(R cos g, R sin g)
        has unit normal n and offset d = A . n; a point is inside iff its
        projection on n does not exceed d.
        """
        gamma = np.pi / self.n_points
        r = self.R * self.ratio
        ax, ay = r, 0.0
        bx, by = self.R * np.cos(gamma), self.R * np.sin(gamma)
        nx, ny = (by - ay), -(bx - ax)            # (B - A) rotated by -90 deg
        norm = float(np.hypot(nx, ny))
        nx, ny = nx / norm, ny / norm
        d = ax * nx + ay * ny
        rho = np.hypot(dx, dy)
        phi = np.arctan2(dy, dx) - self.theta
        psi = np.abs(np.mod(phi, 2.0 * gamma) - gamma)     # fold into [0, gamma]
        return rho * (np.cos(psi) * nx + np.sin(psi) * ny) <= d

    def coverage(self, size: int, supersample: int) -> np.ndarray:
        s = supersample
        n = size * s
        ys = (np.arange(n, dtype=np.float64) + 0.5) / s
        xs = (np.arange(n, dtype=np.float64) + 0.5) / s
        dy = ys[:, None] - self.cy
        dx = xs[None, :] - self.cx
        return _downsample(self._inside(dx, dy).astype(np.float64), size, s)


Shape = Union[Circle, Ellipse, Square, RegularPolygon, Star]


def render(shapes: Sequence[Shape], size: int, supersample: int) -> np.ndarray:
    """Composite `shapes` onto a black canvas, later shapes over earlier ones.

    Overlap is resolved by painter's algorithm: the last shape covering a pixel
    determines its value, blended by that shape's coverage fraction so edges
    stay anti-aliased.
    """
    canvas = np.zeros((size, size), dtype=np.float64)
    for shape in shapes:
        cov = shape.coverage(size, supersample)
        canvas = canvas * (1.0 - cov) + float(shape.intensity) * cov
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)
