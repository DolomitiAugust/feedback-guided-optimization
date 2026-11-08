#!/usr/bin/env python3
"""Initial candidate for packing 26 circles in the unit square."""

import json

import numpy as np


# EVOLVE_START
import math

import numpy as np

try:
    from scipy.optimize import linprog, minimize
except Exception:  # pragma: no cover
    linprog = None
    minimize = None


def _clean_centers(centers):
    """Return finite centers clipped to the unit square."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim == 1:
        centers = centers.reshape(-1, 2)
    centers = np.where(np.isfinite(centers), centers, 0.5)
    return np.clip(centers, 0.0, 1.0)


def _greedy_radii(centers, safety=0.0):
    """Deterministic feasible fallback for fixed centers."""
    centers = _clean_centers(centers)
    n = centers.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)

    limits = np.minimum.reduce(
        [
            centers[:, 0],
            centers[:, 1],
            1.0 - centers[:, 0],
            1.0 - centers[:, 1],
        ]
    )
    radii = np.maximum(limits - safety, 0.0)
    dist = np.sqrt(
        np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    )

    for _ in range(n * n + 5):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                rsum = radii[i] + radii[j]
                allowed = max(float(dist[i, j]) - safety, 0.0)
                if rsum > allowed and rsum > 0.0:
                    factor = allowed / rsum
                    radii[i] *= factor
                    radii[j] *= factor
                    changed = True
        if not changed:
            break

    radii *= 1.0 - 1e-13
    return np.maximum(radii, 0.0)


def _solve_lp_radii(centers, safety=0.0):
    """Maximize sum(r_i) for fixed centers by a linear program."""
    centers = _clean_centers(centers)
    n = centers.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)

    limits = np.minimum.reduce(
        [
            centers[:, 0],
            centers[:, 1],
            1.0 - centers[:, 0],
            1.0 - centers[:, 1],
        ]
    )
    limits = np.maximum(limits - safety, 0.0)

    dist = np.sqrt(
        np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    )

    if linprog is not None:
        pair_i, pair_j = np.triu_indices(n, 1)
        m = len(pair_i)
        if m > 0:
            A = np.zeros((m, n), dtype=float)
            A[np.arange(m), pair_i] = 1.0
            A[np.arange(m), pair_j] = 1.0
            b = np.maximum(dist[pair_i, pair_j] - safety, 0.0)
        else:
            A = None
            b = None

        try:
            res = linprog(
                -np.ones(n),
                A_ub=A,
                b_ub=b,
                bounds=[(0.0, float(lim)) for lim in limits],
                method="highs",
            )
            if res.success and res.x is not None:
                return np.clip(np.asarray(res.x, dtype=float), 0.0, limits)
        except Exception:
            pass

    return _greedy_radii(centers, safety)


def compute_max_radii(centers):
    """Compatibility wrapper for fixed-center radius maximization."""
    return _solve_lp_radii(centers, safety=0.0)


def _objective(z):
    return -float(np.sum(z[2::3]))


def _objective_jac(z):
    grad = np.zeros_like(z)
    grad[2::3] = -1.0
    return grad


def _make_constraints(n):
    """Build joint boundary and pairwise non-overlap constraints."""
    pair_i, pair_j = np.triu_indices(n, 1)

    def fun(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]

        boundary = np.concatenate(
            (
                x - r,
                1.0 - x - r,
                y - r,
                1.0 - y - r,
            )
        )

        dx = x[pair_i] - x[pair_j]
        dy = y[pair_i] - y[pair_j]
        rsum = r[pair_i] + r[pair_j]
        pairs = dx * dx + dy * dy - rsum * rsum
        return np.concatenate((boundary, pairs))

    def jac(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        m = len(pair_i)
        total = 4 * n + m
        J = np.zeros((total, 3 * n), dtype=float)

        idx = np.arange(n)
        J[idx, 3 * idx] = 1.0
        J[idx, 3 * idx + 2] = -1.0

        J[n + idx, 3 * idx] = -1.0
        J[n + idx, 3 * idx + 2] = -1.0

        J[2 * n + idx, 3 * idx + 1] = 1.0
        J[2 * n + idx, 3 * idx + 2] = -1.0

        J[3 * n + idx, 3 * idx + 1] = -1.0
        J[3 * n + idx, 3 * idx + 2] = -1.0

        if m > 0:
            rows = np.arange(m) + 4 * n
            dx = x[pair_i] - x[pair_j]
            dy = y[pair_i] - y[pair_j]
            rsum = r[pair_i] + r[pair_j]

            J[rows, 3 * pair_i] = 2.0 * dx
            J[rows, 3 * pair_j] = -2.0 * dx
            J[rows, 3 * pair_i + 1] = 2.0 * dy
            J[rows, 3 * pair_j + 1] = -2.0 * dy
            J[rows, 3 * pair_i + 2] = -2.0 * rsum
            J[rows, 3 * pair_j + 2] = -2.0 * rsum

        return J

    return fun, jac


def _halton(index, base):
    result = 0.0
    f = 1.0 / float(base)
    i = int(index)
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= float(base)
    return result


def _seed_fibonacci(n):
    centers = np.zeros((n, 2), dtype=float)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        t = (i + 0.5) / float(n) if n > 0 else 0.0
        rad = 0.47 * math.sqrt(t)
        ang = i * golden_angle + 0.37
        centers[i] = [0.5 + rad * math.cos(ang), 0.5 + rad * math.sin(ang)]
    return _clean_centers(centers)


def _adjust_count(points, n):
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(-1, 2)
    if len(points) >= n:
        return _clean_centers(points[:n])

    fib = _seed_fibonacci(n)
    out = np.vstack([points, fib[len(points) : n]])
    return _clean_centers(out[:n])


def _seed_triangular(n):
    counts = [6, 5, 5, 5, 5]
    pts = []
    for row, count in enumerate(counts):
        y = (row + 0.5) / float(len(counts))
        if count == 6:
            xs = (np.arange(count) + 0.5) / float(count)
        else:
            xs = (np.arange(count) + 1.0) / 6.0
        for x in xs:
            pts.append([float(x), y])
    return _adjust_count(pts, n)


def _seed_grid(n):
    pts = []
    for j in range(5):
        for i in range(5):
            pts.append([(i + 0.5) / 5.0, (j + 0.5) / 5.0])
    pts.append([0.52, 0.52])
    return _adjust_count(pts, n)


def _seed_ring(n):
    centers = np.zeros((n, 2), dtype=float)
    if n <= 0:
        return centers

    centers[0] = [0.5, 0.5]
    if n == 1:
        return centers

    inner = min(10, n - 1)
    for i in range(inner):
        ang = 2.0 * math.pi * i / float(inner) + math.pi / 10.0
        centers[1 + i] = [
            0.5 + 0.24 * math.cos(ang),
            0.5 + 0.24 * math.sin(ang),
        ]

    outer = n - 1 - inner
    for k in range(outer):
        ang = 2.0 * math.pi * k / float(outer) + math.pi / 7.0
        centers[1 + inner + k] = [
            0.5 + 0.43 * math.cos(ang),
            0.5 + 0.43 * math.sin(ang),
        ]

    return _clean_centers(centers)


def _seed_halton(n):
    centers = np.zeros((n, 2), dtype=float)
    for i in range(n):
        u = _halton(i + 1, 2)
        v = _halton(i + 1, 3)
        centers[i] = [0.05 + 0.90 * u, 0.05 + 0.90 * v]
    return _clean_centers(centers)


def _rotate_jitter(seed, angle, jitter_scale=0.0):
    centers = np.asarray(seed, dtype=float).copy()
    if centers.ndim == 1:
        centers = centers.reshape(-1, 2)

    ca = math.cos(angle)
    sa = math.sin(angle)
    x = centers[:, 0] - 0.5
    y = centers[:, 1] - 0.5
    centers[:, 0] = 0.5 + ca * x - sa * y
    centers[:, 1] = 0.5 + sa * x + ca * y

    if jitter_scale > 0.0:
        for i in range(centers.shape[0]):
            dx = (_halton(i + 1, 5) - 0.5) * jitter_scale
            dy = (_halton(i + 1, 7) - 0.5) * jitter_scale
            centers[i, 0] += dx
            centers[i, 1] += dy

    return _clean_centers(centers)


def _make_seeds(n):
    tri = _seed_triangular(n)
    grid = _seed_grid(n)
    ring = _seed_ring(n)
    fib = _seed_fibonacci(n)
    hal = _seed_halton(n)

    seeds = [
        tri,
        grid,
        ring,
        fib,
        hal,
        _rotate_jitter(tri, math.pi / 12.0, 0.012),
        _rotate_jitter(tri, -math.pi / 10.0, 0.008),
        _rotate_jitter(hal, math.pi / 8.0, 0.006),
    ]
    return [_adjust_count(seed, n) for seed in seeds]


def _repair_and_scale(centers, radii, safety=1e-12):
    """Apply a conservative global radius scale to guarantee validity."""
    centers = _clean_centers(centers)
    n = centers.shape[0]
    radii = np.asarray(radii, dtype=float).copy()

    if n == 0:
        return centers, np.zeros(0, dtype=float)

    if radii.shape != (n,) or not np.all(np.isfinite(radii)):
        radii = _solve_lp_radii(centers, safety)

    radii = np.maximum(radii, 0.0)

    limits = np.minimum.reduce(
        [
            centers[:, 0],
            centers[:, 1],
            1.0 - centers[:, 0],
            1.0 - centers[:, 1],
        ]
    )
    dist = np.sqrt(
        np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    )

    for _ in range(4):
        scale = 1.0

        for i in range(n):
            if radii[i] > 0.0:
                allowed = max(float(limits[i]) - safety, 0.0)
                scale = min(scale, allowed / radii[i])

        for i in range(n):
            for j in range(i + 1, n):
                rsum = radii[i] + radii[j]
                if rsum > 0.0:
                    allowed = max(float(dist[i, j]) - safety, 0.0)
                    scale = min(scale, allowed / rsum)

        if not np.isfinite(scale) or scale < 0.0:
            scale = 0.0

        if scale < 1.0:
            radii *= scale
        else:
            break

    radii *= 1.0 - 1e-13
    radii = np.maximum(radii, 0.0)
    return centers, radii


def construct_packing(num_circles: int = 26):
    """Return centers and radii for a valid packed circle configuration."""
    n = int(num_circles)
    if n <= 0:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)

    seeds = _make_seeds(n)
    best_centers = None
    best_radii = None
    best_sum = -float("inf")

    if minimize is not None:
        cfun, cjac = _make_constraints(n)
    else:
        cfun = cjac = None

    for seed in seeds:
        centers = _clean_centers(seed)
        radii = _solve_lp_radii(centers, safety=1e-10)

        initial_sum = float(np.sum(radii))
        if initial_sum > best_sum:
            best_sum = initial_sum
            best_centers = centers.copy()
            best_radii = radii.copy()

        if minimize is None:
            continue

        z0 = np.empty(3 * n, dtype=float)
        z0[0::3] = centers[:, 0]
        z0[1::3] = centers[:, 1]
        z0[2::3] = radii

        bounds = []
        for _ in range(n):
            bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])

        cons = {"type": "ineq", "fun": cfun, "jac": cjac}

        try:
            res = minimize(
                _objective,
                z0,
                method="SLSQP",
                jac=_objective_jac,
                bounds=bounds,
                constraints=[cons],
                options={"maxiter": 150, "ftol": 1e-11, "disp": False},
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                opt_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                opt_centers = _clean_centers(opt_centers)
                opt_radii = _solve_lp_radii(opt_centers, safety=1e-10)
                opt_sum = float(np.sum(opt_radii))
                if opt_sum > best_sum:
                    best_sum = opt_sum
                    best_centers = opt_centers.copy()
                    best_radii = opt_radii.copy()
        except Exception:
            pass

    if best_centers is None:
        best_centers = _clean_centers(_seed_fibonacci(n))
        best_radii = _solve_lp_radii(best_centers, safety=1e-10)
        best_sum = float(np.sum(best_radii))

    if minimize is not None:
        centers = _clean_centers(best_centers)
        radii = _solve_lp_radii(centers, safety=1e-10)

        z0 = np.empty(3 * n, dtype=float)
        z0[0::3] = centers[:, 0]
        z0[1::3] = centers[:, 1]
        z0[2::3] = radii

        bounds = []
        for _ in range(n):
            bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])

        cons = {"type": "ineq", "fun": cfun, "jac": cjac}

        try:
            res = minimize(
                _objective,
                z0,
                method="SLSQP",
                jac=_objective_jac,
                bounds=bounds,
                constraints=[cons],
                options={"maxiter": 80, "ftol": 1e-12, "disp": False},
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                polished_centers = np.column_stack(
                    (res.x[0::3], res.x[1::3])
                )
                polished_centers = _clean_centers(polished_centers)
                polished_radii = _solve_lp_radii(
                    polished_centers, safety=1e-10
                )
                if float(np.sum(polished_radii)) >= best_sum:
                    best_centers = polished_centers
                    best_radii = polished_radii
        except Exception:
            pass

    centers = _clean_centers(best_centers)
    radii = _solve_lp_radii(centers, safety=1e-12)
    centers, radii = _repair_and_scale(centers, radii, safety=1e-12)

    centers = np.asarray(centers, dtype=float).reshape(n, 2)
    radii = np.asarray(radii, dtype=float).reshape(n)
    radii = np.maximum(radii, 0.0)
    radii = np.where(np.isfinite(radii), radii, 0.0)
    return centers, radii
# EVOLVE_END


def main() -> None:
    """Serialize the constructed packing for the isolated evaluator."""
    centers, radii = construct_packing(26)
    print(
        json.dumps(
            {
                "centers": np.asarray(centers, dtype=float).tolist(),
                "radii": np.asarray(radii, dtype=float).tolist(),
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
