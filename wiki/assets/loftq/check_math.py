"""CPU teaching checks for LoftQ; custom quantizer, no official/model code.

Requires NumPy. Prints JSON; optionally saves it with --output.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def quantize_absmax_2bit(weight):
    scale = float(np.abs(weight).max())
    if scale == 0:
        return np.zeros_like(weight)
    codes = np.clip(np.rint((weight / scale + 1) * 1.5), 0, 3)
    return (codes / 1.5 - 1) * scale


def balanced_svd(residual, rank):
    u, s, vt = np.linalg.svd(residual, full_matrices=False)
    root = np.sqrt(s[:rank])
    return u[:, :rank] * root, vt[:rank].T * root, s


def squared(weight, quantized, correction):
    return float(np.linalg.norm(weight - quantized - correction) ** 2)


def main():
    w = np.array([[-1.9, -0.6, 1.4], [0.4, -1.9, -0.4], [0.1, 0.5, 1.6]])
    q1 = quantize_absmax_2bit(w)
    a1, b1, s1 = balanced_svd(w - q1, 1)
    c1 = a1 @ b1.T
    f1 = squared(w, q1, c1)
    np.testing.assert_allclose(f1, np.sum(s1[1:] ** 2), atol=1e-12)
    assert f1 <= squared(w, q1, np.zeros_like(w))
    assert np.linalg.matrix_rank(c1) == 1
    checks = [{"name": "one_iteration_svd_optimum_and_tail", "loss": f1}]

    q2 = quantize_absmax_2bit(w - c1)
    a2, b2, s2 = balanced_svd(w - q2, 1)
    c2 = a2 @ b2.T
    f2 = squared(w, q2, c2)
    before_svd = squared(w, q2, c1)
    np.testing.assert_allclose(f2, np.sum(s2[1:] ** 2), atol=1e-12)
    assert f2 <= before_svd and f2 > f1
    np.testing.assert_allclose([f1, f2], [0.2680321475138723, 0.4653816359840354])
    checks.append({"name": "svd_improves_subproblem_but_iteration_can_worsen", "first": f1, "second": f2, "second_before_svd": before_svd})

    wrong_a, wrong_b, _ = balanced_svd(w - q2 - c1, 1)
    wrong = wrong_a @ wrong_b.T
    assert not np.allclose(wrong, c2)
    assert not np.allclose(q1, q2)
    assert not np.allclose(q1 + c2, q2 + c2)
    assert np.linalg.matrix_rank(c1 + c2, tol=1e-10) > 1
    checks.append({"name": "replace_residual_and_pair_same_iteration", "accumulated_rank": int(np.linalg.matrix_rank(c1 + c2))})

    rng = np.random.default_rng(20260922)
    a = rng.normal(size=(3, 2))
    b = np.zeros((4, 2))
    q = rng.normal(size=(3, 4))
    target = rng.normal(size=(3, 4))
    g = q + a @ b.T - target
    da, db = g @ b, g.T @ a
    assert np.allclose(da, 0) and np.linalg.norm(db) > 0
    assert np.allclose(g @ np.zeros_like(b), 0)
    assert np.allclose(g.T @ np.zeros_like(a), 0)
    epsilon = 1e-6
    direction = rng.normal(size=b.shape)
    objective = lambda matrix: 0.5 * np.linalg.norm(q + a @ matrix.T - target) ** 2
    numerical = (objective(b + epsilon * direction) - objective(b - epsilon * direction)) / (2 * epsilon)
    np.testing.assert_allclose(numerical, np.sum(db * direction), rtol=1e-8, atol=1e-8)
    checks.append({"name": "zero_factor_training_gradient_and_finite_difference", "b_gradient_norm": float(np.linalg.norm(db))})

    u, s, vt = np.linalg.svd(w - q1, full_matrices=False)
    unbalanced_a = u[:, :1]
    unbalanced_b = vt[:1].T * s[:1]
    np.testing.assert_allclose(a1 @ b1.T, unbalanced_a @ unbalanced_b.T)
    g = rng.normal(size=w.shape)
    assert not np.isclose(np.linalg.norm(g @ b1), np.linalg.norm(g @ unbalanced_b))
    gamma = 3.0
    assert not np.allclose(gamma * a1 @ b1.T, c1)
    np.testing.assert_allclose(gamma * (a1 / gamma) @ b1.T, c1)
    checks.append({"name": "equal_products_not_equal_factor_gradients_and_adapter_scale", "scale": gamma})

    bits32 = (4 * 4 + 28 * 2) / 32
    bits40 = (4 * 4 + 36 * 2) / 40
    assert bits32 == 2.25 and bits40 == 2.20
    np.testing.assert_allclose(1 - 0.29, 0.71)
    np.testing.assert_allclose(26.5 - 20.9, 5.6)
    checks.append({"name": "prefix_bit_budget_retained_ratio_and_reported_delta", "bits_32_layers": bits32, "bits_40_layers": bits40, "storage_reduction": 0.71, "table_5_delta_pp": 5.6})

    report = {
        "scope": "NumPy float64 CPU examples with a custom absmax quantizer; not official LoftQ or model validation",
        "date": "2026-09-22",
        "checks_passed": len(checks),
        "checks": checks,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
