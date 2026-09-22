"""Small CPU teaching checks for qera.md; no QERA/model code is executed.

Requires NumPy. Without --output this only prints results; --output writes JSON.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def truncated(matrix, rank):
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * singular[:rank]) @ vt[:rank], singular


def psd_sqrt(matrix):
    values, vectors = np.linalg.eigh(matrix)
    assert values.min() > 0
    return (vectors * np.sqrt(values)) @ vectors.T


def loss(residual, moment):
    return float(np.trace(residual.T @ moment @ residual))


def main():
    rng = np.random.default_rng(20260922)
    checks = []

    x = rng.normal(size=(17, 4)) + np.array([1, -2, 0, 0.5])
    error = rng.normal(size=(4, 3))
    moment = x.T @ x / len(x)
    root = psd_sqrt(moment)
    empirical = float(np.mean(np.sum((x @ error) ** 2, axis=1)))
    np.testing.assert_allclose(empirical, loss(error, moment), rtol=1e-12)
    np.testing.assert_allclose(empirical, np.linalg.norm(root @ error) ** 2)
    checks.append({"name": "empirical_trace_and_weighted_norm", "loss": empirical})

    approximation, singular = truncated(root @ error, 2)
    correction = np.linalg.solve(root, approximation)
    optimum = loss(error - correction, moment)
    np.testing.assert_allclose(optimum, np.sum(singular[2:] ** 2), atol=1e-12)
    assert np.linalg.matrix_rank(correction, tol=1e-10) == 2
    # Independent feasible rank-2 candidates must not beat the analytic minimum.
    for _ in range(100):
        candidate = rng.normal(size=(4, 2)) @ rng.normal(size=(2, 3))
        assert loss(error - candidate, moment) >= optimum - 1e-10
    checks.append({"name": "weighted_rank_constraint_and_tail_energy", "optimum": optimum})

    e = np.diag([2.0, 1.0])
    r = np.diag([0.01, 100.0])
    plain, _ = truncated(e, 1)
    t = psd_sqrt(r)
    weighted, _ = truncated(t @ e, 1)
    weighted = np.linalg.solve(t, weighted)
    np.testing.assert_allclose([loss(e - plain, r), loss(e - weighted, r)], [100, 0.04])
    checks.append({"name": "weight_error_vs_output_error", "plain": 100, "weighted": 0.04})

    r = np.array([[1.0, 0.9], [0.9, 1.0]])
    t = psd_sqrt(r)
    exact, _ = truncated(t @ e, 1)
    exact = np.linalg.solve(t, exact)
    diagonal, _ = truncated(e, 1)
    exact_loss = loss(e - exact, r)
    expected = (5 - np.sqrt(21.96)) / 2
    np.testing.assert_allclose(exact_loss, expected, atol=1e-12)
    np.testing.assert_allclose(loss(e - diagonal, r), 1.0)
    checks.append({"name": "off_diagonal_moment_counterexample", "exact": exact_loss, "approx": 1.0})

    # Independent coordinates on this empirical product distribution have means 1, 2.
    noncentered = np.array([[0, 1], [0, 3], [2, 1], [2, 3]], dtype=float)
    mean = noncentered.mean(axis=0)
    covariance = (noncentered - mean).T @ (noncentered - mean) / len(noncentered)
    second = noncentered.T @ noncentered / len(noncentered)
    np.testing.assert_allclose(covariance[0, 1], 0)
    np.testing.assert_allclose(second[0, 1], 2)
    np.testing.assert_allclose(second, covariance + np.outer(mean, mean))
    checks.append({"name": "zero_covariance_is_not_zero_cross_moment", "cross_moment": 2.0})

    ux, sx, vtx = np.linalg.svd(x, full_matrices=False)
    factor = np.diag(sx) @ vtx / np.sqrt(len(x))
    np.testing.assert_allclose(factor.T @ factor, moment, atol=1e-12)
    assert not np.allclose(factor, factor.T)
    projected, _ = truncated(ux.T @ (x @ error), 2)
    alternative = (vtx.T / sx) @ projected
    np.testing.assert_allclose(alternative, correction, atol=1e-11)
    # Countercheck the missing inverse singular values in the printed appendix.
    incorrect = (vtx.T * sx) @ projected
    assert loss(error - incorrect, moment) > optimum + 1
    checks.append({"name": "nonsymmetric_factor_and_inverse_singular_values", "loss": loss(error - alternative, moment)})

    singular_moment = np.diag([1.0, 0.0])
    c = np.diag([2.0, 0.0])
    hidden = np.array([[0.0, 0.0], [7.0, -9.0]])
    np.testing.assert_allclose(loss(e - c, singular_moment), loss(e - c - hidden, singular_moment))
    assert np.linalg.matrix_rank(singular_moment) == 1
    epsilon = 0.1
    np.testing.assert_allclose(
        loss(e - c, singular_moment + epsilon * np.eye(2)),
        loss(e - c, singular_moment) + epsilon * np.linalg.norm(e - c) ** 2,
    )
    checks.append({"name": "unobserved_direction_and_changed_ridge_objective", "ridge_loss": 0.1})

    factor_elements = 32 * (4096 + 4096)
    factor_bytes = factor_elements * 16 // 8
    added_bpw = factor_elements * 16 / (4096 * 4096)
    assert factor_elements == 262144 and factor_bytes == 524288
    assert added_bpw == 0.25 and 4.25 + added_bpw == 4.5
    checks.append({"name": "factor_storage_excludes_other_model_costs", "bytes": factor_bytes, "added_bpw": added_bpw})

    report = {
        "scope": "NumPy float64 CPU teaching examples, not official code or model validation",
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
