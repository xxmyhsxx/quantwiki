"""CPU teaching checks for OSTQuant v1 interpretation, not official model code.
Run with NumPy; optional --output writes this run's JSON explicitly.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    rng = np.random.default_rng(250113987)
    checks = []
    def record(name, **values):
        checks.append(dict(name=name, passed=True, **values))
    def close(a, b):
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)
    O = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    S = np.diag([0.5, 1.0, 2.0, 3.0])
    Si = np.diag(1 / np.diag(S))
    X, W = rng.normal(size=(7, 4)), rng.normal(size=(4, 3))
    close((X @ O @ S) @ (Si @ O.T @ W), X @ W)
    wrong = np.linalg.norm((X @ O @ S) @ (O.T @ Si @ W) - X @ W)
    assert wrong > 1
    close((O @ S).T @ (O @ S), S @ S)
    close(np.linalg.cond(O @ S), 6.0)
    A = rng.normal(size=(5, 7))
    close((A @ X @ O @ S) @ (Si @ O.T @ W), A @ X @ W)
    record('ordered_transform_and_head_pair', wrong_inverse_error=float(wrong), condition_number=6.0)

    def rms(x):
        return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + 1e-6)
    close(rms(X @ O), rms(X) @ O)
    assert np.linalg.norm(rms(X @ S) - rms(X) @ S) > 1
    R = np.array([[0., -1.], [1., 0.]])
    q, k = np.array([1., 0.]), np.array([0., 1.])
    D = np.diag([2., 1.])
    before = q @ R @ k
    invalid = q @ np.linalg.inv(D) @ R @ D @ k
    close(before, -1.)
    close(invalid, -0.5)
    close(q @ (0.5*np.eye(2)) @ R @ (2*np.eye(2)) @ k, before)
    record('rms_and_rope_fusion_boundaries', original_dot=float(before), untied_scale_dot=float(invalid))

    H = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
    U = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2)
    lam = np.diag([4., 1.])
    cov = U @ lam @ U.T
    close(np.diag(cov), [2.5, 2.5])
    # Exact extremum of each coordinate on x = U sqrt(lam) u, ||u|| <= 1.
    root = U @ np.sqrt(lam)
    for j in range(2):
        u = root[j] / np.linalg.norm(root[j])
        close((root @ u)[j], np.sqrt(cov[j, j]))
    true_ratio = np.pi * np.sqrt(np.linalg.det(cov)) / (4*np.max(np.diag(cov)))
    axis_ratio = np.pi * 2 / (4 * 2)
    close(true_ratio, np.pi/5)
    close(axis_ratio, np.pi/4)
    whiten = np.diag([0.5, 1.]) @ U.T
    close(whiten @ cov @ whiten.T, np.eye(2))
    rotated = H @ U.T @ cov @ U @ H.T
    close(np.diag(rotated), [2.5, 2.5])
    close(np.linalg.eigvalsh(rotated), [1., 4.])
    assert not np.allclose(rotated, 2.5*np.eye(2))
    for _ in range(40):
        B = rng.normal(size=(4,4))
        C = B @ B.T + 0.1*np.eye(4)
        assert np.linalg.det(C) <= np.max(np.diag(C))**4 + 1e-10
    record('ellipsoid_extrema_womi_and_whitening', true_ratio=float(true_ratio), axis_endpoint_ratio=float(axis_ratio), true_extent=float(np.sqrt(2.5)), endpoint_extent=float(np.sqrt(2)))

    # Diagonal scaling crosses the elementwise gate; general mixing does not.
    gate, up = rng.normal(size=(7,2)), rng.normal(size=(7,2))
    S2 = np.diag([0.5, 2.])
    W2 = rng.normal(size=(2,3))
    silu = lambda x: x / (1+np.exp(-x))
    z = silu(gate)*up
    close(silu(gate)*(up @ S2), z @ S2)
    close((z @ S2 @ H) @ (H.T @ np.linalg.inv(S2) @ W2), z @ W2)
    bad = np.linalg.norm(silu(gate @ H) - silu(gate) @ H)
    assert bad > 0.1
    record('ffn_scaling_hadamard_order', nonlinear_commutation_error=float(bad))

    def softmax(a):
        e = np.exp(a - np.max(a))
        return e/e.sum()
    p = np.array([0.6,0.3,0.1])
    a = np.array([0.2,-0.3,0.7])
    idx = np.array([0,1])
    mass = p[idx].sum()
    def partial(b):
        qq = softmax(b)
        return np.sum(p[idx]*np.log(p[idx]/qq[idx]))
    def conditional(b):
        pp, qq = p[idx]/mass, softmax(b[idx])
        return np.sum(pp*np.log(pp/qq))
    def fd(f):
        h=1e-6
        return np.array([(f(a+np.eye(3)[j]*h)-f(a-np.eye(3)[j]*h))/(2*h) for j in range(3)])
    grad = mass*softmax(a)
    grad[idx] -= p[idx]
    np.testing.assert_allclose(fd(partial),grad,atol=2e-10,rtol=2e-9)
    cgrad = np.zeros(3)
    cgrad[idx]=softmax(a[idx])-p[idx]/mass
    np.testing.assert_allclose(fd(conditional),cgrad,atol=2e-10,rtol=2e-9)
    assert grad[2] > 0 and cgrad[2] == 0
    negative = float(0.6*np.log(0.6/0.8))
    assert negative < 0
    record('partial_vs_conditional_kl', partial_gradient=grad.tolist(), conditional_gradient=cgrad.tolist(), top_one_partial_loss=negative, top_one_conditional_loss=0.0)

    retention = 64.37/65.21
    gap_reduction = (65.37-64.10)/(68.09-64.10)
    assert retention < 0.995
    assert 0.318 < gap_reduction < 0.319
    record('reported_metric_arithmetic', llama2_7b_w4_retention=float(retention), llama3_8b_gap_reduction=float(gap_reduction))
    result = dict(scope='NumPy float64 CPU teaching calculations; no official optimizer, model training, export, or GPU runtime', source='arXiv:2501.13987v1', date='2026-09-22', numpy_version=np.__version__, checks=checks)
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output:
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
