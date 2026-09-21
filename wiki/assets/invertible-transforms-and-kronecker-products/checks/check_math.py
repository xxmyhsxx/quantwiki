"""核对 Wiki 的教学代数与反例，不运行模型，不以差分冒充 STE 验证。"""
from pathlib import Path
import json
import numpy as np

results = {}
w = np.array([-7., 1., 1., 7.])
x = np.array([.01, 1., 1., .01])
rows = []
for ratio in (1., .5):
    lo, hi = w.min() * ratio, w.max() * ratio
    h = (hi - lo) / 7
    z = -np.round(lo / h)
    reconstructed = h * (np.clip(np.round(w / h) + z, 0, 7) - z)
    rows.append(dict(ratio=ratio, step=h, zero=z, weights=reconstructed.tolist(),
                     weight_sse=float(np.sum((w-reconstructed)**2)),
                     output=float(reconstructed @ x)))
assert np.allclose(rows[0]['weights'], [-8, 0, 0, 6])
assert np.allclose(rows[1]['weights'], [-4, 1, 1, 3])
assert np.allclose([r['weight_sse'] for r in rows], [4, 25])
assert np.allclose([r['output'] for r in rows], [-.02, 1.99])
results['omniquant_clipping_example'] = rows

rng = np.random.default_rng(42)
X, W, b = rng.normal(size=(5, 4)), rng.normal(size=(3, 4)), rng.normal(size=3)
delta = rng.normal(size=4)
P1 = np.array([[1., 2.], [0., 1.]])
P2 = np.array([[2., 0.], [1., 1.]])
P = np.kron(P1, P2)
inverse = np.linalg.inv(P)
affine_error = np.max(np.abs((X-delta) @ inverse @ (P @ W.T) + b + delta @ W.T - (X @ W.T+b)))
assert affine_error < 1e-12
results['affine_shift_and_weight_layout_max_error'] = float(affine_error)

V = np.array([[1., 2.], [3., 4.]])
fast = P1.T @ V @ P2
assert np.allclose(fast.ravel(), V.ravel() @ P)
assert np.allclose(fast.ravel(), [4, 2, 18, 8])
assert np.allclose(inverse, np.kron(np.linalg.inv(P1), np.linalg.inv(P2)))
assert np.allclose(np.linalg.cond(P), np.linalg.cond(P1)*np.linalg.cond(P2))
results['kronecker_row_major_example'] = fast.ravel().tolist()
results['kronecker_inverse_and_condition_identity'] = 'passed'

M = np.array([[1., 0.], [1., 1.]])
M = M / np.linalg.norm(M, axis=1, keepdims=True)
assert np.isclose((M @ M.T)[0, 1], 1/np.sqrt(2))
bad = np.diag([1., 1e-6])
assert np.isclose(np.linalg.cond(bad), 1e6)
assert np.linalg.det(np.eye(2)+np.diag([-1., 0.])) == 0
results['unit_rows_not_orthogonal'] = float((M @ M.T)[0, 1])
results['strictly_diagonally_dominant_condition_number'] = float(np.linalg.cond(bad))
results['diagonal_update_can_destroy_invertibility'] = True

B = np.array([[0., .6], [-.6, 0.]])
U = (np.eye(2)+B/2) @ np.linalg.inv(np.eye(2)-B/2)
V2 = np.array([[0., -1.], [1., 0.]])
s = np.array([.3, 2.])
S = U @ np.diag(s) @ V2.T
inverse_transpose = U @ np.diag(1/s) @ V2.T
assert np.allclose(U.T @ U, np.eye(2))
assert np.allclose(inverse_transpose, np.linalg.inv(S).T)
results['cayley_and_inverse_transpose'] = 'passed'

q, k = rng.normal(size=(4, 2)), rng.normal(size=(7, 2))
assert np.allclose((q @ np.linalg.inv(S).T) @ (k @ S).T, q @ k.T)
attention, values = rng.random((4, 7)), rng.normal(size=(7, 2))
assert np.allclose(attention @ (values @ S), (attention @ values) @ S)
gate, up, scales = rng.normal(size=(4, 3)), rng.normal(size=(4, 3)), np.array([.3, 2., 3.])
silu = gate/(1+np.exp(-gate))
assert np.allclose((silu * up)/scales, silu * (up/scales))
results['post_rope_qk_value_and_gated_scaling_identities'] = 'passed'

results['status'] = 'passed'
results['scope'] = 'Teaching algebra and counterexamples only; no model, kernel, optimizer convergence or empirical STE gradient validation.'
Path(__file__).with_name('math-validation.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(json.dumps(results, indent=2, ensure_ascii=False))
