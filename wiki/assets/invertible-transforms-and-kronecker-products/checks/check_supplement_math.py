"""核对第二轮新增教学推导；不运行模型或将差分用于验证 STE。"""
from pathlib import Path
import json
import numpy as np

rng = np.random.default_rng(915)
results = {}

# 固定网格：改善一个输入方向，并不保证另一个方向受益。
A = np.array([[1., .5], [0., 1.]])
w = np.array([.49, .49])
X = np.array([[1., 1.], [1., -1.]])
quant = lambda value: np.clip(np.round(value), -4, 3)
baseline = quant(w)
effective = np.linalg.solve(A, quant(A @ w))
assert np.allclose(X @ w, [.98, 0.])
assert np.allclose(X @ baseline, [0., 0.])
assert np.allclose(X @ effective, [1., 1.])
assert np.allclose([np.sum((baseline-w)**2), np.sum((effective-w)**2)], [.4802, .5002])
results['mixing_example'] = dict(float_output=(X @ w).tolist(),
    baseline_output=(X @ baseline).tolist(), transformed_output=(X @ effective).tolist(),
    original_coordinate_weight_sse=[float(np.sum((baseline-w)**2)), float(np.sum((effective-w)**2))])

# 对平滑代理目标独立差分，核对逆矩阵侧和权重侧的链式法则。
A = np.eye(3) + rng.normal(size=(3, 3)) * .1
X, W = rng.normal(size=(4, 3)), rng.normal(size=(3, 2))
target_z, target_t = rng.normal(size=(4, 3)), rng.normal(size=(3, 2))
inv = np.linalg.inv(A)
Z, T = X @ inv, A @ W
gz, gt = Z-target_z, T-target_t
analytic = -Z.T @ gz @ inv.T + gt @ W.T
def smooth_loss(matrix):
    return .5 * (np.sum((X @ np.linalg.inv(matrix)-target_z)**2)
                 + np.sum((matrix @ W-target_t)**2))
numerical = np.zeros_like(A)
epsilon = 1e-6
for i, j in np.ndindex(A.shape):
    perturb = np.zeros_like(A)
    perturb[i, j] = epsilon
    numerical[i, j] = (smooth_loss(A+perturb)-smooth_loss(A-perturb))/(2*epsilon)
gradient_error = float(np.max(np.abs(analytic-numerical)))
assert gradient_error < 1e-7
delta_a = rng.normal(size=A.shape)
cancel = (-Z @ delta_a @ inv) @ T + Z @ (delta_a @ W)
assert np.max(np.abs(cancel)) < 1e-12
results['smooth_inverse_chain_rule_max_error'] = gradient_error
results['unquantized_two_sided_differential_cancels'] = True

# GQA：4 个 Q heads、2 个 KV heads，每头使用不同的注意力矩阵。
heads, kv_heads, dh, query_len, key_len = 4, 2, 2, 3, 5
q = rng.normal(size=(heads, query_len, dh))
k, v = rng.normal(size=(kv_heads, key_len, dh)), rng.normal(size=(kv_heads, key_len, dh))
ph = np.array([[1., .2], [-.3, 1.2]])
pv = np.array([[.8, -.1], [.4, 1.1]])
po = np.eye(heads) + rng.normal(size=(heads, heads)) * .1
def softmax(scores):
    shifted = scores-scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp/exp.sum(axis=-1, keepdims=True)
expanded_k, expanded_v = np.repeat(k, 2, axis=0), np.repeat(v, 2, axis=0)
scores = q @ expanded_k.transpose(0, 2, 1)/np.sqrt(dh)
scores_new = (q @ np.linalg.inv(ph).T) @ np.repeat(k @ ph, 2, axis=0).transpose(0, 2, 1)/np.sqrt(dh)
assert np.allclose(scores, scores_new)
original = (softmax(scores) @ expanded_v).transpose(1, 0, 2)
transformed_values = softmax(scores_new) @ np.repeat(v @ pv, 2, axis=0)
online = po.T @ transformed_values.transpose(1, 0, 2)
r_o = np.kron(po, pv)
flat_o = original.reshape(query_len, heads*dh)
assert np.allclose(online.reshape(query_len, -1), flat_o @ r_o)
w_o = rng.normal(size=(6, heads*dh))
w_o_new = w_o @ np.linalg.inv(r_o).T
attention_error = float(np.max(np.abs(online.reshape(query_len, -1) @ w_o_new.T-flat_o @ w_o.T)))
assert attention_error < 1e-12
results['gqa_attention_value_head_output_pairing_max_error'] = attention_error

# 包含 D 的总变换方向；关闭量化时恢复原投影。
p1, p2 = np.array([[1., .2], [.1, 1.]]), np.array([[.7, -.2], [.3, 1.]])
p = np.kron(p1, p2)
d = np.diag([.5, 1., 2., 3.])
x, w = rng.normal(size=(5, 4)), rng.normal(size=(3, 4))
r = np.linalg.inv(d) @ p
w_new = w @ d @ np.linalg.inv(p).T
assert np.allclose(w_new, w @ np.linalg.inv(r).T)
assert np.allclose((x @ r) @ w_new.T, x @ w.T)
assert np.linalg.matrix_rank(p) == 4
counterexample = np.diag([1., 1., 1., 2.])
assert np.linalg.matrix_rank(counterexample) == 4
blocks = np.stack([counterexample[:2, :2].ravel(), counterexample[2:, 2:].ravel()])
assert np.linalg.matrix_rank(blocks) == 2  # 两个非零块不成比例。
results['diagonal_and_kronecker_weight_direction'] = 'passed'
results['full_rank_kronecker_and_nonrepresentable_block_example'] = 'passed'

z, a, b = rng.normal(size=(3, 5)), rng.normal(size=(3, 5)), rng.normal(size=(3, 5))
mse = lambda value: np.mean(value**2)
lhs = mse(z-a)+mse(z-b)
rhs = 2*mse(z-(a+b)/2)+.5*mse(a-b)
assert np.allclose(lhs, rhs)
results['two_teacher_mse_identity_max_error'] = float(abs(lhs-rhs))
results['status'] = 'passed'
results['scope'] = 'New teaching derivations only; no model, discrete-rounding derivative, optimizer convergence, export or kernel benchmark.'
Path(__file__).with_name('supplement-math-validation.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(json.dumps(results, indent=2, ensure_ascii=False))
