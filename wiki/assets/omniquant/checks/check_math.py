"""教学代数核对；不运行 OmniQuant，也不验证模型效果。"""
from pathlib import Path
import json
import numpy as np

rng = np.random.default_rng(7)
X, W, b = rng.normal(size=(4, 3)), rng.normal(size=(3, 2)), rng.normal(size=2)
s, delta = np.array([2., .5, 3.]), np.array([.2, -.3, .4])
original = X @ W + b
transformed = ((X - delta) / s) @ (s[:, None] * W) + (b + delta @ W)
affine_error = float(np.max(np.abs(original - transformed)))
assert affine_error < 1e-12

# Attention 点积处的成对缩放抵消；移到位置旋转之前不一定抵消。
q, k = np.array([1., 0.]), np.array([0., 1.])
R = np.array([[0., -1.], [1., 0.]])
D, paired = np.diag([2., 1.]), 2 * np.eye(2)
before = float(q @ R @ k)
unconstrained = float(q @ np.linalg.inv(D) @ R @ D @ k)
shared_pair = float(q @ np.linalg.inv(paired) @ R @ paired @ k)
assert before == -1 and unconstrained == -.5 and shared_pair == before

# V 平移跨过注意力加权和，需要概率行和为 1。
P = np.array([[.2, .3, .5], [.1, .6, .3]])
V = rng.normal(size=(3, 3))
v_original = P @ V @ W + b
v_transformed = (P @ ((V - delta) / s)) @ (s[:, None] * W) + b + delta @ W
v_error = float(np.max(np.abs(v_original - v_transformed)))
assert v_error < 1e-12
P_bad = P * 1.1
bad_original = P_bad @ V @ W + b
bad_transformed = (P_bad @ ((V - delta) / s)) @ (s[:, None] * W) + b + delta @ W
predicted_error = (1 - P_bad.sum(axis=1))[:, None] * (delta @ W)
assert np.allclose(bad_transformed - bad_original, predicted_error)
assert np.max(np.abs(predicted_error)) > 1e-4

# 相对范围与对齐到整数零点后的表示范围是两项不同量。
l, u, levels = -.7 * 4, .6 * 4, 7
h = (u-l)/levels
z = -np.round(l/h)
represented = [float(-h*z), float(h*(levels-z))]
assert not np.allclose(represented, [l, u])
assert abs(represented[0]-l) <= h/2 + 1e-12
assert abs(represented[1]-u) <= h/2 + 1e-12

result = {
    'status': 'passed',
    'scope': 'teaching algebra only; no model reproduction or empirical gradient validation',
    'affine_fusion_max_abs_error': affine_error,
    'rope_counterexample': {'original': before, 'independent_scales': unconstrained, 'shared_pair': shared_pair},
    'value_shift_max_abs_error': v_error,
    'unnormalized_attention_error_formula': 'verified',
    'requested_range': [l, u], 'represented_range': represented,
}
Path(__file__).with_name('math-validation.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
