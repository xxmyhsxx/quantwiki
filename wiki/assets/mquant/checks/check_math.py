"""Small algebra examples for the Wiki; no model or performance reproduction."""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(20260914)
results = {}

def close(name, a, b):
    error = float(np.max(np.abs(a - b)))
    assert error < 1e-10, (name, error)
    results[name] = error

def attention(q, k, v, mask):
    logits = q @ k.T / np.sqrt(q.shape[1]) + mask
    logits -= logits.max(axis=-1, keepdims=True)
    prob = np.exp(logits)
    return prob / prob.sum(axis=-1, keepdims=True) @ v

def rope(x, positions):
    y = x.copy()
    for pair in range(x.shape[1] // 2):
        angle = positions / (10000 ** (2 * pair / x.shape[1]))
        c, s = np.cos(angle), np.sin(angle)
        a, b = x[:, 2 * pair], x[:, 2 * pair + 1]
        y[:, 2 * pair] = c * a - s * b
        y[:, 2 * pair + 1] = s * a + c * b
    return y

q, k, v = (rng.normal(size=(4, 4)) for _ in range(3))
position = np.arange(4)
perm = np.array([1, 2, 0, 3])
mask = np.where(position[None, :] <= position[:, None], 0.0, -np.inf)
reordered_mask = mask[np.ix_(perm, perm)]
reference = attention(rope(q, position), rope(k, position), v, mask)[perm]
actual = attention(rope(q[perm], position[perm]), rope(k[perm], position[perm]), v[perm], reordered_mask)
close('attention_with_original_positions_and_permuted_mask', reference, actual)
bad_mask = attention(rope(q[perm], position[perm]), rope(k[perm], position[perm]), v[perm], mask)
bad_pos = attention(rope(q[perm], position), rope(k[perm], position), v[perm], reordered_mask)
for name, bad in [('naive_causal_mask_error', bad_mask), ('new_position_ids_error', bad_pos)]:
    results[name] = float(np.max(np.abs(reference - bad)))
    assert results[name] > 1e-3

h2 = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
h = np.kron(h2, h2)
a, b = rng.normal(size=(7, 4)), rng.normal(size=(4, 3))
close('orthogonality', h.T @ h, np.eye(4))
close('rotated_linear', a @ b, (a @ h) @ (h.T @ b))
ap, bp = a @ h, h.T @ b
close('channel_split', ap @ bp, ap[:, 1:] @ bp[1:, :] + ap[:, :1] @ bp[:1, :])
close('mean_concentration', (h @ b)[0], np.sqrt(4) * b.mean(axis=0))
results['constant_vector_peak_ratio'] = float(np.max(np.abs(np.ones(4) @ h)))
assert results['constant_vector_peak_ratio'] > 1.9

eps = 1e-5
def rms(x):
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
center = np.eye(4) - np.ones((4, 4)) / 4
centered = a - a.mean(axis=-1, keepdims=True)
ln = centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + eps)
close('layernorm_centering', ln, rms(a @ center))
close('rms_rotation', rms(a @ h), rms(a) @ h)

out = Path(__file__).with_name('math-validation.json')
out.write_text(json.dumps({'kind': 'teaching_algebra_only', 'checks': results}, indent=2) + '\n')
print(json.dumps(results, indent=2))
