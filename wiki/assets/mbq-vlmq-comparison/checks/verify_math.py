"""Teaching/algebra checks for this ingest; no model or paper reproduction."""
from decimal import Decimal
import json
from pathlib import Path
import numpy as np

report = {}

# A counterexample to replacing max absolute gradient by its mean in an L1 bound.
grad, delta = np.array([2., 0.]), np.array([1., 0.])
actual = abs(grad @ delta)
mean_bound = np.abs(grad).mean() * np.abs(delta).sum()
assert actual > mean_bound
report['mbq_mean_gradient_counterexample'] = {'actual': actual, 'claimed_bound': mean_bound}

# Right multiplication before squaring gives squared token factors.
factors = np.array([1., 2.])
costs = (np.ones(2) * factors) ** 2
assert np.allclose(costs, [1., 4.])
report['token_factor_costs'] = costs.tolist()

# Compare the closed form with a separate constrained linear solve and finite differences.
rng = np.random.default_rng(20260914)
u = rng.normal(size=(4, 9))
f = u + rng.normal(scale=.15, size=(4, 9))
w = rng.normal(size=4)
t = rng.uniform(.1, 1.8, size=9)
r = w @ (f-u)
a_mat = (u * t**2) @ u.T
c = (u * t**2) @ r
b = np.linalg.inv(a_mat)
v = b @ c
q, constraint = 2, .3
closed = v + (constraint-v[q]) / b[q, q] * b[:, q]
free = [i for i in range(4) if i != q]
direct = np.zeros(4)
direct[q] = constraint
direct[free] = np.linalg.solve(a_mat[np.ix_(free, free)], c[free]-a_mat[free, q]*constraint)
assert np.allclose(closed, direct, rtol=1e-10, atol=1e-10)
elim = b-np.outer(b[:, q], b[q, :])/b[q, q]
assert np.allclose(elim[q], 0) and np.allclose(elim[:, q], 0)
assert np.allclose(elim[np.ix_(free, free)], np.linalg.inv(a_mat[np.ix_(free, free)]))
def objective(d):
    return float(np.sum(((d @ u-r)*t)**2))
h = 1e-4
fd_hessian = np.zeros((4, 4))
origin = rng.normal(size=4)
for i in range(4):
    for j in range(4):
        ei, ej = np.eye(4)[i]*h, np.eye(4)[j]*h
        fd_hessian[i, j] = (objective(origin+ei+ej)-objective(origin+ei-ej)
                            -objective(origin-ei+ej)+objective(origin-ei-ej))/(4*h*h)
assert np.allclose(fd_hessian, 2*a_mat, atol=2e-6, rtol=2e-6)
report['weighted_asymmetric_single_step'] = {
    'closed_vs_direct_max_error': float(np.max(np.abs(closed-direct))),
    'hessian_finite_difference_max_error': float(np.max(np.abs(fd_hessian-2*a_mat))),
    'constraint_error': float(abs(closed[q]-constraint)),
}

# Zero residual implies zero derivative for a deterministic squared reconstruction loss.
same_output = np.array([1., 2., 3.])
zero_grad = 2*(same_output-same_output)
assert np.count_nonzero(zero_grad) == 0
report['zero_residual_mse_gradient'] = zero_grad.tolist()

# Check printed averages against the eight task columns; not an evaluation rerun.
rows = {
 'qwen25_32b_gptq': ('74.20', '43.88 88.26 51.77 47.76 76.50 87.22 64.87 73.88'),
 'qwen25_32b_gptaq': ('73.68', '52.52 88.72 52.13 42.67 77.40 79.46 67.89 72.88'),
 'qwen25_32b_vlmq': ('74.31', '49.84 87.47 53.05 43.45 76.90 83.92 65.96 73.44'),
 'llava_05b_fp': ('63.46', '61.48 69.01 38.94 32.13 57.60 63.36 53.05 65.83'),
}
report['vlmq_table12_arithmetic'] = {}
for label, (printed, values) in rows.items():
    avg = sum(map(Decimal, values.split())) / Decimal(8)
    assert abs(avg-Decimal(printed)) > Decimal('.01')
    report['vlmq_table12_arithmetic'][label] = {'printed': printed, 'recomputed': str(avg)}

out = Path(__file__).with_name('math-check-results.json')
out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
