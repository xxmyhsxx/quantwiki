"""Teaching checks of entropy definitions and local/global error; no model run."""
import json
from pathlib import Path
import numpy as np

def entropy(p):
    p = np.asarray(p)
    assert np.all(p >= 0) and np.isclose(p.sum(), 1)
    active = p[p > 0]
    return float(-np.sum(active * np.log2(active)))

def measures(joint):
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    hx, hy, hxy = entropy(px), entropy(py), entropy(joint)
    conditional = sum(px[i] * entropy(joint[i] / px[i]) for i in range(len(px)) if px[i] > 0)
    assert abs(hxy - hx - conditional) < 1e-12
    return {'H_U': hx, 'H_V': hy, 'H_V_given_U': conditional, 'I_U_V': hx + hy - hxy}

results = {}
results['identical_binary_variables'] = measures(np.diag([0.5, 0.5]))
results['independent_binary_variables'] = measures(np.full((2, 2), 0.25))
assert results['identical_binary_variables']['H_V_given_U'] == 0
assert results['independent_binary_variables']['H_V_given_U'] == 1
assert results['independent_binary_variables']['I_U_V'] == 0

# A tensor normalized by L2 is not in general a probability mass function.
x = np.ones(4)
l2 = x / np.linalg.norm(x)
results['l2_normalization_sum'] = float(l2.sum())
assert not np.isclose(l2.sum(), 1)
results['uniform_four_state_entropy_bits'] = entropy(x / x.sum())

# Different quantizer grids can prefer different first-layer rounding values.
first_candidates = np.array([1.1, 1.2])
local_errors = (first_candidates - 1.0) ** 2
joint_errors = ((5 / 3) * first_candidates - 2.0) ** 2
assert local_errors.argmin() == 0 and joint_errors.argmin() == 1
results['local_vs_final_error'] = {'first_layer_candidates': first_candidates.tolist(), 'local_squared_errors': local_errors.tolist(), 'final_squared_errors': joint_errors.tolist()}

# Soft probabilities are normalized; they do not by themselves define a joint law.
levels = np.array([-1.0, 0.0, 1.0])
logits = -(0.2 - levels) ** 2
prob = np.exp(logits - logits.max())
prob /= prob.sum()
assert np.isclose(prob.sum(), 1) and np.all(prob > 0)
results['soft_quantization_probabilities'] = prob.tolist()
results['scale_metadata_bits_per_weight'] = 8 / 64 + 32 / (64 * 256)

path = Path(__file__).with_name('math-validation.json')
path.write_text(json.dumps({'kind': 'teaching_examples_only', 'results': results}, indent=2) + '\n')
print('Entropy, normalization, local/final error and scale arithmetic checks passed.')
