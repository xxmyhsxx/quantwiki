"""Exact Gaussian teaching counterexample; not a model or kernel experiment."""
import json
import math
from pathlib import Path

epsilon, delta = 0.1, 0.05
m = math.ceil(4 / 3 * (1 + epsilon) / epsilon**2 * math.log(2 / delta))
# q=(1,0), k=(0,1), S rows iid standard normal.
# s_1 sign(s_2) is standard normal; the m-row estimate is N(0, pi/(2m)).
variance = math.pi / (2 * m)
failure = math.erfc(epsilon / math.sqrt(2 * variance))
assert m == 542 and failure > delta
assert math.exp(2 * 0.5) > 1 + 3 * 0.5
report = {
    'scope': 'Analytic special case of QJL v2 Lemma 3.5; fixed orthogonal unit vectors and iid Gaussian rows.',
    'epsilon': epsilon, 'claimed_delta': delta, 'paper_required_m': m,
    'exact_variance': variance, 'exact_failure_probability': failure,
    'conclusion': 'Printed sample-size constant does not meet claimed failure probability in this case.',
    'not_tested': ['orthogonalized practical transform', 'model quality', 'GPU kernels'],
}
Path(__file__).with_name('estimator-check.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
