"""CPU teaching checks for the shared quantization knowledge, not a model/kernel test.

Run with Python 3 and NumPy: python3 check_mechanisms.py
No downloads, GPU, input files or persistent outputs.
"""
import itertools
import math
import numpy as np

rng = np.random.default_rng(20260929)
passed = []


def close(actual, expected, name, **kwargs):
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10, **kwargs)
    passed.append(name)


def softmax(z):
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


q, k, v = (rng.normal(size=s) for s in [(5, 3), (5, 3), (5, 2)])
ek, ev = rng.normal(size=k.shape) / 10, rng.normal(size=v.shape) / 10
z = q @ k.T / np.sqrt(3)
dz = q @ ek.T / np.sqrt(3)
close(np.sum(dz**2), np.trace(ek @ (q.T @ q) @ ek.T) / 3, 'query-weighted logit identity')
masked = np.tril(np.ones((5, 5), dtype=bool))
causal = sum(ek[j] @ (q[j:].T @ q[j:]) @ ek[j] for j in range(5)) / 3
close(np.sum(dz[masked]**2), causal, 'causal per-key query metric')
assert np.sum(dz**2) > causal

p = softmax(z)
ph = softmax(z + dz)
close(softmax(z + 100), p, 'softmax common shift')
x = z[0]
jac = np.diag(p[0]) - np.outer(p[0], p[0])
eps = 1e-5
numerical = np.column_stack([(softmax(x + eps*e)-softmax(x-eps*e))/(2*eps) for e in np.eye(5)])
close(numerical, jac, 'softmax Jacobian finite differences')
assert np.linalg.norm(jac, 2) <= .5 + 1e-12
assert np.linalg.norm(ph-p) <= .5*np.linalg.norm(dz)
eta = np.max(np.abs(dz), axis=1, keepdims=True)
assert np.all(ph/p <= np.exp(2*eta)) and np.all(ph/p >= np.exp(-2*eta))
passed.append('softmax absolute and multiplicative bounds on fixed example')
close((softmax([0., 0.])[0]+softmax([2*np.log(3), 0.])[0])/2, .7, 'unbiased logit, biased probability')
assert softmax([np.log(3), 0.])[0] > .7
close(np.array([.5, .5]) @ np.array([2., -2.]), 0., 'Value error cancellation')
close(np.array([.5, .5]) @ np.array([2., 2.]), 2., 'Value error reinforcement')
close(ph @ (v+ev)-p @ v, (ph-p) @ v + p @ ev + (ph-p) @ ev, 'joint K/V error identity')

samples = rng.normal(size=(13, 3)) + np.array([10., 2., -1.])
mean = samples.mean(axis=0)
centered = samples-mean
close(samples.T@samples/13, centered.T@centered/13+np.outer(mean, mean), 'second moment versus covariance')
m = q.T @ q
lam, u = np.linalg.eigh(m)
r = u @ np.diag(np.sqrt(lam))
close(np.sum((ek@r)**2), np.trace(ek@m@ek.T), 'metric Euclideanization')
close((q@np.linalg.inv(r).T) @ (k@r).T, q@k.T, 'paired inverse query transform')
sigma = np.diag([4., 1.])
w = np.diag([.5, 1.])
close(w.T@sigma@w, np.eye(2), 'whitening uses inverse square root')

# Integrate the two polynomial errors exactly, using uniform density 1/2.
def uniform_error(c):
    return ((1-c)**3 - (-c)**3)/3
close(uniform_error(.5), 1/12, 'one-bit Lloyd centroid distortion')
close(uniform_error(1.), 1/3, 'one-bit endpoint distortion')
close((2*(.5**(1/3)))**3 / (12*2**2), 1/12, 'uniform high-resolution constant')
a = np.array([16., 1.])
b_free = .5 + .5*np.log2(a/np.sqrt(a.prod()))
close(b_free, [1.5, -.5], 'unconstrained allocation can request negative bits')
alloc = np.linspace(0, 1, 10001)
cost = 16*2**(-2*alloc)+2**(-2*(1-alloc))
close(alloc[cost.argmin()], 1., 'nonnegative budget optimum on dense one-dimensional grid')

# All 256 adjacent 2-bit quadruples; no float quantizer hides a packing defect.
for codes in itertools.product(range(4), repeat=4):
    byte = sum(c << (2*i) for i, c in enumerate(codes))
    assert tuple((byte >> (2*i)) & 3 for i in range(4)) == codes
assert sum(c << (2*i) for i, c in enumerate([0, 1, 2, 3])) == 0xe4
for code in range(16):
    assert (code & 3) + 4*((code >> 2) & 3) == code
passed.append('all 2-bit quadruples and all 4-bit split codes round-trip')
g = d = 128
boost = 32
k_bytes = (2*g*d+2*g*boost+8*d+32*d)//8
v_bytes = (2*g*d+32*g)//8
close([k_bytes, v_bytes, (k_bytes+v_bytes)*8/(2*g*d)], [5760, 4608, 2.53125], 'complete page metadata accounting')
assert round(.5) == 0 and math.floor(.5+.5) == 1
passed.append('ties-to-even versus nonnegative half-up')

# Simplified read-then-pack cache: S=2, G=4, R=2; not an actual supported GPU shape.
sink, kp, vp, kb, vb, recent = [], [], [], [], [], []
trace = {}
for t in range(1, 21):
    if len(sink) < 2:
        sink.append(t)
    else:
        kb.append(t)
        recent.append(t)
        if len(recent) > 2:
            vb.append(recent.pop(0))
    keys = sink + sum(kp, []) + kb
    values = sink + sum(vp, []) + vb + recent
    assert keys == values == list(range(1, t+1))
    trace[t] = (kb.copy(), vb.copy(), recent.copy())
    if len(kb) == 4:
        kp.append(kb.copy()); kb.clear()
    if len(vb) == 4:
        vp.append(vb.copy()); vb.clear()
    assert t == len(sink)+4*len(kp)+len(kb)
    assert t == len(sink)+4*len(vp)+len(vb)+len(recent)
assert trace[6] == ([3,4,5,6], [3,4], [5,6])
assert trace[8] == ([7,8], [3,4,5,6], [7,8])
passed.append('twenty-step cache identity/order and read-before-pack boundaries')

print(f'{len(passed)} CPU teaching checks passed; no model or GPU execution.')
for name in passed:
    print(f'  - {name}')
