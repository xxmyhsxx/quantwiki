"""CPU teaching algebra; no model calibration or INT8 kernel execution."""
import argparse
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(20260922)
x = rng.normal(size=(5, 16))
x = (x-x.mean(axis=-1, keepdims=True))/np.sqrt(x.var(axis=-1, keepdims=True)+1e-5)
gamma, beta = rng.normal(size=(2,16))
weights = [rng.normal(size=(n,16)) for n in (7,9,11)]
biases = [rng.normal(size=w.shape[0]) for w in weights]
norm = x*gamma+beta
a = np.abs(norm).max(axis=0)
b = np.max([np.abs(w).max(axis=0) for w in weights], axis=0).clip(1e-5)
s = (a**.5/b**.5).clip(1e-5)
errors, omitted_bias_errors = [], []
for w, bias in zip(weights,biases):
    target = norm@w.T+bias
    fused = (x*(gamma/s)+beta/s)@(w*s).T+bias
    broken = (x*(gamma/s)+beta)@(w*s).T+bias
    errors.append(float(np.max(np.abs(target-fused))))
    omitted_bias_errors.append(float(np.max(np.abs(target-broken))))
assert max(errors)<1e-12 and max(omitted_bias_errors)>1e-3
qx = rng.integers(-127,128,(4,32),dtype=np.int32)
qw = rng.integers(-127,128,(6,32),dtype=np.int32)
qb = rng.integers(-127,128,6,dtype=np.int32)
sx, sw, sb, sy = .031, .007, .004, .023
acc = qx@qw.T
grid = (sx*sw/sy)*acc+(sb/sy)*qb
float_reference = ((qx*sx)@(qw*sw).T+qb*sb)/sy
assert np.allclose(grid,float_reference,atol=1e-10)
# Outer row/column scales are valid only when independent of reduction K.
sx_row = rng.uniform(.01,.05,(4,1)); sw_col = rng.uniform(.01,.03,(1,6))
outer = acc*sx_row*sw_col
assert np.allclose(outer,(qx*sx_row)@(qw*sw_col.T).T,atol=1e-12)
# Dynamic zero-row protection and static clipping differ in deployment.
row = np.array([0., 1., 2., 100.])
static = np.clip(np.rint(row/0.1),-128,127)*0.1
dynamic_scale = max(np.max(np.abs(row)),1e-10)/127
dynamic = np.rint(row/dynamic_scale)*dynamic_scale
assert np.isclose(static[-1],12.7) and np.isclose(dynamic[-1],100)
assert max(np.max(np.abs(np.zeros(8))),1e-10)/127 > 0
result = {
    "evidence": "NumPy float64 teaching; no CUDA rounding equivalence asserted",
    "seed": 20260922, "checks_passed": 4,
    "checks": ["shared-consumer smoothing and LayerNorm bias",
               "INT8 output-grid epilogue", "per-token/per-channel outer scales",
               "static clipping versus dynamic range"],
    "smoothing_max_abs_error": max(errors),
    "omitted_bias_max_abs_error": max(omitted_bias_errors),
    "epilogue_max_abs_error": float(np.max(np.abs(grid-float_reference)))
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output",type=Path)
    args=parser.parse_args(); text=json.dumps(result,indent=2,ensure_ascii=False)+"\n"
    if args.output: args.output.write_text(text)
    print(text,end="")
