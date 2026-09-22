"""CPU teaching checks, not execution of GPTQ CUDA or a model."""
import argparse
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(20260922)
q = rng.integers(0, 8, size=(256, 7), dtype=np.uint32)
words = np.zeros((24, 7), dtype=np.uint32)
# Reproduce the three-word packing contract; decode with an independent bitstream.
for base in range(0, 256, 32):
    a = q[base:base + 32]
    row = base // 32 * 3
    for j in range(10):
        words[row] |= a[j] << (3*j)
    words[row] |= a[10] << 30
    words[row+1] |= a[10] >> 2
    for j in range(11, 21):
        words[row+1] |= a[j] << (1+3*(j-11))
    words[row+1] |= a[21] << 31
    words[row+2] |= a[21] >> 1
    for j in range(22, 32):
        words[row+2] |= a[j] << (2+3*(j-22))
decoded = np.empty_like(q)
for n in range(q.shape[1]):
    for tile in range(8):
        stream = sum(int(words[3*tile+j,n]) << (32*j) for j in range(3))
        for k in range(32):
            decoded[32*tile+k,n] = (stream >> (3*k)) & 7
assert np.array_equal(q, decoded)
s = rng.uniform(.01, 1, size=7)
z = rng.integers(0, 8, size=7)
x = rng.normal(size=256)
direct = x @ ((q.astype(float)-z)*s)
via_packed = (x @ decoded.astype(float))*s - x.sum()*(z*s)
assert np.allclose(direct, via_packed, atol=1e-11)
# Out-of-range codes corrupt the next logical field.
bad = 8 | (0 << 3)
assert (bad & 7) != 8 and ((bad >> 3) & 7) == 1
# Group sorting must change X, W and group lookup consistently.
g = rng.permutation(np.repeat(np.arange(4), 8))
codes = rng.integers(0, 16, (32, 5))
scales = rng.uniform(.01, 2, (4, 5))
w = (codes-8)*scales[g]
a = rng.normal(size=(3, 32))
perm = np.argsort(g)
reference = a @ w
consistent = a[:, perm] @ ((codes[perm]-8)*scales[g[perm]])
wrong = a @ ((codes[perm]-8)*scales[g[perm]])
assert np.allclose(reference, consistent, atol=1e-12)
assert np.max(np.abs(reference-wrong)) > 1
# add_batch normalizes by batch sequences, not flattened token count.
batches = [rng.normal(size=(2, 3, 5)), rng.normal(size=(1, 7, 5))]
h = np.zeros((5, 5)); samples = 0
for batch in batches:
    tmp = batch.shape[0]
    h *= samples/(samples+tmp)
    samples += tmp
    inp = batch.reshape(-1, 5).T * np.sqrt(2/samples)
    h += inp @ inp.T
all_tokens = np.concatenate([b.reshape(-1,5) for b in batches])
assert np.allclose(h, 2/samples * all_tokens.T @ all_tokens)
assert not np.allclose(h, 2/len(all_tokens) * all_tokens.T @ all_tokens)
result = {
    "evidence": "NumPy CPU teaching, no upstream module/CUDA/model executed",
    "seed": 20260922, "checks_passed": 4,
    "checks": ["3bit packing and dequantized dot", "invalid code corrupts neighbor",
               "group sorting coordinate consistency", "Hessian sequence normalization"],
    "packing_shape": list(words.shape),
    "dequant_dot_max_abs_error": float(np.max(np.abs(direct-via_packed))),
    "consistent_group_sort_max_abs_error": float(np.max(np.abs(reference-consistent))),
    "wrong_group_sort_max_abs_error": float(np.max(np.abs(reference-wrong)))
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); text = json.dumps(result, indent=2, ensure_ascii=False)+"\n"
    if args.output: args.output.write_text(text)
    print(text, end="")
