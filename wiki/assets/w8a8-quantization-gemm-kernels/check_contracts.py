#!/usr/bin/env python3
"""CPU teaching references only; never imports or executes upstream GPU kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

results = []
def record(name, detail):
    results.append({"name": name, "status": "passed", "detail": detail})

def rounding():
    x = np.array([-.5, .5, -1.5, 1.5, -2.5, 2.5], dtype=np.float32)
    even = np.rint(x)
    away = np.copysign(np.floor(np.abs(x) + .5), x)
    assert even.tolist() == [0, 0, -2, 2, -2, 2]
    assert away.tolist() == [-1, 1, -2, 2, -3, 3]
    before = int(np.rint(np.float32(.5))) + 1
    after = int(np.rint(np.float32(.5) + 1))
    assert (before, after) == (1, 2)
    record("rounding_and_static_zero_point_order",
           {"ties_even": even.tolist(), "ties_away": away.tolist(),
            "round_then_add": before, "add_then_round": after})

def zero_scales():
    sg32 = np.float32(1e-10) / np.float32(127)
    sg16 = np.float16(sg32)
    vl32 = np.float32(0)
    assert sg32 > 0 and sg16 == 0 and vl32 == 0
    record("zero_row_scale_storage",
           {"sglang_fp32": float(sg32), "sglang_fp16": float(sg16),
            "vllm_dynamic_symmetric": float(vl32)})

def integer_correction():
    rng = np.random.default_rng(22)
    a = rng.integers(-128, 128, (7, 129), dtype=np.int64)
    b = rng.integers(-128, 128, (129, 11), dtype=np.int64)
    z = rng.integers(-20, 21, (7, 1), dtype=np.int64)
    c = a @ b
    adj = b.sum(axis=0)
    corrected = c - z * adj
    np.testing.assert_array_equal(corrected, (a - z) @ b)
    static_z = 3
    np.testing.assert_array_equal(c - static_z * adj, (a - static_z) @ b)
    # A small corrected answer alone cannot bound the raw INT32 intermediates.
    raw = 127 * 127 * 150000
    assert raw > np.iinfo(np.int32).max and raw - 127 * (127 * 150000) == 0
    record("integer_epilogue_identity_and_range",
           {"MKN": [7, 129, 11], "raw_overflow_example": raw, "corrected_example": 0})

def epilogue_association():
    rng = np.random.default_rng(23)
    c = rng.integers(-10000000, 10000000, 10000).astype(np.float32)
    sx = rng.uniform(.001, 1, 10000).astype(np.float32)
    sw = rng.uniform(.001, 1, 10000).astype(np.float32)
    scale_first = c * (sw * sx)
    column_first = (c * sw) * sx
    diff = np.abs(scale_first - column_first)
    assert np.any(diff > 0)
    i = int(np.flatnonzero(diff > 0)[0])
    record("fp32_scale_association",
           {"different_fp32_results": int(np.count_nonzero(diff)),
            "example": [float(c[i]), float(sw[i]), float(sx[i])],
            "outputs": [float(scale_first[i]), float(column_first[i])],
            "scope": "No compiler FMA or GPU instruction emulation"})

def qoq_output():
    coords = []
    for lane in range(32):
        for j in range(8):
            r = lane // 4 + 8 * ((j % 4) // 2)
            c = 2 * (lane % 4) + 8 * (j // 4) + j % 2
            coords.append((r, c))
    assert len(set(coords)) == 256
    assert set(coords) == {(r, c) for r in range(16) for c in range(16)}
    # Four adjacent 16-wide units in a 64-wide CTA: a row guard cannot mask N=63.
    tile = [(r, c + offset) for offset in range(0, 64, 16) for r, c in coords]
    outside = [(r, c) for r, c in tile if r < 9 and c >= 63]
    assert len(outside) == 9
    # Small-M g128: K=128 is group-aligned but shorter than two 128-wide prefetches.
    assert 128 % 128 == 0 and 128 < (3 - 1) * 128
    record("qoq_lane_coverage_and_boundary_counterexamples",
           {"unique_16x16_positions": len(set(coords)),
            "unmasked_columns_for_M9_N63": len(outside),
            "short_K_example": {"K": 128, "CTA_K": 128, "stages": 3}})

def marlin_repack():
    rng = np.random.default_rng(24)
    original = rng.integers(0, 16, (16, 64), dtype=np.uint32)
    perm = rng.permutation(16)
    inverse = [0, 4, 1, 5, 2, 6, 3, 7]  # inverse nibble mapping
    pack_index = [0, 2, 4, 6, 1, 3, 5, 7]
    for mapping in [np.arange(16), perm]:
        packed = np.zeros(128, dtype=np.uint32)
        ownership = []
        for warp in range(4):
            for lane in range(32):
                rr = (lane % 4) * 2 + np.array([0, 1, 8, 9])
                col = warp * 16 + lane // 4
                vals = np.r_[original[mapping[rr], col], original[mapping[rr], col + 8]]
                word = sum(int(vals[i]) << (4*j) for j, i in enumerate(pack_index))
                packed[lane*4 + warp] = word
                ownership.extend((int(k), int(n)) for n in [col, col+8] for k in rr)
        assert len(set(ownership)) == 1024
        # Decode by output logical coordinate, independently invert lane/warp ownership.
        restored = np.empty_like(original)
        for k in range(16):
            for n in range(64):
                warp, n16 = divmod(n, 16)
                lane = 4 * (n16 % 8) + (k % 8) // 2
                component = (k % 2) + 2 * (k // 8) + 4 * (n16 // 8)
                restored[k, n] = (int(packed[lane*4 + warp]) >> (4*inverse[component])) & 15
        np.testing.assert_array_equal(restored, original[mapping])
    record("marlin_16x64_repack", {"positions": 1024, "permuted_and_unpermuted": True})

def marlin_u4_half_magic():
    def h(bits):
        return np.array([bits], dtype=np.uint16).view(np.float16)[0]
    out = []
    for q in range(16):
        low = np.float16(h(0x6400 | q) - h(0x6408))
        # FMA has one final half rounding, so emulate with exact-enough float64.
        high = np.float16(float(h(0x6400 | (q << 4))) * float(h(0x2c00)) + float(h(0xd480)))
        assert low == q - 8 and high == q - 8
        out.append(float(low))
    record("marlin_u4b8_half_bit_constants", {"decoded": out})

def act_order():
    rng = np.random.default_rng(25)
    a = rng.integers(-8, 8, (3, 16), dtype=np.int64)
    w = rng.integers(-8, 8, (16, 5), dtype=np.int64)
    p = rng.permutation(16)
    np.testing.assert_array_equal(a @ w, a[:, p] @ w[p])
    assert np.any(a @ w != a @ w[p])
    bad = np.array([0, 1, 0])
    assert bad[0] == bad[-1] and not np.all(bad == bad[0])
    record("act_order_pairing_and_sorted_group_precondition",
           {"paired_permutation_equal": True, "weight_only_equal": False,
            "unsorted_endpoint_counterexample": bad.tolist()})

def partial_precision():
    # Each partial is FP32, but the hand-off buffer can round it to FP16.
    partials = [np.float32(2048), np.float32(1), np.float32(-2048)]
    handoff = np.float16(partials[0])
    for p in partials[1:-1]:
        handoff = np.float16(np.float32(handoff) + p)
    low = np.float16(np.float32(handoff) + partials[-1])
    full = np.float16(sum(partials, np.float32(0)))
    assert low == 0 and full == 1
    with np.errstate(over="ignore"):
        cast_first = np.float16(np.float16(70000) * np.float16(.001))
    scale_first = np.float16(np.float32(70000) * np.float32(np.float16(.001)))
    assert np.isinf(cast_first) and np.isfinite(scale_first)
    record("marlin_partial_storage_and_scale_order",
           {"fp16_handoff": float(low), "fp32_handoff": float(full),
            "cast_before_scale": "inf", "scale_before_cast": float(scale_first)})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for check in [rounding, zero_scales, integer_correction, epilogue_association,
                  qoq_output, marlin_repack, marlin_u4_half_magic, act_order, partial_precision]:
        check()
    report = {"date": "2026-09-22", "evidence": "CPU NumPy teaching reference; no upstream/GPU execution",
              "numpy": np.__version__, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "passed": len(results), "checks": results}
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(content)
    print(content, end="")

if __name__ == "__main__":
    main()
