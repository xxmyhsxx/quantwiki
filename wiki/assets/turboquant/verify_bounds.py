"""Reproduce the finite-dimensional MSE counterexample to TurboQuant v1.
No model, external data, or third-party libraries are used.
"""
import json
import math


def verify():
    radius = 2 * math.sqrt(2) / math.pi
    mse = 1 - radius**2
    printed_lower_bound = 0.25  # d=2, B=2, b=B/d=1
    # Averaging a shared uniform rotation makes every fixed input see
    # a uniform residual angle in [-pi/4, pi/4].
    n = 100_000
    integral = sum(
        1 + radius**2 - 2 * radius * math.cos(-math.pi/4 + (i+.5)*math.pi/(2*n))
        for i in range(n)
    ) / n
    assert abs(mse - integral) < 1e-10
    assert mse < printed_lower_bound
    assert (32*3 + 96*2)/128 == 2.25
    return {
        "sphere_dimension": 2, "total_index_bits": 2,
        "mse_exact": mse, "mse_midpoint_integration": integral,
        "v1_printed_mse_lower_bound": printed_lower_bound,
        "v1_mixed_channel_example_actual_bits": 2.25,
        "scope": "Teaching calculation; no model or kernel run",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
