"""Test methods for faster fishers"""
import numpy as np
import pytest
from faster_fishers import exact, exact_with_odds_ratios


def test_exact():
    """Test the exact method with numpy arrays as input"""
    a_values = np.array([1, 3], dtype=np.uint64)
    b_values = np.array([2, 5], dtype=np.uint64)
    c_values = np.array([1, 4], dtype=np.uint64)
    d_values = np.array([5, 50], dtype=np.uint64)

    lesses = exact(a_values, b_values, c_values, d_values, "less")
    np.testing.assert_array_almost_equal(lesses, np.array([0.916666, 0.996303]))

    greaters = exact(a_values, b_values, c_values, d_values, "greater")
    np.testing.assert_array_almost_equal(greaters, np.array([0.583333, 0.039707]))

    two_tails = exact(a_values, b_values, c_values, d_values, "two-sided")
    np.testing.assert_array_almost_equal(two_tails, np.array([1.0, 0.039707]))

    odds_greaters = exact_with_odds_ratios(a_values, b_values, c_values, d_values, "greater")
    np.testing.assert_array_almost_equal(odds_greaters[0], np.array([2.5, 7.5]))
    np.testing.assert_array_almost_equal(odds_greaters[1], greaters)


@pytest.mark.benchmark
def test_benchmark_ppi(benchmark):
    """Benchmark fisher tests."""
    values = np.random.rand(4, 10000).astype(dtype=np.uint64)
    benchmark(exact, *values, "less")
