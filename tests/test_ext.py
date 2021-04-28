"""Test methods for faster fishers"""
import numpy as np
from faster_fishers import exact_py


def test_exact():
    """Test the exact method with numpy arrays as input"""
    a_values = np.array([1, 3], dtype=np.uint32)
    b_values = np.array([2, 5], dtype=np.uint32)
    c_values = np.array([1, 4], dtype=np.uint32)
    d_values = np.array([5, 50], dtype=np.uint32)
    lesses, greaters, two_tails = exact_py(a_values, b_values, c_values, d_values)
    np.testing.assert_array_almost_equal(lesses, np.array([0.916666, 0.996303]))
    np.testing.assert_array_almost_equal(greaters, np.array([0.583333, 0.039707]))
    np.testing.assert_array_almost_equal(two_tails, np.array([1.0, 0.039707]))
