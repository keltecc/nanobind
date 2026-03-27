import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import test_xtensor_ext as t


def test_add():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert_array_equal(t.test_add(a, b), a + b)


def test_funcs():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    result = t.test_funcs(a, b)
    expected = np.sin(a) + np.cos(b)
    assert_array_almost_equal(result, expected, decimal=10)


def test_scalar():
    a = np.array([1.0, 2.0, 3.0])
    assert_array_equal(t.test_scalar(a, 2.0, 3.0), a * 2.0 + 3.0)

def test_view():
    a = np.array([1.0, 2.0, 3.0])
    assert_array_almost_equal(t.test_view(a, 2.0, 3.0), np.sin(a) * 2.0 + 3.0, decimal=10)
    assert_array_equal(a, np.array([1.0, 2.0, 3.0]))

def test_view_zerocopy():
    a = np.array([1.0, 2.0, 3.0])
    b = t.test_view_zerocopy(a)
    a[0] = 1234.0
    assert_array_equal(a, b)
    b[0] = 5678.0
    assert_array_equal(a, b)
