import test_xtensor_ext as t
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_xarray_add():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert_array_equal(t.test_xarray_add(a, b), a + b)


def test_xarray_funcs():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    assert_array_almost_equal(t.test_xarray_funcs(a, b), np.sin(a) + np.cos(b))


def test_xarray_scalar():
    a = np.array([1.0, 2.0, 3.0])
    assert_array_equal(t.test_xarray_scalar(a, 2.0, 3.0), a * 2.0 + 3.0)


def test_xarray_return_by_value():
    result = t.test_xarray_return_by_value()
    assert_array_equal(result, [1.0, 2.0, 3.0])


def test_xarray_return_by_ref():
    result = t.test_xarray_return_by_ref()
    assert_array_equal(result, [10.0, 20.0, 30.0])


def test_xarray_return_by_const_ref():
    result = t.test_xarray_return_by_const_ref()
    assert_array_equal(result, [10.0, 20.0, 30.0])


def test_xarray_accept_by_value():
    a = np.array([1.0, 2.0, 3.0])
    result = t.test_xarray_accept_by_value(a)
    assert_array_equal(a, [1.0, 2.0, 3.0])
    assert_array_equal(result, [999.0, 2.0, 3.0])


def test_xarray_view():
    a = np.array([1.0, 2.0, 3.0])
    assert_array_almost_equal(
        t.test_xarray_view(a, 2.0, 3.0), np.sin(a) * 2.0 + 3.0,
    )


def test_xarray_view_zerocopy():
    a = np.array([1.0, 2.0, 3.0])
    b = t.test_xarray_view_zerocopy(a)
    a[0] = 1234.0
    assert_array_equal(a, b)
    b[0] = 5678.0
    assert_array_equal(a, b)


def test_xarray_view_mutate():
    a = np.array([1.0, 2.0, 3.0])
    t.test_xarray_view_mutate(a)
    assert_array_equal(a, [999.0, 2.0, 3.0])


def test_xtensor_add():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert_array_equal(t.test_xtensor_add(a, b), a + b)


def test_xtensor_funcs():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert_array_almost_equal(t.test_xtensor_funcs(a, b), np.sin(a) + np.cos(b))


def test_xtensor_scalar():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert_array_equal(t.test_xtensor_scalar(a, 2.0, 3.0), a * 2.0 + 3.0)


def test_xtensor_wrong_ndim():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    try:
        t.test_xtensor_add(a, b)
        assert False, "should raise TypeError"
    except TypeError:
        pass


def test_xtensor_return_by_value():
    result = t.test_xtensor_return_by_value()
    assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])


def test_xtensor_return_by_ref():
    result = t.test_xtensor_return_by_ref()
    assert_array_equal(result, [[10.0, 20.0], [30.0, 40.0]])


def test_xtensor_return_by_const_ref():
    result = t.test_xtensor_return_by_const_ref()
    assert_array_equal(result, [[10.0, 20.0], [30.0, 40.0]])


def test_xtensor_accept_by_value():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = t.test_xtensor_accept_by_value(a)
    assert_array_equal(a, [[1.0, 2.0], [3.0, 4.0]])
    assert_array_equal(result, [[999.0, 2.0], [3.0, 4.0]])


def test_xtensor_view():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert_array_almost_equal(
        t.test_xtensor_view(a, 2.0, 3.0), np.sin(a) * 2.0 + 3.0,
    )


def test_xtensor_view_zerocopy():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = t.test_xtensor_view_zerocopy(a)
    a[0, 0] = 1234.0
    assert_array_equal(a, b)
    b[0, 0] = 5678.0
    assert_array_equal(a, b)


def test_xtensor_view_mutate():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    t.test_xtensor_view_mutate(a)
    assert_array_equal(a, [[999.0, 2.0], [3.0, 4.0]])


def test_xtensor_view_wrong_ndim():
    a = np.array([1.0, 2.0, 3.0])
    try:
        t.test_xtensor_view(a, 2.0, 3.0)
        assert False, "should raise TypeError"
    except TypeError:
        pass


def test_xarray_complex():
    a = np.array([1 + 2j, 3 + 4j])
    b = np.array([5 + 6j, 7 + 8j])
    assert_array_equal(t.test_xarray_complex(a, b), a + b)


def test_xtensor_complex():
    a = np.array([[1 + 0j, 0 + 1j], [1 + 1j, 2 + 0j]])
    result = t.test_xtensor_complex(a)
    assert_array_equal(result, a * 1j)


def test_vectorize():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert_array_equal(t.test_vectorize(a, b), a + b)


def test_vectorize_lambda():
    a = np.array([1.0, 2.0, 3.0])
    assert_array_almost_equal(t.test_vectorize_lambda(a), np.sin(a))


def test_template_func():
    a = np.array([1.0, 2.0, 3.0])
    assert_array_equal(t.test_template_func(a), a * 1234.0)


def test_template_view_func():
    a = np.array([1.0, 2.0, 3.0])
    assert_array_equal(t.test_template_view_func(a), a * 1234.0)


def test_xarray_non_contiguous():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sliced = a[::2]
    result = t.test_xarray_accept_by_value(sliced)
    assert_array_equal(sliced, [1.0, 3.0, 5.0])
    assert_array_equal(result, [999.0, 3.0, 5.0])


def test_xarray_view_non_contiguous():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sliced = a[::2]
    t.test_xarray_view_mutate(sliced)
    assert sliced[0] == 999.0
    assert a[0] == 999.0
    assert a[1] == 2.0


def test_xtensor_non_contiguous():
    a = np.arange(16.0).reshape(4, 4)
    sliced = a[::2, ::2]
    result = t.test_xtensor_accept_by_value(sliced)
    assert_array_equal(sliced, [[0.0, 2.0], [8.0, 10.0]])
    assert_array_equal(result, [[999.0, 2.0], [8.0, 10.0]])


def test_xtensor_view_non_contiguous():
    a = np.arange(16.0).reshape(4, 4)
    sliced = a[::2, ::2]
    t.test_xtensor_view_mutate(sliced)
    assert sliced[0, 0] == 999.0
    assert a[0, 0] == 999.0
    assert a[0, 1] == 1.0


def test_xarray_column_major():
    a = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert not a.flags["C_CONTIGUOUS"]
    assert_array_equal(t.test_xarray_scalar(a, 2.0, 3.0), a * 2.0 + 3.0)


def test_xarray_view_column_major():
    a = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert not a.flags["C_CONTIGUOUS"]
    assert_array_almost_equal(
        t.test_xarray_view(a, 2.0, 3.0), np.sin(a) * 2.0 + 3.0,
    )


def test_xtensor_column_major():
    a = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert not a.flags["C_CONTIGUOUS"]
    assert_array_equal(t.test_xtensor_scalar(a, 2.0, 3.0), a * 2.0 + 3.0)


def test_xtensor_view_column_major():
    a = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert not a.flags["C_CONTIGUOUS"]
    assert_array_almost_equal(
        t.test_xtensor_view(a, 2.0, 3.0), np.sin(a) * 2.0 + 3.0,
    )


def test_xarray_view_type_mismatch():
    a = np.array([1, 2, 3], dtype=np.int32)
    try:
        t.test_xarray_view_zerocopy(a)
        assert False, "should raise TypeError"
    except TypeError:
        pass


def test_xtensor_view_type_mismatch():
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    try:
        t.test_xtensor_view_zerocopy(a)
        assert False, "should raise TypeError"
    except TypeError:
        pass
