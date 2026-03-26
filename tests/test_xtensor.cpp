#include <complex>
#include <nanobind/nanobind.h>
#include <nanobind/xtensor.h>

using complex_t = std::complex<double>;

namespace nb = nanobind;

static xt::xarray<double> static_array = {10.0, 20.0, 30.0};
static xt::xtensor<double, 2> static_tensor = {{10.0, 20.0}, {30.0, 40.0}};

double scalar_add(double a, double b) { return a + b; }

template <typename E>
inline auto array_multiply(E &&arr) {
    return arr * 1234.0;
}

NB_MODULE(test_xtensor_ext, m) {
    m.def("test_xarray_add", [](const xt::xarray<double>& a, const xt::xarray<double>& b) {
        return a + b;
    });

    m.def("test_xarray_funcs", [](const xt::xarray<double>& a, const xt::xarray<double>& b) {
        return xt::sin(a) + xt::cos(b);
    });

    m.def("test_xarray_scalar", [](const xt::xarray<double>& a, const double& s, const double& t) {
        return a * s + t;
    });

    m.def("test_xarray_return_by_value", []() {
        return xt::xarray<double>{1.0, 2.0, 3.0};
    });

    m.def("test_xarray_return_by_ref", []() -> xt::xarray<double>& {
        return static_array;
    });

    m.def("test_xarray_return_by_const_ref", []() -> const xt::xarray<double>& {
        return static_array;
    });

    m.def("test_xarray_accept_by_value", [](xt::xarray<double> a) {
        a(0) = 999.0;
        return a;
    });

    m.def("test_xarray_view", [](const nb::xarray_view<double>& a, const double& s, const double& t) {
        return xt::sin(a) * s + t;
    });

    m.def("test_xarray_view_zerocopy", [](const nb::xarray_view<double>& a) {
        return a;
    });

    m.def("test_xarray_view_mutate", [](nb::xarray_view<double>& a) {
        a(0) = 999.0;
    });

    m.def("test_xtensor_add", [](const xt::xtensor<double, 2>& a, const xt::xtensor<double, 2>& b) {
        return a + b;
    });

    m.def("test_xtensor_funcs", [](const xt::xtensor<double, 2>& a, const xt::xtensor<double, 2>& b) {
        return xt::sin(a) + xt::cos(b);
    });

    m.def("test_xtensor_scalar", [](const xt::xtensor<double, 2>& a, const double& s, const double& t) {
        return a * s + t;
    });

    m.def("test_xtensor_return_by_value", []() {
        return xt::xtensor<double, 2>{{1.0, 2.0}, {3.0, 4.0}};
    });

    m.def("test_xtensor_return_by_ref", []() -> xt::xtensor<double, 2>& {
        return static_tensor;
    });

    m.def("test_xtensor_return_by_const_ref", []() -> const xt::xtensor<double, 2>& {
        return static_tensor;
    });

    m.def("test_xtensor_accept_by_value", [](xt::xtensor<double, 2> a) {
        a(0, 0) = 999.0;
        return a;
    });

    m.def("test_xtensor_view", [](const nb::xtensor_view<double, 2>& a, const double& s, const double& t) {
        return xt::sin(a) * s + t;
    });

    m.def("test_xtensor_view_zerocopy", [](const nb::xtensor_view<double, 2>& a) {
        return a;
    });

    m.def("test_xtensor_view_mutate", [](nb::xtensor_view<double, 2>& a) {
        a(0, 0) = 999.0;
    });

    m.def("test_xarray_complex", [](const xt::xarray<complex_t>& a, const xt::xarray<complex_t>& b) {
        return a + b;
    });

    m.def("test_xtensor_complex", [](const xt::xtensor<complex_t, 2>& a) {
        return a * complex_t(0.0, 1.0);
    });

    m.def("test_vectorize", nb::xvectorize(scalar_add));

    m.def("test_vectorize_lambda", nb::xvectorize([](double x) {
        return std::sin(x);
    }));

    m.def("test_template_func", [](const xt::xarray<double>& a) {
        return array_multiply(a);
    });

    m.def("test_template_view_func", [](const nb::xarray_view<double>& a) {
        return array_multiply(a);
    });
}
