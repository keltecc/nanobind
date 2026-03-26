#pragma once

#include <nanobind/xtensor/version.h>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xexpression.hpp>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename T>
struct xcaster_traits;

template <typename T, typename = void>
constexpr bool has_xcaster_traits_v = false;

template <typename T>
constexpr bool has_xcaster_traits_v<T,
    std::void_t<typename xcaster_traits<T>::scalar_type>> = true;

template <typename T>
constexpr bool is_xexpression_v =
    xt::is_xexpression<T>::value && !has_xcaster_traits_v<T>;

template <typename T>
using xarray_view = xt::xarray_adaptor<
    xt::xbuffer_adaptor<T*, xt::no_ownership>,
    xt::layout_type::dynamic,
    std::vector<std::size_t>>;

template <typename T, std::size_t N>
using xtensor_view = xt::xtensor_adaptor<
    xt::xbuffer_adaptor<T*, xt::no_ownership>,
    N,
    xt::layout_type::dynamic>;

template <typename T> struct xcaster_traits<xt::xarray<T>> {
    using scalar_type = T;
    using shape_type = std::vector<size_t>;
    using stride_type = std::vector<int64_t>;
    using view_type = xarray_view<T>;

    static bool check_ndim(size_t) { return true; }
    static shape_type make_shape(size_t nd) { return shape_type(nd); }
    static stride_type make_strides(size_t nd) { return stride_type(nd); }
};

template <typename T, std::size_t N> struct xcaster_traits<xt::xtensor<T, N>> {
    using scalar_type = T;
    using shape_type = std::array<size_t, N>;
    using stride_type = std::array<int64_t, N>;
    using view_type = xtensor_view<T, N>;

    static bool check_ndim(size_t nd) { return nd == N; }
    static shape_type make_shape(size_t) { return {}; }
    static stride_type make_strides(size_t) { return {}; }
};

template <typename T> struct xcaster_traits<xarray_view<T>> {
    using scalar_type = T;
};

template <typename T, std::size_t N> struct xcaster_traits<xtensor_view<T, N>> {
    using scalar_type = T;
};

NAMESPACE_END(detail)

template <typename T>
using xarray_view = detail::xarray_view<T>;

template <typename T, std::size_t N>
using xtensor_view = detail::xtensor_view<T, N>;

NAMESPACE_END(NB_NAMESPACE)
