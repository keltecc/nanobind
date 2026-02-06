#pragma once

#include <nanobind/xtensor/version.h>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xexpression.hpp>
#include <type_traits>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

// xarray detection
template <typename T>
struct is_xarray : std::false_type {};

template <typename T>
struct is_xarray<xt::xarray<T>> : std::true_type {};

template <typename T>
constexpr bool is_xarray_v = is_xarray<T>::value;

// xarray_adaptor detection
template <typename T>
struct is_xarray_adaptor : std::false_type {};

template <typename EC, xt::layout_type L, typename SC, typename Tag>
struct is_xarray_adaptor<xt::xarray_adaptor<EC, L, SC, Tag>> : std::true_type {};

template <typename EC, xt::layout_type L, typename SC>
struct is_xarray_adaptor<xt::xarray_adaptor<EC, L, SC>> : std::true_type {};

template <typename T>
constexpr bool is_xarray_adaptor_v = is_xarray_adaptor<T>::value;

// xexpression detection
template <typename T>
constexpr bool is_xexpression_v =
    xt::is_xexpression<T>::value &&
    !is_xarray_v<T> &&
    !is_xarray_adaptor_v<T>;

NAMESPACE_END(detail)

using detail::is_xarray_v;
using detail::is_xarray_adaptor_v;
using detail::is_xexpression_v;

NAMESPACE_END(NB_NAMESPACE)
