#pragma once

#include <nanobind/xtensor/xcontainer.h>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename T>
struct type_caster<T, enable_if_t<is_xexpression_v<T> &&
                                  is_ndarray_scalar_v<typename T::value_type>>> {
    using Scalar = typename T::value_type;
    using XArray = xt::xarray<Scalar>;
    using XArrayCaster = make_caster<XArray>;

    static constexpr auto Name = XArrayCaster::Name;
    template <typename T_> using Cast = T;
    template <typename T_> static constexpr bool can_cast() { return true; }

    bool from_python(handle, uint8_t, cleanup_list*) noexcept = delete;

    template <typename T_>
    static handle from_cpp(T_ &&expr, rv_policy, cleanup_list *cl) noexcept {
        return XArrayCaster::from_cpp(XArray(std::forward<T_>(expr)), rv_policy::move, cl);
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
