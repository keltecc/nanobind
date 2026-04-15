#pragma once

#include <nanobind/xtensor/xcontainer.h>
#include <xtensor/core/xnoalias.hpp>

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

    /// Expressions are output-only: they are evaluated into an xarray
    /// and returned as a NumPy array. Cannot be received from Python.
    bool from_python(handle, uint8_t, cleanup_list*) noexcept = delete;

    template <typename T_>
    static handle from_cpp(T_ &&expr, rv_policy, cleanup_list *cl) noexcept {
        using NDArray = ndarray<Scalar, numpy>;
        using NDCaster = make_caster<NDArray>;

        size_t ndim = expr.dimension();
        std::vector<size_t> shape(ndim);
        for (size_t i = 0; i < ndim; ++i)
            shape[i] = expr.shape()[i];

        /// Allocate output buffer directly (single allocation).
        size_t size = expr.size();
        Scalar *data = new Scalar[size];
        object owner = capsule(data, [](void *p) noexcept {
            delete[] static_cast<Scalar*>(p);
        });

        /// Evaluate expression into the buffer via xtensor adaptor.
        /// Output buffer is freshly allocated, no aliasing with input data possible.
        auto out = xt::adapt<XTENSOR_DEFAULT_LAYOUT>(
            data, size, xt::no_ownership(), shape, XTENSOR_DEFAULT_LAYOUT);
        xt::noalias(out) = std::forward<T_>(expr);

        std::vector<int64_t> strides(ndim);
        int64_t stride = 1;
        for (size_t i = ndim; i-- > 0;) {
            strides[i] = stride;
            stride *= static_cast<int64_t>(shape[i]);
        }

        NDArray ndarr(data, ndim, shape.data(), owner, strides.data());
        return NDCaster::from_cpp(ndarr, rv_policy::reference, cl);
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
