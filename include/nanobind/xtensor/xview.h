#pragma once

#include <optional>
#include <nanobind/ndarray.h>
#include <nanobind/xtensor/traits.h>
#include <xtensor/containers/xadapt.hpp>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename View>
struct xview_caster {
    using Traits = xcaster_traits<View>;
    using Scalar = typename Traits::scalar_type;
    using NDArray = ndarray<Scalar, numpy>;
    using Caster = make_caster<NDArray>;

    /// Compile-time layout from the View type. Determines iteration strategy:
    /// known layout (row_major / column_major) enables fast flat-pointer iteration,
    /// dynamic falls back to slower stride-based stepping.
    static constexpr xt::layout_type ViewLayout = View::static_layout;

    static constexpr auto Name = Caster::Name;
    template <typename T_> using Cast = movable_cast_t<T_>;
    template <typename T_> static constexpr bool can_cast() { return true; }

    Caster caster;
    std::optional<View> view_;

    bool from_python(handle src, uint8_t flags, cleanup_list *cl) noexcept {
        /// Strip the convert flag: views wrap existing memory without copying,
        /// so implicit conversions (e.g. int->float) are not supported.
        if (!caster.from_python(src, flags & ~(uint8_t)cast_flags::convert, cl))
            return false;

        NDArray &arr = caster.value;
        size_t ndim = arr.ndim();
        if (!Traits::check_ndim(ndim))
            return false;

        auto shape = Traits::make_shape(ndim);
        for (size_t i = 0; i < ndim; ++i)
            shape[i] = arr.shape(i);

        xt::layout_type layout = detect_layout(arr);
        if constexpr (ViewLayout != xt::layout_type::dynamic) {
            /// Known layout: reject mismatched arrays, but allow 1D contiguous
            /// arrays through (row_major == column_major for 1D).
            /// Non-contiguous 1D (layout==dynamic) is still rejected.
            if (layout != ViewLayout && (ndim > 1 || layout == xt::layout_type::dynamic))
                return false;

            /// adapt<Layout>(..., shape, layout), no strides needed, xtensor
            /// computes them from shape. This enables flat-pointer iteration.
            view_.emplace(xt::adapt<ViewLayout>(
                static_cast<Scalar*>(arr.data()), arr.size(),
                xt::no_ownership(), std::move(shape), ViewLayout));
        } else {
            /// Dynamic layout: accepts any array. When data is contiguous,
            /// pass detected layout to enable xtensor's internal optimization.
            if (layout != xt::layout_type::dynamic) {
                view_.emplace(xt::adapt<xt::layout_type::dynamic>(
                    static_cast<Scalar*>(arr.data()), arr.size(),
                    xt::no_ownership(), std::move(shape), layout));
            } else {
                // Non-contiguous: must pass explicit strides.
                auto strides = Traits::make_strides(ndim);
                for (size_t i = 0; i < ndim; ++i)
                    strides[i] = static_cast<int64_t>(arr.stride(i));

                view_.emplace(xt::adapt(
                    static_cast<Scalar*>(arr.data()), arr.size(),
                    xt::no_ownership(), std::move(shape), std::move(strides)));
            }
        }
        return true;
    }

    explicit operator View*()  { return &*view_; }
    explicit operator View&()  { return *view_; }
    explicit operator View&&() { return (View&&) *view_; }

    template <typename T_>
    static handle from_cpp(T_ &&arr, rv_policy policy, cleanup_list *cl) noexcept {
        size_t ndim = arr.dimension();

        auto shape = Traits::make_shape(ndim);
        auto strides = Traits::make_strides(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            shape[i] = arr.shape()[i];
            strides[i] = static_cast<int64_t>(arr.strides()[i]);
        }

        object owner;
        if (policy == rv_policy::reference_internal && cl->self()) {
            owner = borrow(cl->self());
            policy = rv_policy::reference;
        }

        NDArray ndarr((void *) arr.data(), ndim, shape.data(), owner, strides.data());
        if (policy == rv_policy::automatic || policy == rv_policy::automatic_reference)
            policy = rv_policy::reference;

        return Caster::from_cpp(ndarr, policy, cl);
    }
};

template <typename T, xt::layout_type L>
struct type_caster<xarray_view<T, L>, enable_if_t<is_ndarray_scalar_v<T>>>
    : xview_caster<xarray_view<T, L>> {};

template <typename T, std::size_t N, xt::layout_type L>
struct type_caster<xtensor_view<T, N, L>, enable_if_t<is_ndarray_scalar_v<T>>>
    : xview_caster<xtensor_view<T, N, L>> {};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
