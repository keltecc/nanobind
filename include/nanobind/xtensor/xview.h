#pragma once

#include <optional>
#include <nanobind/ndarray.h>
#include <nanobind/xtensor/traits.h>
#include <xtensor/containers/xadapt.hpp>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename Container>
struct xview_caster {
    using Traits = xcaster_traits<Container>;
    using Scalar = typename Traits::scalar_type;
    using View = typename Traits::view_type;
    using NDArray = ndarray<Scalar, numpy>;
    using Caster = make_caster<NDArray>;

    static constexpr auto Name = Caster::Name;
    template <typename T_> using Cast = movable_cast_t<T_>;
    template <typename T_> static constexpr bool can_cast() { return true; }

    Caster caster;
    typename Traits::shape_type shape_;
    typename Traits::stride_type strides_;
    std::optional<View> view_;

    bool from_python(handle src, uint8_t flags, cleanup_list *cl) noexcept {
        if (!caster.from_python(src, flags & ~(uint8_t)cast_flags::convert, cl))
            return false;

        NDArray &arr = caster.value;
        size_t ndim = arr.ndim();
        if (!Traits::check_ndim(ndim))
            return false;

        shape_ = Traits::make_shape(ndim);
        strides_ = Traits::make_strides(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            shape_[i] = arr.shape(i);
            strides_[i] = static_cast<int64_t>(arr.stride(i));
        }

        view_.emplace(xt::adapt(static_cast<Scalar*>(arr.data()), arr.size(),
                                xt::no_ownership(), shape_, strides_));
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

template <typename T>
struct type_caster<xarray_view<T>, enable_if_t<is_ndarray_scalar_v<T>>>
    : xview_caster<xt::xarray<T>> {};

template <typename T, std::size_t N>
struct type_caster<xtensor_view<T, N>, enable_if_t<is_ndarray_scalar_v<T>>>
    : xview_caster<xt::xtensor<T, N>> {};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
