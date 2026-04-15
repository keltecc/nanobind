#pragma once

#include <nanobind/ndarray.h>
#include <nanobind/xtensor/traits.h>
#include <xtensor/containers/xadapt.hpp>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename Container>
struct xcontainer_caster {
    using Traits = xcaster_traits<Container>;
    using Scalar = typename Traits::scalar_type;
    using NDArray = ndarray<Scalar, numpy>;
    using Caster = make_caster<NDArray>;

    NB_TYPE_CASTER(Container, Caster::Name)

    Caster caster;

    bool from_python(handle src, uint8_t flags, cleanup_list *cl) noexcept {
        if (!caster.from_python(src, flags, cl))
            return false;

        NDArray &arr = caster.value;
        size_t ndim = arr.ndim();
        if (!Traits::check_ndim(ndim))
            return false;

        auto shape = Traits::make_shape(ndim);
        for (size_t i = 0; i < ndim; ++i)
            shape[i] = arr.shape(i);

        /// Build a temporary adaptor over the input buffer, then assign into
        /// an owning container which copies and re-layouts the data.
        /// The adaptor uses adapt<dynamic> because the input layout may differ
        /// from the container's layout, this is fine since the adaptor is
        /// never iterated by user code, only by xtensor's copy/assign logic.
        xt::layout_type layout = detect_layout(arr);
        if (layout != xt::layout_type::dynamic) {
            value = xt::adapt<xt::layout_type::dynamic>(
                arr.data(), arr.size(), xt::no_ownership(), shape, layout);
        } else {
            auto strides = Traits::make_strides(ndim);
            for (size_t i = 0; i < ndim; ++i)
                strides[i] = static_cast<int64_t>(arr.stride(i));

            value = xt::adapt(arr.data(), arr.size(), xt::no_ownership(), shape, strides);
        }

        return true;
    }

    template <typename T_>
    static handle from_cpp(T_ &&v, rv_policy policy, cleanup_list *cl) noexcept {
        policy = infer_policy<T_>(policy);
        if constexpr (std::is_pointer_v<T_>)
            return from_cpp_internal((const Value &) *v, policy, cl);
        else
            return from_cpp_internal((const Value &) v, policy, cl);
    }

private:
    static handle from_cpp_internal(const Value &arr, rv_policy policy, cleanup_list *cl) noexcept {
        size_t ndim = arr.dimension();

        auto shape = Traits::make_shape(ndim);
        auto strides = Traits::make_strides(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            shape[i] = arr.shape()[i];
            strides[i] = static_cast<int64_t>(arr.strides()[i]);
        }

        void *ptr = (void *) arr.data();
        object owner;

        if (policy == rv_policy::move) {
            Value *temp = new Value((Value&&) arr);
            owner = capsule(temp, [](void *p) noexcept { delete (Value*)p; });
            ptr = temp->data();
            policy = rv_policy::reference;
        } else if (policy == rv_policy::reference_internal && cl->self()) {
            owner = borrow(cl->self());
            policy = rv_policy::reference;
        }

        NDArray ndarr(ptr, ndim, shape.data(), owner, strides.data());
        return Caster::from_cpp(ndarr, policy, cl);
    }
};

template <typename EC, xt::layout_type L, typename SC, typename Tag>
struct type_caster<xt::xarray_container<EC, L, SC, Tag>,
    enable_if_t<is_ndarray_scalar_v<typename EC::value_type>>>
    : xcontainer_caster<xt::xarray_container<EC, L, SC, Tag>> {};

template <typename EC, std::size_t N, xt::layout_type L, typename Tag>
struct type_caster<xt::xtensor_container<EC, N, L, Tag>,
    enable_if_t<is_ndarray_scalar_v<typename EC::value_type>>>
    : xcontainer_caster<xt::xtensor_container<EC, N, L, Tag>> {};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
