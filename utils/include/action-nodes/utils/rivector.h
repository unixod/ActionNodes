#ifndef ACTION_NODES_UTILS_RIVECTOR_H
#define ACTION_NODES_UTILS_RIVECTOR_H

#include <vector>

namespace anodes::utils {

template<typename T, typename Idx, typename Allocator = std::allocator<T>>
struct RIVector : std::vector<T, Allocator> {
private:
    using Base = std::vector<T>;
    using SizeType = typename Base::size_type;

public:
    using Base::Base;

    auto& operator[](Idx idx)
    {
        assert(ez::support::std23::to_underlying(idx) >= 0);
        assert(ez::utils::toUnsigned(ez::support::std23::to_underlying(idx)) < std::numeric_limits<SizeType>::max());

        return Base::operator[](ez::utils::toUnsigned(ez::support::std23::to_underlying(idx)));
    }

    auto& operator[](Idx idx) const
    {
        assert(ez::support::std23::to_underlying(idx) >= 0);
        assert(ez::utils::toUnsigned(ez::support::std23::to_underlying(idx)) < std::numeric_limits<SizeType>::max());

        return Base::operator[](ez::utils::toUnsigned(ez::support::std23::to_underlying(idx)));
    }

    Idx rsize() const noexcept
    {
        assert(Base::size() < std::numeric_limits<std::underlying_type_t<Idx>>::max());

        return static_cast<Idx>(Base::size());
    }
};

} // namespace anodes::utils

#endif // ACTION_NODES_UTILS_RIVECTOR_H
