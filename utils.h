#ifndef UTILS_H
#define UTILS_H

#include <variant>
#include <vector>
#include <limits>
#include <cassert>

namespace calc::utils {

template<typename T>
constexpr auto toUnderlying(T value) noexcept
{
    if constexpr (std::is_enum_v<T>) {
        return static_cast<std::underlying_type_t<T>>(value);
    }
    else {
        return value;
    }
}

template<typename T>
constexpr auto makeUnsigned(T value) noexcept
{
    assert(value >= 0);
    static_assert(std::is_integral<T>::value || std::is_enum<T>::value, "The value must be of either integral of enumeration type.");
    static_assert(!std::is_same<T, bool>::value, "The value can't be boolean.");
    return static_cast<std::make_unsigned_t<T>>(value);
}

template<typename T>
constexpr auto makeSigned(T value) noexcept
{
    static_assert(std::is_integral<T>::value || std::is_enum<T>::value, "The value must be of either integral of enumeration type.");
    static_assert(!std::is_same<T, bool>::value, "The value can't be boolean.");
    return static_cast<std::make_signed_t<T>>(value);
}

template<typename T>
class Badge {
    constexpr Badge() {};   // replace {} with = default since C++20.
    friend T;
};

template<typename T, typename Idx, typename Allocator = std::allocator<T>>
struct RIVector : std::vector<T, Allocator> {
private:
    using Base = std::vector<T>;
    using SizeType = typename Base::size_type;

//    class S {
//    public:
//        S(SizeType s, Badge<RIVector>)
//            : s_{s}
//        {}

//        operator SizeType() const noexcept
//        {
//            return s_;
//        }

//        operator Idx() const noexcept
//        {
//            assert(s_ < std::numeric_limits<std::underlying_type_t<Idx>>::max());

//            return static_cast<Idx>(s_);
//        }

//    private:
//        SizeType s_;
//    };

public:
    using Base::Base;

    auto& operator[](Idx idx)
    {
        assert(toUnderlying(idx) >= 0);
        assert(makeUnsigned(toUnderlying(idx)) < std::numeric_limits<SizeType>::max());

        return Base::operator[](makeUnsigned(toUnderlying(idx)));
    }

    auto& operator[](Idx idx) const
    {
        assert(toUnderlying(idx) >= 0);
        assert(makeUnsigned(toUnderlying(idx)) < std::numeric_limits<SizeType>::max());

        return Base::operator[](makeUnsigned(toUnderlying(idx)));
    }

    Idx rsize() const noexcept
    {
        assert(Base::size() < std::numeric_limits<std::underlying_type_t<Idx>>::max());

        return static_cast<Idx>(Base::size());
    }
};

template<typename T>
struct SparseSet {
    static_assert(std::is_integral_v<T> || std::is_enum_v<T>);

public:
    SparseSet() = default;

    SparseSet(std::size_t size)
        : sparseSet_(size)
    {}

    void set(T v)
    {
        if (contains(v)) {
            return;
        }

        if (v >= sparseSet_.rsize()) {
            sparseSet_.resize(toUnderlying(v)+1);
        }

        sparseSet_[v] = top_;

        if (top_ == denseSet_.size()) {
            denseSet_.push_back(v);
        }
        else {
            assert(top_ < denseSet_.size());
            denseSet_[top_] = v;
        }

        ++top_;
    }

    bool contains(T v)
    {
        assert(top_ <= denseSet_.size());

        if (v >= sparseSet_.rsize()) {
            return false;
        }

        auto denseId = sparseSet_[v];
        return denseId < top_ && denseSet_[denseId] == v;
    }

    void clear()
    {
        top_ = 0;
    }

private:
    std::size_t top_ = 0;
    std::vector<T> denseSet_;
    RIVector<std::size_t, T> sparseSet_;
};

template<typename... Callable>
struct Overloaded : Callable... {
    using Callable::operator()...;
};

template<typename... Callable>
Overloaded(Callable&&...) -> Overloaded<Callable...>;

template<typename Visitor, typename... Callable>
decltype(auto) match(Visitor&& v, Callable&&... func)
{
    return std::visit(Overloaded{std::forward<Callable>(func)...}, std::forward<Visitor>(v));
}

} // namespace calc::utils

#endif // UTILS_H
