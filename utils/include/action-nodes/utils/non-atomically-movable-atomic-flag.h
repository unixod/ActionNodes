#ifndef ACTION_NODES_UTILS_NAMAF_H
#define ACTION_NODES_UTILS_NAMAF_H

#include <atomic>

namespace anodes::utils {

struct NonAtomicallyMovableAtomicFlag  {
    NonAtomicallyMovableAtomicFlag() = default;

    NonAtomicallyMovableAtomicFlag(NonAtomicallyMovableAtomicFlag&& oth) noexcept
    {
        if (oth.flag.test_and_set()) {
            flag.test_and_set();
        }
    }

    NonAtomicallyMovableAtomicFlag& operator = (NonAtomicallyMovableAtomicFlag&& oth) noexcept
    {
        if (oth.flag.test_and_set()) {
            flag.test_and_set();
        }
        else {
            flag.clear();
        }
        return *this;
    }

    NonAtomicallyMovableAtomicFlag(const NonAtomicallyMovableAtomicFlag& oth) = delete;
    NonAtomicallyMovableAtomicFlag& operator = (const NonAtomicallyMovableAtomicFlag&) = delete;
    ~NonAtomicallyMovableAtomicFlag() = default;

    bool testAndSet() noexcept
    {
        return flag.test_and_set();
    }

    void clear() noexcept
    {
        return flag.clear();
    }

private:
    std::atomic_flag flag;          // NOTE: ATOMIC_FLAG_INIT isn't necessary to use anymore since C++20.
};

} // namespace anodes::utils

#endif // ACTION_NODES_UTILS_NAMAF_H
