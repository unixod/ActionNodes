#ifndef ACTION_NODES_UTILS_SPARSE_SET_H
#define ACTION_NODES_UTILS_SPARSE_SET_H

#include "action-nodes/utils/rivector.h"

namespace anodes::utils {

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
            sparseSet_.resize(ez::support::std23::to_underlying(v)+1);
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

} // namespace anodes::utils

#endif // ACTION_NODES_UTILS_SPARSE_SET_H
