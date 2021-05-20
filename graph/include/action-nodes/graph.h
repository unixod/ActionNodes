#ifndef ACTION_NODES_GRAPH_H
#define ACTION_NODES_GRAPH_H

#include <cassert>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <atomic>
#include <numeric>
#include <type_traits>
#include <execution>
#include <thread>
#include <vector>
#include <string>
#include <map>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <iterator>
#include <any>
#include <ez/support/c++23-features.h>
#include <ez/utils/utils.h>

namespace anodes {

class ThreadPool {
public:
    ThreadPool(std::size_t numOfThreads = std::thread::hardware_concurrency())
        : busyWorkerThreads_{numOfThreads}
    {
        if (numOfThreads == 0) {
            throw std::runtime_error{"A number of threads must be greater than 0."};
        }

        threads_.reserve(numOfThreads);
        for (auto i = numOfThreads; i; --i) {
            threads_.emplace_back( [this]{ runTh_(); });
        }
    }

    ThreadPool(ThreadPool&&) = delete;
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator = (ThreadPool&&) = delete;
    ThreadPool& operator = (const ThreadPool&) = delete;

    ~ThreadPool()
    {
        std::unique_lock lck{mx_};
        stop_ = true;
        lck.unlock();

        cvThs_.notify_all();

        for (auto& th : threads_) {
            th.join();
        }
    }

    template<
        typename MapCallable,
        typename ReduceCallable,
        typename RaIt,
        typename InitValue
    >
    auto mapReduce(RaIt begin, RaIt end, InitValue&& value, MapCallable&& mapFunc, ReduceCallable&& reduceFunc)
    {
        using RaItCategory = typename std::iterator_traits<RaIt>::iterator_category;
        static_assert(std::is_same_v<RaItCategory, std::random_access_iterator_tag>);

        auto procFunc = [&mapFunc, &reduceFunc, begin, this]() -> std::any {
            auto c = --cnt_;

            if (c > 0) {
                cvThs_.notify_one();
            }

            if (c < 0) {
                return {};
            }

            auto r = mapFunc(*std::next(begin, c));

            for (; c >= 0; c = --cnt_) {
                r = reduceFunc(r, mapFunc(*std::next(begin, c)));
            }

            return r;
        };

        using MapResultType = decltype(mapFunc(std::declval<typename std::iterator_traits<RaIt>::reference>()));
        using ReduceResultType = decltype(reduceFunc(std::declval<MapResultType>(), std::declval<MapResultType>()));

        ReduceResultType result{std::forward<InitValue>(value)};

        auto submitFunc = [&reduceFunc, &result](std::any& r){
            assert(r.has_value());
            auto& v = std::any_cast<ReduceResultType&>(r);
            result = reduceFunc(result, v);
        };

        runMain_(procFunc, submitFunc, std::distance(begin, end));
        return result;
    }

private:
    template<typename ProcCallable, typename SubmitCallable, typename Cnt>
    void runMain_(ProcCallable& procFunc, SubmitCallable& submitFunc, const Cnt cnt)
    {
        assert(cnt >= 0);
        assert(cnt <= std::numeric_limits<decltype(cnt_)::value_type>::max());
        proc_ = std::ref(procFunc);
        submit_ = std::ref(submitFunc);
        cnt_ = cnt;

        auto r = proc_();

        std::unique_lock lck{mx_};

        if (r.has_value()) {
            submit_(r);
        }

        cvMain_.wait(lck, [this]{
            return busyWorkerThreads_ == 0;
        });
    }

    void runTh_()
    {
        std::any r;

        while(!stop_) {
            std::unique_lock lck{mx_};

            if (r.has_value()) {
                submit_(r);
                r.reset();
            }

            assert(busyWorkerThreads_ > 0);
            if (--busyWorkerThreads_ == 0) {
                cvMain_.notify_one();
            }
            cvThs_.wait(lck, [this]{
                return cnt_ > 0 || stop_;
            });
            if (stop_) {
                return;
            }
            ++busyWorkerThreads_;
            lck.unlock();

            r = proc_();
        }
    }

private:
    bool stop_ = false;
    std::size_t busyWorkerThreads_;
    std::mutex mx_;
    std::vector<std::thread> threads_;
    std::condition_variable cvThs_;
    std::condition_variable cvMain_;

    std::function<std::any()> proc_;
    std::function<void(std::any&)> submit_;
    std::atomic<std::ptrdiff_t> cnt_ = 0;
};


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
        assert(ez::std23::to_underlying(idx) >= 0);
        assert(ez::utils::toUnsigned(ez::std23::to_underlying(idx)) < std::numeric_limits<SizeType>::max());

        return Base::operator[](ez::utils::toUnsigned(ez::std23::to_underlying(idx)));
    }

    auto& operator[](Idx idx) const
    {
        assert(ez::std23::to_underlying(idx) >= 0);
        assert(ez::utils::toUnsigned(ez::std23::to_underlying(idx)) < std::numeric_limits<SizeType>::max());

        return Base::operator[](ez::utils::toUnsigned(ez::std23::to_underlying(idx)));
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
            sparseSet_.resize(ez::std23::to_underlying(v)+1);
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
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
};

class CalcGraph {
public:
    class Node;
    enum class NodeRank {Nil = -1};

    enum class NodeId : std::size_t {};

    class Expression;
    enum class Value : std::int64_t {};

    template<typename T>
    using IsExpression = std::is_same<std::decay_t<T>, Expression>;

    template<typename T>
    using IsValue = std::is_same<std::decay_t<T>, Value>;

public:
    template<typename ValueType>
    NodeId addNode(ValueType&& expr, ThreadPool& pool)
    {
        using DecayedValueType = std::decay_t<ValueType>;
        constexpr bool vtIsExprOrValue = IsExpression<DecayedValueType>::value || IsValue<DecayedValueType>::value;
        static_assert(vtIsExprOrValue, "ValueType must be either CalcGraph::Value or CalcGraph::Expression.");

        auto nodeId = NodeId{nodes_.size()};
        nodes_.emplace_back(*this, nodeId, Value{});

        resetValueAt(nodeId, std::forward<ValueType>(expr), pool);
        return nodeId;
    }

    template<typename ValueType>
    void resetValueAt(NodeId nodeId, ValueType&& value, ThreadPool&);

    Value getValueAt(NodeId) const;

private:
    RIVector<Node, NodeId> nodes_;
    SparseSet<NodeId> stopNodes_;       // Used in resetValueAt for handle possible cycles.


    struct LayerInfo {
        LayerInfo(std::ptrdiff_t layerBuffOffset)
            : baseBuffOffset{layerBuffOffset}
        {}

        LayerInfo(const LayerInfo& oth) noexcept
            : baseBuffOffset{oth.baseBuffOffset},
              capacityDeltaUpdate{oth.capacityDeltaUpdate.load()},
              schedSize{oth.schedSize.load()}
        {}

        LayerInfo(LayerInfo&& oth) noexcept
            : LayerInfo{oth}
        {}

        ~LayerInfo() = default;

        LayerInfo& operator=(LayerInfo&& oth) noexcept
        {
            *this = oth;
            return *this;
        }

        LayerInfo& operator=(const LayerInfo& oth)
        {
            baseBuffOffset = oth.baseBuffOffset;
            capacityDeltaUpdate = oth.capacityDeltaUpdate.load();
            schedSize = oth.schedSize.load();
            return *this;
        }

        std::ptrdiff_t baseBuffOffset;
        std::atomic<std::ptrdiff_t> capacityDeltaUpdate = 0;
        std::atomic<std::ptrdiff_t> schedSize = 0;
    };

    struct Layers {
        std::vector<NodeId> buff;
        RIVector<LayerInfo, NodeRank> info;
    };

    Layers layers_;


    friend auto operator - (NodeRank lhs, NodeRank rhs) noexcept
    {
        return ez::std23::to_underlying(lhs) - ez::std23::to_underlying(rhs);
    }
};

class CalcGraph::Expression {
public:
    Expression() = default;

    template<
        typename NodeIdSet,
        typename = std::void_t<decltype(std::begin(std::declval<NodeIdSet&>()))>
    >
    Expression(const NodeIdSet& nodes)
        : nodes_(std::begin(nodes), std::end(nodes))
    {
        using NodeIdSetValType =
            typename std::iterator_traits<decltype(std::begin(nodes))>::value_type;

        static_assert(std::is_same_v<std::decay_t<NodeIdSetValType>, CalcGraph::NodeId>,
                "CalcGraph::Expression must be initialized with a set of CalcGraph::NodeIds");
    }

    Value calc(const CalcGraph& graph) const
    {
        std::underlying_type_t<Value> sum{};

        for (auto& m : nodes_) {
            sum += ez::std23::to_underlying(graph.getValueAt(m));
        }

        return Value{sum};
    }

    const std::vector<NodeId>& dependencies() const noexcept
    {
        return nodes_;
    }

private:
    /// A set of nodes whose values comprise the value of this node.
    std::vector<NodeId> nodes_;
};

class CalcGraph::Node {
public:
    Node(const CalcGraph&, NodeId /*thisNodeId*/, Value v) noexcept
        : value_{v}
    {}

    template<
        typename Expr,
        typename = std::enable_if_t<IsExpression<Expr>::value>
    >
    Node(CalcGraph& graph, NodeId thisNodeId, Expr&& expr) noexcept
        : expr_{std::forward<Expr>(expr)}
    {
        fetchValueAndRank(graph);

        // Subscribe this node for updates.
        for (auto nodeId : expr_.dependencies()) {
            auto& depNode = graph.nodes_[nodeId];
            depNode.subscribedToUpdate_.push_back(thisNodeId);
        }
    }

    /// Reset node value and rank.
    ///
    /// @note
    /// @a thisNodeId is supposed to be equal to id initialy used
    /// to construct this instance.
    template<typename ValueType>
    void reset(CalcGraph& graph, NodeId thisNodeId, ValueType&& expr)
    {
        using DecayedValueType = std::decay_t<ValueType>;
        constexpr bool vtIsExprOrValue = IsExpression<DecayedValueType>::value || IsValue<DecayedValueType>::value;
        static_assert(vtIsExprOrValue, "ValueType must be either CalcGraph::Value or CalcGraph::Expression.");

        if constexpr (IsExpression<DecayedValueType>::value) {
            // Unsubscribe this node for updates by current adjacent dependency nodes.
            for (auto nodeId : expr_.dependencies()) {
                auto& depNode = graph.nodes_[nodeId];
                auto i = std::find(depNode.subscribedToUpdate_.cbegin(), depNode.subscribedToUpdate_.cend(), thisNodeId);
                depNode.subscribedToUpdate_.erase(i);
            }
        }

        // Save subscribed node ids.
        auto tmp = std::move(subscribedToUpdate_);

        *this = Node{graph, thisNodeId, std::forward<ValueType>(expr)};
        assert(subscribedToUpdate_.empty());

        // Restore subscribed node ids.
        subscribedToUpdate_ = std::move(tmp);
    }

    NodeRank rank() const noexcept
    {
        return rank_;
    }

    Value value() const noexcept
    {
        return value_;
    }

    const std::vector<NodeId>& subscribedNodeIds() const noexcept
    {
        return subscribedToUpdate_;
    }

    /// Actualize value and rank if expr isn't empty.
    void fetchValueAndRank(const CalcGraph& graph)
    {
        value_ = expr_.calc(graph);
        rank_ = fetchRank_(graph);
    }

private:
    /// Determine actual rank of this node within @a graph.
    ///
    /// Rank is calculated by finding max rank of among adjacent
    /// dependency nodes and incrementing it by one, if there are
    /// no dependencies result rank is 0.
    NodeRank fetchRank_(const CalcGraph& graph) const noexcept
    {
        NodeRank rank{-1};

        for (auto nodeId : expr_.dependencies()) {
            auto& depNode = graph.nodes_[nodeId];
            rank = std::max(rank, depNode.rank_);
        }

        return NodeRank{ez::std23::to_underlying(rank) + 1};
    }

private:
    Expression expr_;
    NodeRank rank_ = NodeRank::Nil;
    Value value_;

    /// A set of nodes dependent on the value/rank of this node.
    std::vector<NodeId> subscribedToUpdate_;


public:
    NonAtomicallyMovableAtomicFlag isScheduledForIncrementalUpdate;
};

template<typename ValueType>
void CalcGraph::resetValueAt(NodeId nodeId, ValueType&& expr, ThreadPool& pool)
{
    using DecayedValueType = std::decay_t<ValueType>;
    constexpr bool vtIsExprOrValue = IsExpression<DecayedValueType>::value || IsValue<DecayedValueType>::value;
    static_assert(vtIsExprOrValue, "ValueType must be either CalcGraph::Value or CalcGraph::Expression.");

    assert(nodeId < nodes_.rsize());

    // Add predecessors to layers.
    auto schedulePredsForProcessing = [this](NodeId id) {
        auto minTouchedRank = NodeRank::Nil;
        auto maxTouchedRank = NodeRank::Nil;

        // Populate layers (schedule processing of predecessors).
        for (auto predNodeId : nodes_[id].subscribedNodeIds()) {
            // Avoid cycles.
            if (stopNodes_.contains(predNodeId)) {
                continue;
            }

            assert(predNodeId < nodes_.rsize());

            auto predNodeRank = nodes_[predNodeId].rank();
            assert(predNodeRank != NodeRank::Nil);

            std::tie(minTouchedRank, maxTouchedRank) =
                    std::minmax({minTouchedRank, maxTouchedRank, predNodeRank});

            if (auto& predNode = nodes_[predNodeId]; predNode.isScheduledForIncrementalUpdate.testAndSet()) {
                continue;
            }

            auto offset = layers_.info[predNodeRank].schedSize++;
            auto layerBaseBuffOffset = layers_.info[predNodeRank].baseBuffOffset;
            assert(ez::utils::toUnsigned(layerBaseBuffOffset + offset) < layers_.buff.size());
            layers_.buff[ez::utils::toUnsigned(layerBaseBuffOffset + offset)] = predNodeId;
        }

        return std::pair{minTouchedRank, maxTouchedRank};
    };

    auto update = [this, graph=this, &schedulePredsForProcessing](NodeId id) {
        auto& node = nodes_[id];

        node.isScheduledForIncrementalUpdate.clear();

        const auto originalRank = node.rank();
        node.fetchValueAndRank(*graph);
        const auto newRank = node.rank();

        if (newRank != originalRank) {
            assert(newRank < layers_.info.rsize());
            layers_.info[originalRank].capacityDeltaUpdate--;
            layers_.info[newRank].capacityDeltaUpdate++;
        }

        schedulePredsForProcessing(id);

        return std::max(newRank, originalRank);
    };

    stopNodes_.clear();
    if constexpr (!std::is_same_v<DecayedValueType, Value>) {
        // Temporarly mark successor nodes to prevent cycling.
        for (auto id : expr.dependencies()) {
            stopNodes_.set(id);
        }
    }

    auto& node = nodes_[nodeId];
    const auto originalRank = node.rank();
    node.reset(*this, nodeId, std::forward<ValueType>(expr));
    const auto newRank = node.rank();

    // Space in buff is only necessary for nodes with rank > 0.
    const auto schedulePopBuf = newRank < originalRank;

    auto rankToStartFrom = originalRank;
    auto maxTouchedRank = newRank;

    if (newRank > originalRank) {
        // For each node with rank > 0 reserve one element in buf.
        if (originalRank == NodeRank::Nil) {
            layers_.buff.push_back({});
        }

        // Add more layer descriptors if node is moved beyoud the current set of layers.
        assert(layers_.info.size() <= nodes_.size());
        const auto rankDiff = ez::utils::toUnsigned(newRank - originalRank);
        const auto layersCntToAdd = std::min<std::size_t>(rankDiff,  nodes_.size() - layers_.info.size());

        if (layersCntToAdd > 0) {
            layers_.info.resize(layers_.info.size() + layersCntToAdd, 0);
        }

        assert(newRank <= layers_.info.rsize());
        layers_.info[newRank].capacityDeltaUpdate++;

        // Nil rank elements don't have dedicated element in layers_ structure.
        if (originalRank != NodeRank::Nil) {
            layers_.info[originalRank].capacityDeltaUpdate--;
        }
        else {
           rankToStartFrom = newRank;
           assert(!layers_.info.empty() && "There must be at least one layer corresponding to newRank");
           maxTouchedRank = static_cast<NodeRank>(layers_.info.size() - 1);
        }
    }
    else if (newRank < originalRank){
        rankToStartFrom = newRank;
        maxTouchedRank = originalRank;

        layers_.info[originalRank].capacityDeltaUpdate--;

        // Nil rank elements don't have dedicated element in layers_ structure.
        if (newRank != NodeRank::Nil) {
            layers_.info[newRank].capacityDeltaUpdate++;
        }
        else {
            assert(!layers_.info.empty() && "There must be at least one layer corresponding to originalRank");
            maxTouchedRank = static_cast<NodeRank>(layers_.info.size() - 1);
        }
    }

    auto [mi, ma] = schedulePredsForProcessing(nodeId);


    if (rankToStartFrom == NodeRank::Nil) {
        if (mi == NodeRank::Nil) {
            // Move rankToStartFrom to a valid range ranks (low boundary) to iterate over layers.
            rankToStartFrom = NodeRank{0};
        }
        else {
            rankToStartFrom = mi;
        }
    }
    else if (mi == NodeRank::Nil) {
        rankToStartFrom = NodeRank{0};
    }
    else {
        rankToStartFrom = std::min(rankToStartFrom, mi);
    }

    maxTouchedRank = std::max(maxTouchedRank, ma);

    auto layerBuffBaseDelta = std::ptrdiff_t{0};
    for (auto i = ez::std23::to_underlying(rankToStartFrom); i <= ez::std23::to_underlying(maxTouchedRank); ++i) {
        auto currRank = NodeRank{i};
        assert(currRank < layers_.info.rsize());
        auto& layer = layers_.info[currRank];
        assert(layer.baseBuffOffset >= 0);

        if (layer.schedSize > 0) {
            auto wrBegin = layers_.buff.begin() + layer.baseBuffOffset;
            auto wrEnd = wrBegin + layer.schedSize;
            auto max = [](auto a, auto b) { return std::max(a, b); };
            maxTouchedRank = pool.mapReduce(wrBegin, wrEnd, maxTouchedRank, update, max);
        }

        // Commit current layer summary changes.
        layer.baseBuffOffset += layerBuffBaseDelta;
        layerBuffBaseDelta += layer.capacityDeltaUpdate;   // changes of capacity of this layer affect buffOffset* of further layers.
        layer.capacityDeltaUpdate = 0;
        layer.schedSize = 0;
    }

    if (schedulePopBuf) {
        layers_.buff.pop_back();
    }
}

CalcGraph::Value CalcGraph::getValueAt(CalcGraph::NodeId nodeId) const
{
    assert(nodeId < nodes_.rsize());
    return nodes_[nodeId].value();
}
}

#endif // ACTION_NODES_GRAPH_H
