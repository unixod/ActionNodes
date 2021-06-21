#ifndef ACTION_NODES_GRAPH_H
#define ACTION_NODES_GRAPH_H

#include <cassert>
#include <algorithm>
#include <atomic>
#include <numeric>
#include <type_traits>
#include <ez/support/std23.h>
#include <ez/utils/utils.h>
#include <ez/utils/enum-arithmetic.h>
#include "action-nodes/utils/thread-pool.h"
#include "action-nodes/utils/rivector.h"
#include "action-nodes/utils/sparse-set.h"
#include "action-nodes/utils/non-atomically-movable-atomic-flag.h"

namespace anodes {



class CalcGraph {
public:
    enum class NodeId : std::size_t {};

    class Expression;
    enum class Value : std::int64_t {};

private:
    class Node;
    enum class NodeRank {Nil = -1};

    template<typename T>
    using IsExpression = std::is_same<std::decay_t<T>, Expression>;

    template<typename T>
    using IsValue = std::is_same<std::decay_t<T>, Value>;

public:
    template<typename ValueType>
    NodeId addNode(ValueType&& expr, utils::ThreadPool& pool)
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
    void resetValueAt(NodeId nodeId, ValueType&& value, utils::ThreadPool&);

    Value getValueAt(NodeId) const;

private:
    utils::RIVector<Node, NodeId> nodes_;
    utils::SparseSet<NodeId> stopNodes_;       // Used in resetValueAt for handle possible cycles.


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
        utils::RIVector<LayerInfo, NodeRank> info;
    };

    Layers layers_;
};

} //  anodes

template<>
struct ez::utils::EnableEnumArithmeticFor<anodes::CalcGraph::NodeRank>{};

namespace anodes {
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
            sum += ez::support::std23::to_underlying(graph.getValueAt(m));
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

        using namespace ez::utils::enum_arithmethic;
        return rank + 1;
    }

private:
    Expression expr_;
    NodeRank rank_ = NodeRank::Nil;
    Value value_;

    /// A set of nodes dependent on the value/rank of this node.
    std::vector<NodeId> subscribedToUpdate_;


public:
    utils::NonAtomicallyMovableAtomicFlag isScheduledForIncrementalUpdate;
};


template<typename ValueType>
void CalcGraph::resetValueAt(NodeId nodeId, ValueType&& expr, utils::ThreadPool& pool)
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
        using namespace ez::utils::enum_arithmethic;
        const auto rankDiff = ez::utils::toUnsigned(ez::support::std23::to_underlying(newRank - originalRank));
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
    using namespace ez::utils::enum_arithmethic;
    for (auto i = rankToStartFrom; i <= maxTouchedRank; ++i) {
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
} //  anodes

#endif // ACTION_NODES_GRAPH_H
