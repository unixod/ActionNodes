#ifndef ACTION_NODES_GRAPH_H
#define ACTION_NODES_GRAPH_H

#include <cassert>
#include <algorithm>
#include <atomic>
#include <numeric>
#include <type_traits>
#include <span>
#include <optional>
#include <ranges>
#include <ez/support/std23.h>
#include <ez/utils/utils.h>
#include <ez/utils/enum-arithmetic.h>
#include "action-nodes/utils/thread-pool.h"
#include "action-nodes/utils/rivector.h"
#include "action-nodes/utils/sparse-set.h"
#include "action-nodes/utils/non-atomically-movable-atomic-flag.h"

namespace anodes {

class Graph {
public:
    enum class NodeId : std::size_t {};

public:
    NodeId addNode()
    {
        auto nodeId = NodeId{nodes_.size()};
        nodes_.emplace_back();
        return nodeId;
    }

    /// Remove all outedges from nodeId and add new ones which point to @a deps.
    ///
    /// Formarly speaking, given a graph G and relation R representing the
    /// interconnetions between nodes within G, this function removes all
    /// pairs (a, x) from R, where a is a node whose id is @a nodeId, and
    /// then adds a new set of pairs of form: (a, y) where y is nodes whose ids
    /// are given in @nodeNewDependencies.
    template<
        typename NodeIdsRange,
        std::invocable<NodeId> UpdateHandler,
        typename MapReduceEngine
    >
    void reorder(NodeId node, NodeIdsRange&& nodeNewDependencies, MapReduceEngine&&, UpdateHandler&& updateHandler);

    template<
        typename UpdateHandler,
        typename MapReduceEngine
    >
    void touch(NodeId nodeId, MapReduceEngine&&, UpdateHandler&&);

    std::span<const NodeId> getNodeDeps(NodeId) const;

private:
    class Node;
    enum class NodeRank {Nil = -1};

private:
    template<
        typename NodeIdsRange,
        std::invocable<NodeId> UpdateHandler,
        typename MapReduceEngine
    >
    void propagate_(NodeId node, NodeRank rankToStartFrom, NodeRank maxTouchedRank, NodeIdsRange&& nodeNewDependencies, MapReduceEngine&&, UpdateHandler&& updateHandler);


private:
    utils::RIVector<Node, NodeId> nodes_;
//    utils::SparseSet<NodeId> stopNodes_;       // Used in resetValueAt for handle possible cycles.


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
struct ez::utils::EnableEnumArithmeticFor<anodes::Graph::NodeRank>{};

namespace anodes {

class Graph::Node {
public:
    /// Reset node value and rank.
    ///
    /// @note
    /// @a thisNodeId is supposed to be equal to id initialy used
    /// to construct this instance.
    template<typename DepNodesRange>
    void reset(Graph& graph, NodeId thisNodeId, DepNodesRange&& newDeps)
    {
        // Unsubscribe this node for updates by current set of dependency nodes.
        for (auto nodeId : deps_) {
            auto& depNode = graph.nodes_[nodeId];
            auto i = std::ranges::find(depNode.subscribedToUpdate_, thisNodeId);
            depNode.subscribedToUpdate_.erase(i);
        }

        // Subscribe this node for updates by a new set of dependencies.
        for (auto nodeId : newDeps) {
            auto& depNode = graph.nodes_[nodeId];
            depNode.subscribedToUpdate_.push_back(thisNodeId);
        }

        rank_ = getNextRank_(graph, newDeps);
        deps_.assign(std::cbegin(newDeps), std::cend(newDeps));
    }

    NodeRank rank() const noexcept
    {
        return rank_;
    }

    std::span<const NodeId> dependencies() const noexcept
    {
        return deps_;
    }

    std::span<const NodeId> subscribedNodeIds() const noexcept
    {
        return subscribedToUpdate_;
    }

    void fetchRank(const Graph& graph) noexcept
    {
        rank_ = getNextRank_(graph, deps_);
    }

private:
    /// Determine actual rank of this node within @a graph.
    ///
    /// Rank is calculated by finding max rank of among adjacent
    /// dependency nodes and incrementing it by one, if there are
    /// no dependencies result rank is 0.
    template<typename NodeIdsRange>
    static NodeRank getNextRank_(const Graph& graph, NodeIdsRange&& nodes) noexcept
    {
        NodeRank rank = NodeRank::Nil;

        for (auto nodeId : nodes) {
            auto& depNode = graph.nodes_[nodeId];
            rank = std::max(rank, depNode.rank());
        }

        using namespace ez::utils::enum_arithmethic;
        return rank + 1;
    }

private:
    std::vector<NodeId> deps_;
    NodeRank rank_ = NodeRank::Nil;

    /// A set of nodes dependent on the value/rank of this node.
    std::vector<NodeId> subscribedToUpdate_;


public:
    utils::NonAtomicallyMovableAtomicFlag isScheduledForIncrementalUpdate;
};

inline std::span<const Graph::NodeId> Graph::getNodeDeps(Graph::NodeId id) const
{
    return nodes_[id].dependencies();
}

template<
    typename NodeIdsRange,
    std::invocable<Graph::NodeId> UpdateHandler,
    typename MapReduceEngine
>
void Graph::propagate_(NodeId nodeId, NodeRank startRank, NodeRank endRank, NodeIdsRange&& depNodes, MapReduceEngine&& pool, UpdateHandler&& updateHandler)
{
    assert(nodeId < nodes_.rsize());
    assert(startRank != NodeRank::Nil);

    constexpr auto isReorderingOperaion = !std::is_same_v<std::decay_t<NodeIdsRange>, std::nullopt_t>;

    if constexpr (isReorderingOperaion) {
        assert(
            std::ranges::all_of(depNodes, [this](auto n){ return n < nodes_.rsize(); }) && "All nodes from depNodes (if any) must exist."
        );
    }

    // Add predecessors to layers.
    auto schedulePredsForProcessing = [this, skipNodeId = nodeId](NodeId id) {
        auto minTouchedRank = static_cast<NodeRank>(layers_.info.size());
        auto maxTouchedRank = NodeRank::Nil;

        // Populate layers (schedule processing of predecessors).
        for (auto predNodeId : nodes_[id].subscribedNodeIds()) {
            // Avoid cycles.
            if (predNodeId == skipNodeId) {
                continue;
            }

            assert(predNodeId < nodes_.rsize());

            const auto predNodeRank = nodes_[predNodeId].rank();
            assert(predNodeRank != NodeRank::Nil);

            minTouchedRank = std::min(minTouchedRank, predNodeRank);
            maxTouchedRank = std::min(maxTouchedRank, predNodeRank);

            if (auto& predNode = nodes_[predNodeId]; predNode.isScheduledForIncrementalUpdate.testAndSet()) {
                continue;
            }

            const auto offset = layers_.info[predNodeRank].schedSize++;
            const auto layerBaseBuffOffset = layers_.info[predNodeRank].baseBuffOffset;
            assert(ez::utils::toUnsigned(layerBaseBuffOffset + offset) < layers_.buff.size());
            layers_.buff[ez::utils::toUnsigned(layerBaseBuffOffset + offset)] = predNodeId;
        }

        assert(minTouchedRank != NodeRank::Nil);
        return std::pair{minTouchedRank, maxTouchedRank};
    };

    auto update = [this, graph=this, &schedulePredsForProcessing, &updateHandler](NodeId id) {
        auto& node = nodes_[id];

        node.isScheduledForIncrementalUpdate.clear();

        if constexpr (isReorderingOperaion) {
            const auto originalRank = node.rank();
            node.fetchRank(*graph);
            updateHandler(id);
            const auto newRank = node.rank();

            if (newRank != originalRank) {
                assert(newRank < layers_.info.rsize());
                layers_.info[originalRank].capacityDeltaUpdate--;
                layers_.info[newRank].capacityDeltaUpdate++;
            }

            auto [mi, ma] = schedulePredsForProcessing(id);

            return std::max({newRank, originalRank, ma});
        }
        else {
            updateHandler(id);
            auto [mi, ma] = schedulePredsForProcessing(id);
            return std::max(node.rank(), ma);
        }
    };

    auto [mi, ma] = schedulePredsForProcessing(nodeId);
    const auto rankToStartFrom = std::min(startRank, mi);
    auto maxTouchedRank = std::max(endRank, ma);

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
}

template<
    typename NodeIdsRange,
    std::invocable<Graph::NodeId> UpdateHandler,
    typename MapReduceEngine
>
void Graph::reorder(NodeId nodeId, NodeIdsRange&& depNodes, MapReduceEngine&& pool, UpdateHandler&& updateHandler)
{
    auto& node = nodes_[nodeId];
    const auto originalRank = node.rank();
    node.reset(*this, nodeId, depNodes);
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

        //==-----------------------------------------------------------------------------==//
        // Add more layer descriptors if node is moved beyoud the current set of layers.
        //==-----------------------------------------------------------------------------==//
        assert(layers_.info.size() <= nodes_.size() && "Number of ranks can't exceed the number of nodes.");
        using namespace ez::utils::enum_arithmethic;

        const auto maxPossibleNumOrRanks = nodes_.size();
        const auto maxPossibleNumOfNewRanks = maxPossibleNumOrRanks - layers_.info.size(); // Invariant: Number of ranks can't exceed total number of nodes.
        const auto maxPossibleNumOfNewRanksDueToThisOperation = ez::utils::toUnsigned(ez::support::std23::to_underlying(newRank - originalRank));
        const auto layersCntToAdd = std::min<std::size_t>(maxPossibleNumOfNewRanksDueToThisOperation,  maxPossibleNumOfNewRanks);

        if (layersCntToAdd > 0) {
            layers_.info.resize(layers_.info.size() + layersCntToAdd, 0);
        }

        assert(layers_.info.size() <= nodes_.size() && "Number of ranks can't exceed the number of nodes.");
        assert(newRank < layers_.info.rsize());

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
        layers_.info[originalRank].capacityDeltaUpdate--;

        // Nil rank elements don't have dedicated element in layers_ structure.
        if (newRank != NodeRank::Nil) {
            rankToStartFrom = newRank;
            maxTouchedRank = originalRank;
            layers_.info[newRank].capacityDeltaUpdate++;
        }
        else {
            assert(!layers_.info.empty() && "There must be at least one layer corresponding to originalRank");
            assert(rankToStartFrom == originalRank);
            maxTouchedRank = static_cast<NodeRank>(layers_.info.size() - 1);
        }
    }
    else {
        assert(newRank == originalRank);
        rankToStartFrom = static_cast<NodeRank>(layers_.info.size());
        maxTouchedRank = NodeRank::Nil;
        assert(rankToStartFrom > maxTouchedRank && "The range must be non-iteratable.");
    }

    assert(rankToStartFrom != NodeRank::Nil);

    propagate_(nodeId, rankToStartFrom, maxTouchedRank, depNodes, pool, updateHandler);

    if (schedulePopBuf) {
        layers_.buff.pop_back();
    }
}

template<
    typename UpdateHandler,
    typename MapReduceEngine
>
void Graph::touch(NodeId nodeId, MapReduceEngine&& pool, UpdateHandler&& updateHandler)
{
    const auto rankToStartFrom = static_cast<NodeRank>(layers_.info.size());
    const auto maxTouchedRank = NodeRank::Nil;
    assert(rankToStartFrom > maxTouchedRank && "The range must be non-iteratable.");

    propagate_(nodeId, rankToStartFrom, maxTouchedRank, std::nullopt, pool, updateHandler);
}

} //  anodes

#endif // ACTION_NODES_GRAPH_H
