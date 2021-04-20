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
#include "utils.h"
#include "parser.h"



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
            threads_.emplace_back( [this]{ run_(); });
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

        cv_.notify_all();

        for (auto& th : threads_) {
            th.join();
        }
    }

    template<typename Callable>
    void runAsync(Callable&& func)
    {
        std::unique_lock lck{mx_};
        queue_.emplace_back(std::forward<Callable>(func));

        if (mainThreadWaiting) {
            lck.unlock();
            cvMain_.notify_one();
            return;
        }
        else {
            lck.unlock();
            cv_.notify_one();
        }
    }

    void wait()
    {
        while (true) {
            std::unique_lock lck{mx_};

            // suspend if queue.empty && busyWorkerThreads != 0
            cvMain_.wait(lck, [this](){
                return !queue_.empty() || busyWorkerThreads_ == 0;
            });

            if (busyWorkerThreads_ == 0) {
                assert(queue_.empty());
                return;
            }

            auto f = std::move(queue_.front());
            queue_.pop_front();
            lck.unlock();
            f();
        }
    }

private:
    void run_()
    {
        while(true) {
            std::unique_lock lck{mx_};

            cv_.wait(lck, [this]{
                if (--busyWorkerThreads_ == 0 && queue_.empty()) {
                    cvMain_.notify_one();
                }

                return !queue_.empty() || stop_;
            });

            if (stop_) {
                return;
            }

            ++busyWorkerThreads_;

            auto f = std::move(queue_.front());
            queue_.pop_front();
            lck.unlock();

            f();
        }
    }

private:
    std::mutex mx_;
    std::condition_variable cv_;
    std::condition_variable cvMain_;
    std::deque<std::function<void()>> queue_;
    std::vector<std::thread> threads_;

    bool stop_ = false;
    bool mainThreadWaiting = true;
    std::size_t busyWorkerThreads_;
};

class CalcGraph {
public:
    class Node;
    enum class NodeRank {};
    enum class NodeId : std::size_t {};

    class Expression;
    enum class Value : std::int64_t {};
    
    template<typename T>
    using IsExpression = std::is_same<std::decay_t<T>, Expression>;

    template<typename T>
    using IsValue = std::is_same<std::decay_t<T>, Value>;

public:
    template<typename ValueType>
    NodeId addNode(ValueType&& expr)
    {
        using DecayedValueType = std::decay_t<ValueType>;
        constexpr bool vtIsExprOrValue = IsExpression<DecayedValueType>::value || IsValue<DecayedValueType>::value;
        static_assert(vtIsExprOrValue, "ValueType must be either CalcGraph::Value or CalcGraph::Expression.");

        auto nodeId = NodeId{nodes_.size()};
//        nodes_.emplace_back(*this, nodeId, std::move(expr));
        nodes_.emplace_back(*this, nodeId, Value{});
        resetValueAt(nodeId, std::forward<ValueType>(expr));
        return nodeId;
    }

    template<typename ValueType>
    void resetValueAt(NodeId nodeId, ValueType&& value/*, ThreadPool&*/);

    Value getValueAt(NodeId) const;

private:
    calc::utils::RIVector<Node, NodeId> nodes_;
    calc::utils::SparseSet<NodeId> stopNodes_;       // Used in resetValueAt for handle possible cycles.

    struct LayerInfo {
        LayerInfo(std::int64_t layerBuffOffset)
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

        std::int64_t baseBuffOffset;
        std::atomic<std::int64_t> capacityDeltaUpdate = 0;
        std::atomic<std::int64_t> schedSize = 0;
    };

    struct Layers {
        std::vector<NodeId> buff;
        calc::utils::RIVector<LayerInfo, NodeRank> info{LayerInfo{0}};

        std::size_t numOfNewLayers = 0;
    };

    Layers layers_;


    friend auto operator - (NodeRank lhs, NodeRank rhs) noexcept
    {
        return calc::utils::toUnderlying(lhs) - calc::utils::toUnderlying(rhs);
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
            sum += calc::utils::toUnderlying(graph.getValueAt(m));
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
        : rank_{0}, value_{v}
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

        return NodeRank{calc::utils::toUnderlying(rank) + 1};
    }

private:
    Expression expr_;
    NodeRank rank_;
    Value value_;

    /// A set of nodes dependent on the value/rank of this node.
    std::vector<NodeId> subscribedToUpdate_;
};



template<typename ValueType>
void CalcGraph::resetValueAt(NodeId nodeId, ValueType&& expr/*, ThreadPool& pool*/)
{
    using DecayedValueType = std::decay_t<ValueType>;
    constexpr bool vtIsExprOrValue = IsExpression<DecayedValueType>::value || IsValue<DecayedValueType>::value;
    static_assert(vtIsExprOrValue, "ValueType must be either CalcGraph::Value or CalcGraph::Expression.");

    assert(nodeId < nodes_.rsize());

    // TODO: Think about possible optimization: buff.resize(nodes_.size() - num of nodes in node_[nodeId].rank());
    layers_.buff.resize(nodes_.size());

    // Add predecessors to layer.
    auto schedulePredsForProcessing = [this](NodeId id) {
        NodeRank maxRank{};

        // Populate layers (schedule processing of predecessors).
        for (auto predNodeId : nodes_[id].subscribedNodeIds()) {
            // Avoid cycles.
            if (stopNodes_.contains(predNodeId)) {
                continue;
            }

            assert(predNodeId < nodes_.rsize());

            auto predNodeRank = nodes_[predNodeId].rank();

            auto offset = layers_.info[predNodeRank].schedSize++;
            auto layerBaseBuffOffset = layers_.info[predNodeRank].baseBuffOffset;
            layers_.buff[calc::utils::makeUnsigned(layerBaseBuffOffset + offset)] = predNodeId;

            if (maxRank < predNodeRank) {
                maxRank = predNodeRank;
            }
        }
        return maxRank;
    };


    auto update = [this, graph=this, &schedulePredsForProcessing](NodeId id){
        auto& node = nodes_[id];

        auto originalRank = node.rank();

        node.fetchValueAndRank(*graph);

        if (auto newRank = node.rank(); newRank != originalRank) {
            assert(newRank < layers_.info.rsize());
            layers_.info[originalRank].capacityDeltaUpdate--;
            layers_.info[newRank].capacityDeltaUpdate++;
        }

        return schedulePredsForProcessing(id);
    };

    stopNodes_.clear();
    if constexpr (!std::is_same_v<DecayedValueType, Value>) {
        // Temporarly mark successor nodes to prevent cycling.
        for (auto id : expr.dependencies()) {
            stopNodes_.set(id);
        }
    }

    auto& node = nodes_[nodeId];
    auto originalRank = node.rank();

    node.reset(*this, nodeId, std::forward<ValueType>(expr));

    if (auto newRank = node.rank(); newRank > originalRank) {
        // Add more layer descriptors if node is moved beyoud the current set of layers.
        if (newRank >= layers_.info.rsize()) {
            assert(layers_.info.size() <= nodes_.size());
            const auto rankDiff = calc::utils::makeUnsigned(newRank - originalRank);
            const auto layersCntToAdd = std::min<std::size_t>(rankDiff,  nodes_.size() - layers_.info.size());

            if (layersCntToAdd > 0) {
                layers_.info.resize(layers_.info.size() + layersCntToAdd, 0);
                layers_.numOfNewLayers = layersCntToAdd;
            }
        }

        assert(newRank <= layers_.info.rsize());
        layers_.info[originalRank].capacityDeltaUpdate--;
        layers_.info[newRank].capacityDeltaUpdate++;
    }
    else if (newRank < originalRank){
        for (auto i = calc::utils::toUnderlying(newRank) + 1; i <= calc::utils::toUnderlying(originalRank); ++i) {
            layers_.info[NodeRank{i}].baseBuffOffset += 1;
        }
    }

    // A point where to stop further processing (during processing of further layers this boundary may be changed
    // in a higher value).
    auto maxTouchedRank = schedulePredsForProcessing(nodeId);

    auto layerBuffBaseDelta = layers_.info[originalRank].capacityDeltaUpdate.load();
    layers_.info[originalRank].capacityDeltaUpdate = 0;

    // Update value and ranks of predecessors.
    for (auto i = calc::utils::toUnderlying(originalRank) + 1; i <= calc::utils::toUnderlying(maxTouchedRank); ++i) {
        auto currRank = NodeRank{i};
        assert(currRank < layers_.info.rsize());

        assert(layers_.info[currRank].baseBuffOffset >= 0);

        auto wrBegin = layers_.buff.begin() + layers_.info[currRank].baseBuffOffset;
        auto wrEnd = wrBegin + layers_.info[currRank].schedSize;
        auto max = [](auto a, auto b) { return std::max(a, b); };
        maxTouchedRank = std::transform_reduce(/*std::execution::par, */wrBegin, wrEnd, maxTouchedRank, max, update);

        // Commit current layer summary changes.
        layers_.info[currRank].baseBuffOffset += layerBuffBaseDelta;
        layerBuffBaseDelta += layers_.info[currRank].capacityDeltaUpdate;   // changes of capacity of this layer affect buffOffset* of further layers.
        layers_.info[currRank].capacityDeltaUpdate = 0;
        layers_.info[currRank].schedSize = 0;
    }

    assert(calc::utils::toUnderlying(maxTouchedRank)+1 == calc::utils::makeSigned(layers_.info.size() - layers_.numOfNewLayers)
           || layerBuffBaseDelta == 0);

//    for (auto i = maxTouchedRank + 1; i < layers_.info.size(); ++i) {

//    }

//    // If the layer corresponding to the newRank of the node hasn't been
//    // processed then check and if necessasry adjust base buf offset
//    // of this and all rest layers.
//    if (auto newRank = node.rank(); layers_.info[newRank].capacityDeltaUpdate != 0) {
//        assert(layers_.info[newRank].capacityDeltaUpdate > 0);

//        layerBuffBaseDelta = layers_.info[newRank].capacityDeltaUpdate;
//        auto lBegin = layers_.info.begin() + calc::utils::toUnderlying(newRank);
//        auto lEnd = layers_.info.end();
//        for (auto i = lBegin; i != lEnd; ++i) {
//            assert(i->schedSize == 0);
//            i->baseBuffOffset += layerBuffBaseDelta;
//            layerBuffBaseDelta += i->capacityDeltaUpdate;
//            i->capacityDeltaUpdate = 0;
//        }
//    }

    // Adjust base buffer offsets of new layers (if any).
    auto newLayersRBegin = layers_.info.rbegin();
    auto newLayersREnd = newLayersRBegin + static_cast<std::ptrdiff_t>(layers_.numOfNewLayers);
    layers_.numOfNewLayers = 0;

    std::size_t sumOfNodesInNextLayers = 0;
    for (auto i = newLayersRBegin; i != newLayersREnd; ++i)  {
        auto nodesOnLayer = i->capacityDeltaUpdate.load();
        assert(nodesOnLayer >= 0);
        sumOfNodesInNextLayers += calc::utils::makeUnsigned(nodesOnLayer);
        i->capacityDeltaUpdate = 0;
        i->baseBuffOffset = calc::utils::makeSigned(nodes_.size() - sumOfNodesInNextLayers);
        assert(i->baseBuffOffset > 0);
    }

    assert(
        std::all_of(layers_.info.begin(), layers_.info.end(), [](const auto& e){
            return e.capacityDeltaUpdate == 0 && e.schedSize == 0;
        })
    );
}

//template<typename Container>
//void CalcGraph::refresh(Container&& nodeIds, ThreadPool& exec)
//{
//    using IteratorType = typename std::decay_t<Container>::iterator;
//    static_assert(std::is_same_v<typename std::iterator_traits<IteratorType>::value_type, NodeId>,
//            "The function refresh expects a sequence of NodeId on input.");


//    auto proc = [this, &exec](auto self, NodeId id) {
//        auto& node = nodes_[utils::toUnderlying(id)];

//        if (node.refs-- == 0) {
//            node.expr.recalcValue();

//            // This node value is ready to read by dependant nodes,
//            // hence enqueue dependent node for similar processing.
//            for (auto& nodeId : node.dependentNodes()) {
//                exec.runAsync([self, nodeId]{
//                    self(self, nodeId);
//                });
//            }
//        }
//    };

//    for (auto& nodeId : nodeIds) {
//        exec.runAsync([nodeId, proc]{
//            proc(proc, nodeId);
//        });
//    }

//    exec.wait();
//}
//void CalcGraph::addDependency(CalcGraph::NodeId nodeId, CalcGraph::NodeId depNodeId)
//{
//    assert(toUnderlying(nodeId) < nodes_.size());
//    assert(toUnderlying(depNodeId) < nodes_.size());

//    auto& node = nodes_[toUnderlying(nodeId)];
//    auto& depNode = nodes_[toUnderlying(depNodeId)];
//    depNode.scheduleUpdate(node);
//}

CalcGraph::Value CalcGraph::getValueAt(CalcGraph::NodeId nodeId) const
{
    assert(nodeId < nodes_.rsize());
    return nodes_[nodeId].value();
}

int main()
{
    namespace parser = calc::parser;
    namespace utils = calc::utils;

//    ThreadPool pool;
    CalcGraph graph;

    using CellId = std::string;
    std::map<CellId, CalcGraph::NodeId, std::less<>> symbolTable;

    auto getNode = [&graph, &symbolTable](auto cellId) {
        auto i = symbolTable.lower_bound(cellId);
        if (i == symbolTable.end() || i->first != cellId) {
            auto n = graph.addNode(CalcGraph::Value{});
            i = symbolTable.emplace_hint(i, cellId, n);
        }
        return i->second;
    };

    auto resetNodeValue = [&graph, &symbolTable/*, &pool*/](auto cellId, auto&& value){
        auto cellIter = symbolTable.lower_bound(cellId);
        if (cellIter == symbolTable.end() || cellIter->first != cellId) {
            auto n = graph.addNode(std::forward<decltype(value)>(value));
            symbolTable.emplace_hint(cellIter, cellId, n);
        }
        else {
            graph.resetValueAt(cellIter->second, std::forward<decltype(value)>(value)/*, pool*/);
        }
    };

    for (std::string line; std::getline(std::cin, line) && !line.empty();) {
        auto [cellId, cellExpr] = parser::parseLine(line);

        utils::match(cellExpr,
            [&resetNodeValue, cellId = cellId](parser::Number n) {
                resetNodeValue(cellId, CalcGraph::Value{n});
            },
            [&getNode, &resetNodeValue, cellId = cellId](const parser::CellsSum& sum) {
                std::vector<CalcGraph::NodeId> cs;
                for (auto sumMemberCellId : sum) {
                    cs.push_back(getNode(sumMemberCellId));
                }
                resetNodeValue(cellId, CalcGraph::Expression{std::move(cs)});
            }
        );
    }

    for (auto& [cellId, nodeId] : symbolTable) {
        std::cout << cellId << '=' << utils::toUnderlying(graph.getValueAt(nodeId)) << '\n';
    }

    return EXIT_SUCCESS;
}


