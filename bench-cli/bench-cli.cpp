#include <map>
#include <iostream>
#include <ez/support/std23.h>
#include "action-nodes/graph.h"
#include "action-nodes/utils/thread-pool.h"
#include "action-nodes/bench-cli/parser.h"

namespace anodes::bench_cli::utils {

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

}

using Value = std::intptr_t;


using anodes::utils::ThreadPool;
using anodes::Graph;


auto cellValSum(const auto& nodes, const auto& nodesVs)
{
    auto getNodeVal = [&nodesVs](Graph::NodeId n){
        return nodesVs[n];
    };

    return std::transform_reduce(nodes.begin(), nodes.end(), Value{0}, std::plus<>{}, getNodeVal);
};

int main()
{
    namespace parser = anodes::bench_cli::parser;
    namespace utils = anodes::bench_cli::utils;

    ThreadPool pool;
    Graph graph;

    using CellId = std::string;
    std::map<CellId, Graph::NodeId, std::less<>> symbolTable;
    anodes::utils::RIVector<Value, Graph::NodeId> nodesValues;

    auto getNode = [&graph, &symbolTable, &nodesValues](auto cellId) {
        auto i = symbolTable.lower_bound(cellId);
        if (i == symbolTable.end() || i->first != cellId) {
            auto n = graph.addNode();

            nodesValues.push_back(0);
            assert(ez::support::std23::to_underlying(n) == nodesValues.size() - 1 &&
                   "Each element within nodesValues have a corresponding element in the graph");

            i = symbolTable.emplace_hint(i, cellId, n);
        }
        return i->second;
    };

    auto resetNodeValue = [&graph, &symbolTable, &pool, &nodesValues](auto cellId, auto&& value){
        auto cellIter = symbolTable.lower_bound(cellId);
        if (cellIter == symbolTable.end() || cellIter->first != cellId) {
            auto n = graph.addNode();

            nodesValues.push_back(value);
            assert(ez::support::std23::to_underlying(n) == nodesValues.size() - 1 &&
                   "Each element within nodesValues have a corresponding element in the graph");

            symbolTable.emplace_hint(cellIter, cellId, n);
        }
        else {
            nodesValues[cellIter->second] = value;

            graph.touch(cellIter->second, pool, [&nodesValues, &graph](Graph::NodeId nodeId){
                auto deps = graph.getNodeDeps(nodeId);
                nodesValues[nodeId] = cellValSum(deps, nodesValues);
            });
        }
    };


    auto resetNodeDeps = [&graph, &symbolTable, &pool, &nodesValues](auto cellId, auto&& value){
        auto cellIter = symbolTable.lower_bound(cellId);
        if (cellIter == symbolTable.end() || cellIter->first != cellId) {
            auto n = graph.addNode();

            nodesValues.push_back(cellValSum(value, nodesValues));
            assert(ez::support::std23::to_underlying(n) == nodesValues.size() - 1 &&
                   "Each element within nodesValues have a corresponding element in the graph");


            graph.reorder(n, std::forward<decltype(value)>(value), pool, [](Graph::NodeId){
                assert(false && "Unrecheable");
            });

            symbolTable.emplace_hint(cellIter, cellId, n);
        }
        else {
            nodesValues[cellIter->second] = cellValSum(value, nodesValues);

            graph.reorder(cellIter->second, std::forward<decltype(value)>(value), pool, [&nodesValues, &graph](Graph::NodeId nodeId){
                auto deps = graph.getNodeDeps(nodeId);
                nodesValues[nodeId] = cellValSum(deps, nodesValues);
            });
        }
    };

    for (std::string line; std::getline(std::cin, line) && !line.empty();) {
        auto [cellId, cellExpr] = parser::parseLine(line);

        utils::match(cellExpr,
            [&resetNodeValue, cellId = cellId](parser::Number n) {
                resetNodeValue(cellId, Value{n});
            },
            [&getNode, &resetNodeDeps, cellId = cellId](const parser::CellsSum& sum) {
                std::vector<Graph::NodeId> cs;
                for (auto sumMemberCellId : sum) {
                    cs.push_back(getNode(sumMemberCellId));
                }
                resetNodeDeps(cellId, cs);
            }
        );
    }

    for (auto& [cellId, nodeId] : symbolTable) {
        std::cout << cellId << '=' << nodesValues[nodeId] << '\n';
    }

    return EXIT_SUCCESS;
}
