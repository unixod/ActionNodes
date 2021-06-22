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

using anodes::utils::ThreadPool;
using anodes::Graph;

int main()
{
    namespace parser = anodes::bench_cli::parser;
    namespace utils = anodes::bench_cli::utils;

    ThreadPool pool;
    Graph graph;

    using CellId = std::string;
    std::map<CellId, Graph::NodeId, std::less<>> symbolTable;

    auto getNode = [&graph, &symbolTable, &pool](auto cellId) {
        auto i = symbolTable.lower_bound(cellId);
        if (i == symbolTable.end() || i->first != cellId) {
            auto n = graph.addNode(Graph::Value{}, pool);
            i = symbolTable.emplace_hint(i, cellId, n);
        }
        return i->second;
    };

    auto resetNodeValue = [&graph, &symbolTable, &pool](auto cellId, auto&& value){
        auto cellIter = symbolTable.lower_bound(cellId);
        if (cellIter == symbolTable.end() || cellIter->first != cellId) {
            auto n = graph.addNode(std::forward<decltype(value)>(value), pool);
            symbolTable.emplace_hint(cellIter, cellId, n);
        }
        else {
            graph.resetValueAt(cellIter->second, std::forward<decltype(value)>(value), pool);
        }
    };

    for (std::string line; std::getline(std::cin, line) && !line.empty();) {
        auto [cellId, cellExpr] = parser::parseLine(line);

        utils::match(cellExpr,
            [&resetNodeValue, cellId = cellId](parser::Number n) {
                resetNodeValue(cellId, Graph::Value{n});
            },
            [&getNode, &resetNodeValue, cellId = cellId](const parser::CellsSum& sum) {
                std::vector<Graph::NodeId> cs;
                for (auto sumMemberCellId : sum) {
                    cs.push_back(getNode(sumMemberCellId));
                }
                resetNodeValue(cellId, Graph::Expression{std::move(cs)});
            }
        );
    }

    for (auto& [cellId, nodeId] : symbolTable) {
        std::cout << cellId << '=' << ez::support::std23::to_underlying(graph.getValueAt(nodeId)) << '\n';
    }

    return EXIT_SUCCESS;
}

