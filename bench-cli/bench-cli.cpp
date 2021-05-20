#include <ez/support/c++23-features.h>
#include "action-nodes/graph.h"
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


using anodes::CalcGraph;
int main()
{
    namespace parser = anodes::bench_cli::parser;
    namespace utils = anodes::bench_cli::utils;

    anodes::ThreadPool pool;
    CalcGraph graph;

    using CellId = std::string;
    std::map<CellId, CalcGraph::NodeId, std::less<>> symbolTable;

    auto getNode = [&graph, &symbolTable, &pool](auto cellId) {
        auto i = symbolTable.lower_bound(cellId);
        if (i == symbolTable.end() || i->first != cellId) {
            auto n = graph.addNode(CalcGraph::Value{}, pool);
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
        std::cout << cellId << '=' << ez::std23::to_underlying(graph.getValueAt(nodeId)) << '\n';
    }

    return EXIT_SUCCESS;
}

