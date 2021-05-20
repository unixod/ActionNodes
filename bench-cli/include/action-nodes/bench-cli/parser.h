#ifndef PARSER_H
#define PARSER_H

#include <vector>
#include <string_view>
#include <variant>
#include <string>
#include <cassert>
#include <tuple>

namespace anodes::bench_cli::parser {

using CellsSum = std::vector<std::string_view>;
using Number = std::int64_t;
using CellValue = std::variant<CellsSum, Number>;

namespace details {

std::pair<std::string_view, std::string_view>
consumeCellId(std::string_view line)
{
    assert(!line.empty());
    assert('A' <= line.front() && line.front() <= 'Z');

    auto lexemeEndPos = line.find_first_of(" +");
    assert(lexemeEndPos != 0);

    auto cellId = line.substr(0, lexemeEndPos-0);
    auto rest = line.substr(cellId.size());

    return std::pair{cellId, rest};
}

std::string_view consumeEq(std::string_view line)
{
    assert(!line.empty());
    auto pos = line.find(" = ");
    return line.substr(pos + 3);
}

CellsSum parseSum(std::string_view line)
{
    CellsSum sum;

    auto [cellId, rest] = consumeCellId(line);

    while (!rest.empty()) {
        assert(rest.front() == '+');
        rest.remove_prefix(1); // skip '+'
        sum.emplace_back(cellId);
        std::tie(cellId, rest) = consumeCellId(rest);
    }

    sum.emplace_back(cellId);

    return sum;
}

} // namespace details

std::pair<std::string_view, CellValue>
parseLine(std::string_view line)
{
    auto [cellId, rest] = details::consumeCellId(line);
    assert(!rest.empty());

    rest = details::consumeEq(rest);
    assert(!rest.empty());

    if (auto ch = rest.front(); 'A' <= ch  && ch <= 'Z') {
        return {cellId, details::parseSum(rest)};
    }
    else {
        return {cellId, std::stoll(std::string{rest})};
    }
}

} // namespace anodes::bench_cli::parser

#endif // PARSER_H
