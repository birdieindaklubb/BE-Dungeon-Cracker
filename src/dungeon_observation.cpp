#include "dungeon_observation.hpp"

#include "mt19937.hpp"

#include <cctype>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace pe16015::dungeon {
namespace {

[[nodiscard]] std::int32_t floor_div_16(std::int32_t value) noexcept {
    if (value >= 0) {
        return value / 16;
    }
    return -static_cast<std::int32_t>((-static_cast<std::int64_t>(value) + 15) / 16);
}

// Native population chooses each X/Z origin in the owning chunk's block
// interval [8, 23].  Moving the observed origin back by eight before floor
// division therefore recovers that chunk both across zero and for negatives.
[[nodiscard]] std::int32_t owner_population_chunk_coordinate(std::int32_t origin) noexcept {
    return floor_div_16(origin - 8);
}

[[nodiscard]] bool is_floor_symbol(char value) noexcept {
    return value == 'M' || value == 'C' || value == '?';
}

[[nodiscard]] DungeonObservation parse_observation(
    BlockPosition spawner,
    std::istream& input,
    bool finish_at_first_blank_line) {
    DungeonObservation observation{};
    observation.spawner = spawner;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line.resize(comment);
        }

        std::string row;
        for (const unsigned char character : line) {
            if (std::isspace(character) != 0) {
                continue;
            }
            const char symbol = static_cast<char>(std::toupper(character));
            if (!is_floor_symbol(symbol)) {
                throw std::runtime_error(
                    "floor rows may contain only M, C, ?, whitespace, and comments");
            }
            row.push_back(symbol);
        }
        if (row.empty()) {
            if (finish_at_first_blank_line && !observation.floor_rows.empty()) {
                break;
            }
            continue;
        }
        observation.floor_rows.push_back(std::move(row));
    }

    if (observation.floor_rows.empty()) {
        throw std::runtime_error("floor observation contains no rows");
    }
    const std::size_t width = observation.floor_rows.front().size();
    for (const std::string& row : observation.floor_rows) {
        if (row.size() != width) {
            throw std::runtime_error("all floor rows must have the same width");
        }
    }
    if ((width != 7U && width != 9U)
        || (observation.floor_rows.size() != 7U && observation.floor_rows.size() != 9U)) {
        throw std::runtime_error(
            "a complete Bedrock Edition 1.6.0.15 monster-room floor must be 7 or 9 blocks on each axis");
    }

    observation.x_radius = static_cast<std::int32_t>((width - 3U) / 2U);
    observation.z_radius = static_cast<std::int32_t>((observation.floor_rows.size() - 3U) / 2U);
    return observation;
}

} // namespace

std::int32_t DungeonObservation::population_chunk_x() const noexcept {
    return owner_population_chunk_coordinate(spawner.x);
}

std::int32_t DungeonObservation::population_chunk_z() const noexcept {
    return owner_population_chunk_coordinate(spawner.z);
}

std::size_t DungeonObservation::floor_roll_count() const noexcept {
    return floor_rows.size() * floor_rows.front().size();
}

std::size_t DungeonObservation::words_needed_after_origin() const noexcept {
    return 5U + floor_roll_count();
}

bool DungeonObservation::matches_stream(
    const std::vector<std::uint32_t>& stream,
    std::size_t stream_word) const noexcept {
    return matches_words(stream.data(), stream.size(), stream_word);
}

bool DungeonObservation::matches_words(
    const std::uint32_t* stream,
    std::size_t stream_size,
    std::size_t stream_word) const noexcept {
    if (stream_word > stream_size
        || words_needed_after_origin() > stream_size - stream_word) {
        return false;
    }

    const std::int64_t base_x = static_cast<std::int64_t>(population_chunk_x()) * 16;
    const std::int64_t base_z = static_cast<std::int64_t>(population_chunk_z()) * 16;
    const std::int64_t expected_x = base_x + 8 + (stream[stream_word] & 15U);
    const std::int64_t expected_y = stream[stream_word + 1U] & 127U;
    const std::int64_t expected_z = base_z + 8 + (stream[stream_word + 2U] & 15U);
    if (expected_x != spawner.x || expected_y != spawner.y || expected_z != spawner.z
        || static_cast<std::int32_t>(stream[stream_word + 3U] & 1U) + 2 != x_radius
        || static_cast<std::int32_t>(stream[stream_word + 4U] & 1U) + 2 != z_radius) {
        return false;
    }

    std::size_t word = stream_word + 5U;
    for (std::size_t x = 0; x < floor_rows.front().size(); ++x) {
        for (std::size_t z = 0; z < floor_rows.size(); ++z) {
            const char observed = floor_rows[z][x];
            const bool mossy = (stream[word++] & 3U) != 0U;
            if (observed != '?' && (observed == 'M') != mossy) {
                return false;
            }
        }
    }
    return true;
}

std::string DungeonObservation::predicted_floor(
    const std::vector<std::uint32_t>& stream,
    std::size_t stream_word) const {
    if (stream_word > stream.size()
        || words_needed_after_origin() > stream.size() - stream_word) {
        throw std::out_of_range("population stream is too short for this observation");
    }

    std::vector<std::string> rows(
        floor_rows.size(), std::string(floor_rows.front().size(), 'C'));
    std::size_t word = stream_word + 5U;
    for (std::size_t x = 0; x < rows.front().size(); ++x) {
        for (std::size_t z = 0; z < rows.size(); ++z) {
            rows[z][x] = (stream[word++] & 3U) == 0U ? 'C' : 'M';
        }
    }

    std::string result;
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (row != 0U) {
            result.push_back('\n');
        }
        result += rows[row];
    }
    return result;
}

DungeonObservation load_observation(
    BlockPosition spawner,
    const std::filesystem::path& floor_file) {
    if (spawner.y < 0 || spawner.y > 127) {
        throw std::invalid_argument("spawner Y must be in 0..127");
    }

    if (floor_file == "-") {
        return parse_observation(spawner, std::cin, true);
    }

    std::ifstream input(floor_file);
    if (!input) {
        throw std::runtime_error("cannot open floor observation: " + floor_file.string());
    }
    return parse_observation(spawner, input, false);
}

std::uint32_t population_seed(
    std::uint32_t world_seed,
    std::int32_t chunk_x,
    std::int32_t chunk_z) noexcept {
    const auto seed_words = initial_mt19937_outputs(world_seed, 2U);
    const std::uint32_t x_multiplier = (seed_words[0] >> 2U) * 2U + 1U;
    const std::uint32_t z_multiplier = (seed_words[1] >> 2U) * 2U + 1U;
    return (static_cast<std::uint32_t>(chunk_z) * z_multiplier
        + static_cast<std::uint32_t>(chunk_x) * x_multiplier) ^ world_seed;
}

std::vector<std::uint32_t> population_stream(
    std::uint32_t world_seed,
    std::int32_t chunk_x,
    std::int32_t chunk_z,
    std::size_t word_count) {
    return initial_mt19937_outputs(
        population_seed(world_seed, chunk_x, chunk_z), word_count);
}

} // namespace pe16015::dungeon
