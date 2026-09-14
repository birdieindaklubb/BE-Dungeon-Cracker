#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace pe16015::dungeon {

struct BlockPosition final {
    std::int32_t x{};
    std::int32_t y{};
    std::int32_t z{};
};

// A complete unmodified MonsterRoomFeature floor. Rows are north-to-south
// (increasing Z); characters in a row are west-to-east (increasing X).
// `M` represents mossy cobblestone, `C` ordinary cobblestone, and `?` an
// unknown tile. A wildcard retains its native floor-roll position but adds no
// constraint for that roll.
struct DungeonObservation final {
    BlockPosition spawner{};
    std::vector<std::string> floor_rows{};
    std::int32_t x_radius{};
    std::int32_t z_radius{};

    [[nodiscard]] std::int32_t population_chunk_x() const noexcept;
    [[nodiscard]] std::int32_t population_chunk_z() const noexcept;
    [[nodiscard]] std::size_t floor_roll_count() const noexcept;
    [[nodiscard]] std::size_t words_needed_after_origin() const noexcept;

    // stream_word is the MT word used for the selected attempt's X origin.
    [[nodiscard]] bool matches_stream(
        const std::vector<std::uint32_t>& stream,
        std::size_t stream_word) const noexcept;
    [[nodiscard]] bool matches_words(
        const std::uint32_t* stream,
        std::size_t stream_size,
        std::size_t stream_word) const noexcept;
    [[nodiscard]] std::string predicted_floor(
        const std::vector<std::uint32_t>& stream,
        std::size_t stream_word) const;
};

[[nodiscard]] DungeonObservation load_observation(
    BlockPosition spawner,
    const std::filesystem::path& floor_file);

[[nodiscard]] std::uint32_t population_seed(
    std::uint32_t world_seed,
    std::int32_t chunk_x,
    std::int32_t chunk_z) noexcept;

[[nodiscard]] std::vector<std::uint32_t> population_stream(
    std::uint32_t world_seed,
    std::int32_t chunk_x,
    std::int32_t chunk_z,
    std::size_t word_count);

} // namespace pe16015::dungeon
