#pragma once

#include "dungeon_observation.hpp"

#include <cstdint>
#include <vector>

namespace pe16015::dungeon::avx2 {

struct DirectMatch final {
    std::uint32_t seed{};
    std::uint32_t stream_word{};
};

struct ScanRange final {
    std::uint64_t start{};
    std::uint64_t count{};
};

// This translation unit is compiled with AVX2 separately.  Callers must test
// this before dispatching, so non-AVX2 processors continue to use the scalar
// implementation.
[[nodiscard]] bool is_available() noexcept;

// Tests the eight direct native MonsterRoomFeature attempts at words 0, 5, ... 35.
// This is an exact fast path only when no earlier structure consumes the
// population stream and every earlier room attempt fails before placement.
[[nodiscard]] std::vector<DirectMatch> scan_unprefixed_attempts(
    ScanRange range,
    const DungeonObservation& observation);

} // namespace pe16015::dungeon::avx2
