#include "avx2.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pe16015::dungeon::avx2 {
namespace {

constexpr std::size_t lane_count = 8U;
constexpr std::size_t monster_room_attempt_count = 8U;
constexpr std::size_t words_per_attempt_before_placement = 5U;
constexpr std::size_t max_floor_roll_count = 81U;
constexpr std::size_t max_stream_word =
    (monster_room_attempt_count - 1U) * words_per_attempt_before_placement;
constexpr std::size_t max_stream_count =
    max_stream_word + words_per_attempt_before_placement + max_floor_roll_count;
constexpr std::size_t max_state_count = max_stream_count + 398U;

[[nodiscard]] __m256i seed_step(__m256i value, std::uint32_t index) noexcept {
    const __m256i multiplier = _mm256_set1_epi32(1'812'433'253U);
    const __m256i mixed = _mm256_xor_si256(value, _mm256_srli_epi32(value, 30));
    return _mm256_add_epi32(
        _mm256_mullo_epi32(mixed, multiplier),
        _mm256_set1_epi32(static_cast<std::int32_t>(index)));
}

[[nodiscard]] __m256i temper(__m256i value) noexcept {
    value = _mm256_xor_si256(value, _mm256_srli_epi32(value, 11));
    value = _mm256_xor_si256(value, _mm256_and_si256(
        _mm256_slli_epi32(value, 7), _mm256_set1_epi32(0x9d2c5680U)));
    value = _mm256_xor_si256(value, _mm256_and_si256(
        _mm256_slli_epi32(value, 15), _mm256_set1_epi32(0xefc60000U)));
    return _mm256_xor_si256(value, _mm256_srli_epi32(value, 18));
}

template <std::size_t OutputCount>
void initial_outputs(
    __m256i seed,
    std::array<__m256i, OutputCount>& output) noexcept {
    static_assert(OutputCount <= 227U);

    std::array<__m256i, OutputCount + 398U> state{};
    state[0] = seed;
    for (std::size_t index = 1; index < state.size(); ++index) {
        state[index] = seed_step(state[index - 1U], static_cast<std::uint32_t>(index));
    }

    const __m256i upper_mask = _mm256_set1_epi32(0x80000000U);
    const __m256i lower_mask = _mm256_set1_epi32(0x7fffffffU);
    const __m256i matrix_a = _mm256_set1_epi32(0x9908b0dfU);
    const __m256i one = _mm256_set1_epi32(1);
    for (std::size_t index = 0; index < output.size(); ++index) {
        const __m256i joined = _mm256_or_si256(
            _mm256_and_si256(state[index], upper_mask),
            _mm256_and_si256(state[index + 1U], lower_mask));
        const __m256i odd = _mm256_cmpeq_epi32(_mm256_and_si256(joined, one), one);
        const __m256i twisted = _mm256_xor_si256(
            _mm256_xor_si256(state[index + 397U], _mm256_srli_epi32(joined, 1)),
            _mm256_and_si256(odd, matrix_a));
        output[index] = temper(twisted);
    }
}

[[nodiscard]] __m256i population_seed(
    __m256i world_seed,
    std::int32_t chunk_x,
    std::int32_t chunk_z) noexcept {
    std::array<__m256i, 2U> initial{};
    initial_outputs(world_seed, initial);
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i x_multiplier = _mm256_add_epi32(
        _mm256_slli_epi32(_mm256_srli_epi32(initial[0], 2), 1), one);
    const __m256i z_multiplier = _mm256_add_epi32(
        _mm256_slli_epi32(_mm256_srli_epi32(initial[1], 2), 1), one);
    const __m256i x = _mm256_set1_epi32(chunk_x);
    const __m256i z = _mm256_set1_epi32(chunk_z);
    return _mm256_xor_si256(
        _mm256_add_epi32(_mm256_mullo_epi32(x_multiplier, x),
                         _mm256_mullo_epi32(z_multiplier, z)),
        world_seed);
}

[[nodiscard]] std::uint8_t matching_lanes(
    const std::array<std::uint32_t, lane_count>& seeds,
    const DungeonObservation& observation) noexcept {
    const __m256i world_seed = _mm256_setr_epi32(
        static_cast<std::int32_t>(seeds[0]), static_cast<std::int32_t>(seeds[1]),
        static_cast<std::int32_t>(seeds[2]), static_cast<std::int32_t>(seeds[3]),
        static_cast<std::int32_t>(seeds[4]), static_cast<std::int32_t>(seeds[5]),
        static_cast<std::int32_t>(seeds[6]), static_cast<std::int32_t>(seeds[7]));
    std::array<__m256i, max_stream_count> stream{};
    initial_outputs(population_seed(world_seed, observation.population_chunk_x(),
                                    observation.population_chunk_z()), stream);

    alignas(32) std::array<std::array<std::uint32_t, lane_count>, max_stream_count> words{};
    for (std::size_t index = 0; index < stream.size(); ++index) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(words[index].data()), stream[index]);
    }

    std::uint8_t matched{};
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
        std::array<std::uint32_t, max_stream_count> lane_stream{};
        for (std::size_t index = 0; index < lane_stream.size(); ++index) {
            lane_stream[index] = words[index][lane];
        }
        for (std::size_t attempt = 0; attempt < monster_room_attempt_count; ++attempt) {
            const std::size_t word = attempt * words_per_attempt_before_placement;
            if (observation.matches_words(lane_stream.data(), lane_stream.size(), word)) {
                matched = static_cast<std::uint8_t>(matched | (1U << lane));
                break;
            }
        }
    }
    return matched;
}

} // namespace

bool is_available() noexcept {
#if defined(_MSC_VER)
    int registers[4]{};
    __cpuidex(registers, 0, 0);
    if (registers[0] < 7) {
        return false;
    }
    __cpuidex(registers, 1, 0);
    constexpr int osxsave_bit = 1 << 27;
    constexpr int avx_bit = 1 << 28;
    if ((registers[2] & (osxsave_bit | avx_bit)) != (osxsave_bit | avx_bit)) {
        return false;
    }
    if ((_xgetbv(0) & 0x6U) != 0x6U) {
        return false;
    }
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

std::vector<DirectMatch> scan_unprefixed_attempts(
    ScanRange range,
    const DungeonObservation& observation) {
    std::vector<DirectMatch> result;
    const auto total = static_cast<std::int64_t>(range.count);

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<DirectMatch> local_result;
#pragma omp for schedule(static)
        for (std::int64_t base = 0; base < total; base += static_cast<std::int64_t>(lane_count)) {
            std::array<std::uint32_t, lane_count> seeds{};
            for (std::size_t lane = 0; lane < lane_count; ++lane) {
                seeds[lane] = static_cast<std::uint32_t>(
                    range.start + static_cast<std::uint64_t>(base) + lane);
            }
            const std::uint8_t lanes = matching_lanes(seeds, observation);
            for (std::size_t lane = 0; lane < lane_count; ++lane) {
                if ((lanes & (1U << lane)) != 0U
                    && static_cast<std::uint64_t>(base) + lane < range.count) {
                    const std::uint32_t seed = seeds[lane];
                    const auto stream = population_stream(
                        seed, observation.population_chunk_x(), observation.population_chunk_z(),
                        max_stream_count);
                    for (std::size_t attempt = 0; attempt < monster_room_attempt_count; ++attempt) {
                        const std::size_t word = attempt * words_per_attempt_before_placement;
                        if (observation.matches_stream(stream, word)) {
                            local_result.push_back({seed, static_cast<std::uint32_t>(word)});
                        }
                    }
                }
            }
        }
#pragma omp critical
        result.insert(result.end(), local_result.begin(), local_result.end());
    }
#else
    for (std::uint64_t base = 0; base < range.count; base += lane_count) {
        std::array<std::uint32_t, lane_count> seeds{};
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
            seeds[lane] = static_cast<std::uint32_t>(range.start + base + lane);
        }
        const std::uint8_t lanes = matching_lanes(seeds, observation);
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
            if ((lanes & (1U << lane)) != 0U && base + lane < range.count) {
                const std::uint32_t seed = seeds[lane];
                const auto stream = population_stream(
                    seed, observation.population_chunk_x(), observation.population_chunk_z(),
                    max_stream_count);
                for (std::size_t attempt = 0; attempt < monster_room_attempt_count; ++attempt) {
                    const std::size_t word = attempt * words_per_attempt_before_placement;
                    if (observation.matches_stream(stream, word)) {
                        result.push_back({seed, static_cast<std::uint32_t>(word)});
                    }
                }
            }
        }
    }
#endif
    return result;
}

} // namespace pe16015::dungeon::avx2
