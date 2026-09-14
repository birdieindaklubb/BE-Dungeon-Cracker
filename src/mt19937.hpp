#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace pe16015::dungeon {

[[nodiscard]] inline std::uint32_t temper_mt19937(std::uint32_t value) noexcept {
    value ^= value >> 11U;
    value ^= (value << 7U) & 0x9d2c5680U;
    value ^= (value << 15U) & 0xefc60000U;
    return value ^ (value >> 18U);
}

// Exact 32-bit MT19937 used by the Bedrock Edition 1.6.0.15 target's Random.
class Mt19937 final {
public:
    explicit Mt19937(std::uint32_t seed) noexcept {
        set_seed(seed);
    }

    void set_seed(std::uint32_t seed) noexcept {
        state_[0] = seed;
        for (std::size_t index = 1; index < state_.size(); ++index) {
            const std::uint32_t previous = state_[index - 1];
            state_[index] = 1812433253U * (previous ^ (previous >> 30U))
                + static_cast<std::uint32_t>(index);
        }
        index_ = state_.size();
    }

    [[nodiscard]] std::uint32_t next_u32() noexcept {
        if (index_ == state_.size()) {
            twist();
        }

        return temper_mt19937(state_[index_++]);
    }

private:
    void twist() noexcept {
        constexpr std::uint32_t upper_mask = 0x80000000U;
        constexpr std::uint32_t lower_mask = 0x7fffffffU;
        constexpr std::uint32_t matrix_a = 0x9908b0dfU;

        for (std::size_t index = 0; index < state_.size(); ++index) {
            const std::uint32_t merged = (state_[index] & upper_mask)
                | (state_[(index + 1U) % state_.size()] & lower_mask);
            state_[index] = state_[(index + 397U) % state_.size()]
                ^ (merged >> 1U)
                ^ ((merged & 1U) == 0U ? 0U : matrix_a);
        }
        index_ = 0;
    }

    std::array<std::uint32_t, 624> state_{};
    std::size_t index_{};
};

// The first 227 outputs after MT19937's initial twist read only old state
// entries 0 through 623.  For a short population-prefix scan this avoids
// constructing unused state entries or performing a full twist, while using
// the target's exact seed recurrence, twist expression, and tempering.
[[nodiscard]] inline std::vector<std::uint32_t> initial_mt19937_outputs(
    std::uint32_t seed,
    std::size_t count) {
    constexpr std::size_t prefix_limit = 227U;
    if (count > prefix_limit) {
        Mt19937 random(seed);
        std::vector<std::uint32_t> result;
        result.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            result.push_back(random.next_u32());
        }
        return result;
    }

    if (count == 0U) {
        return {};
    }

    std::vector<std::uint32_t> state(count + 397U);
    state[0] = seed;
    for (std::size_t index = 1; index < state.size(); ++index) {
        const std::uint32_t previous = state[index - 1U];
        state[index] = 1812433253U * (previous ^ (previous >> 30U))
            + static_cast<std::uint32_t>(index);
    }

    std::vector<std::uint32_t> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::uint32_t merged = (state[index] & 0x80000000U)
            | (state[index + 1U] & 0x7fffffffU);
        const std::uint32_t twisted = state[index + 397U]
            ^ (merged >> 1U)
            ^ ((merged & 1U) == 0U ? 0U : 0x9908b0dfU);
        result.push_back(temper_mt19937(twisted));
    }
    return result;
}

} // namespace pe16015::dungeon
