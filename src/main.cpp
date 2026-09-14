#include "dungeon_observation.hpp"
#include "mt19937.hpp"

#ifdef PE16015_HAVE_AVX2
#include "avx2.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <syncstream>
#include <thread>
#include <utility>
#include <vector>

#ifdef PE16015_HAS_OPENMP
#include <omp.h>
#endif

namespace {

using pe16015::dungeon::BlockPosition;
using pe16015::dungeon::DungeonObservation;

struct Options final {
    std::optional<BlockPosition> spawner{};
    std::optional<std::filesystem::path> floor_file{};
    std::optional<std::uint32_t> verify_seed{};
    std::optional<std::uint64_t> range_start{};
    std::optional<std::uint64_t> range_count{};
    std::optional<std::size_t> word_start{};
    std::optional<std::size_t> word_end{};
    std::optional<std::size_t> prefix_limit{};
    std::optional<int> threads{};
    bool native_attempts_only{};
    bool self_test{};
    bool help{};
};

struct Match final {
    std::uint32_t seed{};
    std::size_t stream_word{};

    [[nodiscard]] friend bool operator<(const Match& left, const Match& right) noexcept {
        return left.seed != right.seed
            ? left.seed < right.seed
            : left.stream_word < right.stream_word;
    }
};

[[nodiscard]] std::uint64_t parse_u64(std::string_view text, std::string_view label) {
    try {
        std::size_t consumed{};
        const std::uint64_t value = std::stoull(std::string(text), &consumed, 0);
        if (consumed != text.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + std::string(text));
    }
}

[[nodiscard]] std::uint32_t parse_seed(std::string_view text) {
    if (!text.empty() && text.front() == '-') {
        try {
            std::size_t consumed{};
            const std::int64_t signed_value = std::stoll(std::string(text), &consumed, 10);
            if (consumed != text.size()
                || signed_value < std::numeric_limits<std::int32_t>::min()) {
                throw std::invalid_argument("out of range");
            }
            return static_cast<std::uint32_t>(static_cast<std::int32_t>(signed_value));
        } catch (const std::exception&) {
            throw std::invalid_argument("invalid signed 32-bit seed: " + std::string(text));
        }
    }
    const std::uint64_t value = parse_u64(text, "seed");
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("seed is outside the unsigned 32-bit domain");
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::int32_t parse_i32(std::string_view text, std::string_view label) {
    try {
        std::size_t consumed{};
        const std::int64_t value = std::stoll(std::string(text), &consumed, 0);
        if (consumed != text.size()
            || value < std::numeric_limits<std::int32_t>::min()
            || value > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("out of range");
        }
        return static_cast<std::int32_t>(value);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + std::string(text));
    }
}

[[nodiscard]] std::size_t parse_size(std::string_view text, std::string_view label) {
    const std::uint64_t value = parse_u64(text, label);
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string(label) + " is too large");
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] std::string require_value(
    int argc,
    char* argv[],
    int& index,
    std::string_view option) {
    if (++index >= argc) {
        throw std::invalid_argument(std::string(option) + " requires a value");
    }
    return argv[index];
}

[[nodiscard]] Options parse_options(int argc, char* argv[]) {
    Options options{};
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else if (argument == "--self-test") {
            options.self_test = true;
        } else if (argument == "--spawner") {
            options.spawner = BlockPosition{
                parse_i32(require_value(argc, argv, index, argument), "spawner X"),
                parse_i32(require_value(argc, argv, index, argument), "spawner Y"),
                parse_i32(require_value(argc, argv, index, argument), "spawner Z"),
            };
        } else if (argument == "--floor") {
            options.floor_file = require_value(argc, argv, index, argument);
        } else if (argument == "--verify") {
            options.verify_seed = parse_seed(require_value(argc, argv, index, argument));
        } else if (argument == "--range-start") {
            options.range_start = parse_u64(require_value(argc, argv, index, argument), "range start");
        } else if (argument == "--count") {
            options.range_count = parse_u64(require_value(argc, argv, index, argument), "range count");
        } else if (argument == "--word-start") {
            options.word_start = parse_size(require_value(argc, argv, index, argument), "word start");
        } else if (argument == "--word-end") {
            options.word_end = parse_size(require_value(argc, argv, index, argument), "word end");
        } else if (argument == "--prefix-limit") {
            options.prefix_limit = parse_size(
                require_value(argc, argv, index, argument), "prefix limit");
        } else if (argument == "--threads") {
            const std::size_t count = parse_size(require_value(argc, argv, index, argument), "thread count");
            if (count == 0U || count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("thread count must be positive");
            }
            options.threads = static_cast<int>(count);
        } else if (argument == "--native-attempts") {
            options.native_attempts_only = true;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }
    return options;
}

void configure_interactive_observation(Options& options) {
    std::cout
        << "Bedrock Edition 1.6.0.15 monster-room cracker\n"
        << "Spawner block coordinate (X Y Z): ";
    std::string coordinate_line;
    if (!std::getline(std::cin, coordinate_line)) {
        throw std::runtime_error("no spawner coordinate was supplied");
    }
    std::istringstream coordinates(coordinate_line);
    std::string x;
    std::string y;
    std::string z;
    std::string extra;
    if (!(coordinates >> x >> y >> z) || (coordinates >> extra)) {
        throw std::invalid_argument("enter the spawner coordinate as exactly: X Y Z");
    }
    options.spawner = {
        parse_i32(x, "spawner X"),
        parse_i32(y, "spawner Y"),
        parse_i32(z, "spawner Z"),
    };
    options.floor_file = "-";
    std::cout
        << "Paste the complete 7x7, 7x9, 9x7, or 9x9 floor map now.\n"
        << "Use M for mossy, C for cobblestone, or ? for unknown; finish with a blank line.\n";
}

void print_usage() {
    std::cout
        << "Bedrock Edition 1.6.0.15 monster-room stream cracker (loot filtering is not included)\n\n"
        << "Verify a known seed:\n"
        << "  pe16015_dungeon_cracker --spawner X Y Z --floor FLOOR.txt --verify SEED\n\n"
        << "Crack the complete 32-bit world-seed domain (normal use):\n"
        << "  pe16015_dungeon_cracker --spawner X Y Z --floor FLOOR.txt\n"
        << "  Use --floor - to paste the floor, ending with one blank line.\n\n"
        << "Advanced: scan only an explicit unsigned-32-bit seed range:\n"
        << "  pe16015_dungeon_cracker --spawner X Y Z --floor FLOOR.txt\n"
        << "    --range-start START --count COUNT [--threads N]\n\n"
        << "Normal use automatically searches candidate stream words 0 through 1023.\n"
        << "Word N is the MT word used for the selected attempt's X coordinate.\n"
        << "Advanced control: --prefix-limit N, or --word-start N --word-end N.\n"
        << "Fast special case: --native-attempts checks only the eight direct\n"
        << "MonsterRoom calls, before any shared structure prefix.\n"
        << "Use --self-test to validate this executable.\n";
}

[[nodiscard]] DungeonObservation require_observation(const Options& options) {
    if (!options.spawner.has_value() || !options.floor_file.has_value()) {
        throw std::invalid_argument("--spawner and --floor are required");
    }
    return pe16015::dungeon::load_observation(*options.spawner, *options.floor_file);
}

struct SeedRange final {
    std::uint64_t start{};
    std::uint64_t count{};
    bool is_complete_domain{};
};

[[nodiscard]] SeedRange resolve_seed_range(const Options& options) {
    constexpr std::uint64_t seed_space = std::uint64_t{1} << 32U;
    const bool has_range_start = options.range_start.has_value();
    const bool has_range_count = options.range_count.has_value();
    if (has_range_start != has_range_count) {
        throw std::invalid_argument("--range-start and --count must be supplied together");
    }
    if (!has_range_start) {
        return {0U, seed_space, true};
    }

    const std::uint64_t start = *options.range_start;
    const std::uint64_t count = *options.range_count;
    if (count == 0U || start >= seed_space || count > seed_space - start) {
        throw std::invalid_argument("scan range must stay within the unsigned 32-bit seed domain");
    }
    return {start, count, start == 0U && count == seed_space};
}

[[nodiscard]] std::pair<std::size_t, std::size_t> resolve_word_range(const Options& options) {
    constexpr std::size_t default_prefix_limit = 1023U;
    const bool has_word_start = options.word_start.has_value();
    const bool has_word_end = options.word_end.has_value();
    if (has_word_start != has_word_end) {
        throw std::invalid_argument("--word-start and --word-end must be supplied together");
    }
    if (has_word_start) {
        if (options.prefix_limit.has_value()) {
            throw std::invalid_argument("--prefix-limit cannot be combined with --word-start/--word-end");
        }
        if (*options.word_start > *options.word_end) {
            throw std::invalid_argument("word start must not exceed word end");
        }
        return {*options.word_start, *options.word_end};
    }
    return {0U, options.prefix_limit.value_or(default_prefix_limit)};
}

[[nodiscard]] std::vector<Match> find_matches(
    std::uint32_t seed,
    const DungeonObservation& observation,
    std::size_t word_start,
    std::size_t word_end) {
    if (word_end > std::numeric_limits<std::size_t>::max()
            - observation.words_needed_after_origin()) {
        throw std::invalid_argument("word range is too large");
    }
    const auto stream = pe16015::dungeon::population_stream(
        seed,
        observation.population_chunk_x(),
        observation.population_chunk_z(),
        word_end + observation.words_needed_after_origin());

    std::vector<Match> matches;
    for (std::size_t word = word_start; word <= word_end; ++word) {
        if (observation.matches_stream(stream, word)) {
            matches.push_back({seed, word});
        }
    }
    return matches;
}

[[nodiscard]] std::vector<Match> find_unprefixed_attempt_matches(
    std::uint32_t seed,
    const DungeonObservation& observation) {
    constexpr std::size_t attempt_count = 8U;
    constexpr std::size_t words_before_placement = 5U;
    constexpr std::size_t maximum_floor_rolls = 81U;
    constexpr std::size_t stream_size =
        (attempt_count - 1U) * words_before_placement + words_before_placement
        + maximum_floor_rolls;

    const auto stream = pe16015::dungeon::population_stream(
        seed, observation.population_chunk_x(), observation.population_chunk_z(), stream_size);
    std::vector<Match> matches;
    for (std::size_t attempt = 0; attempt < attempt_count; ++attempt) {
        const std::size_t word = attempt * words_before_placement;
        if (observation.matches_stream(stream, word)) {
            matches.push_back({seed, word});
        }
    }
    return matches;
}

[[nodiscard]] std::vector<Match> scan_unprefixed_attempts(
    SeedRange range,
    const DungeonObservation& observation) {
#ifdef PE16015_HAVE_AVX2
    if (pe16015::dungeon::avx2::is_available()) {
        const auto direct_matches = pe16015::dungeon::avx2::scan_unprefixed_attempts(
            {range.start, range.count}, observation);
        std::vector<Match> matches;
        matches.reserve(direct_matches.size());
        for (const auto& direct_match : direct_matches) {
            matches.push_back({direct_match.seed, direct_match.stream_word});
        }
        return matches;
    }
#endif

    std::vector<Match> matches;
#ifdef PE16015_HAS_OPENMP
#pragma omp parallel
    {
        std::vector<Match> local_matches;
#pragma omp for schedule(static)
        for (std::int64_t offset = 0; offset < static_cast<std::int64_t>(range.count); ++offset) {
            const std::uint32_t seed = static_cast<std::uint32_t>(
                range.start + static_cast<std::uint64_t>(offset));
            auto found = find_unprefixed_attempt_matches(seed, observation);
            local_matches.insert(local_matches.end(), found.begin(), found.end());
        }
#pragma omp critical
        matches.insert(matches.end(), local_matches.begin(), local_matches.end());
    }
#else
    for (std::uint64_t offset = 0; offset < range.count; ++offset) {
        const std::uint32_t seed = static_cast<std::uint32_t>(range.start + offset);
        auto found = find_unprefixed_attempt_matches(seed, observation);
        matches.insert(matches.end(), found.begin(), found.end());
    }
#endif
    return matches;
}

[[nodiscard]] std::filesystem::path result_file_path() {
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count() % 1000;
    const std::time_t local_time = std::chrono::system_clock::to_time_t(now);
    std::tm calendar{};
#ifdef _WIN32
    localtime_s(&calendar, &local_time);
#else
    localtime_r(&local_time, &calendar);
#endif

    std::ostringstream stem;
    stem << "pe16015-dungeon-" << std::put_time(&calendar, "%y%m%d-%H%M%S-")
         << std::setfill('0') << std::setw(3) << milliseconds;
    const std::filesystem::path directory = "results";
    std::filesystem::create_directories(directory);
    for (unsigned int suffix = 0;; ++suffix) {
        const std::filesystem::path candidate = directory
            / (stem.str() + (suffix == 0U ? "" : "-" + std::to_string(suffix)) + ".txt");
        if (!std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
}

void write_matches(const std::vector<Match>& matches, const std::filesystem::path& output) {
    std::ofstream file(output);
    if (!file) {
        throw std::runtime_error("cannot create result file: " + output.string());
    }
    for (const Match& match : matches) {
        file << "unsigned=" << match.seed
             << " signed=" << static_cast<std::int32_t>(match.seed)
             << " hex=0x" << std::hex << std::setw(8) << std::setfill('0') << match.seed
             << std::dec << std::setfill(' ')
             << " stream_word=" << match.stream_word << '\n';
    }
}

[[nodiscard]] bool self_test() {
    using pe16015::dungeon::Mt19937;

    Mt19937 standard_mt(5489U);
    if (standard_mt.next_u32() != 3499211612U) {
        std::cerr << "FAIL: MT19937 reference output\n";
        return false;
    }
    Mt19937 full_prefix_mt(0x12345678U);
    const auto reduced_prefix = pe16015::dungeon::initial_mt19937_outputs(
        0x12345678U, 227U);
    for (const std::uint32_t value : reduced_prefix) {
        if (full_prefix_mt.next_u32() != value) {
            std::cerr << "FAIL: reduced MT prefix differs from full MT\n";
            return false;
        }
    }

    DungeonObservation observation{};
    observation.spawner = {10, 7, 13};
    observation.x_radius = 2;
    observation.z_radius = 2;
    observation.floor_rows = {
        "MCMCMCM",
        "CMCMCMC",
        "MCMCMCM",
        "CMCMCMC",
        "MCMCMCM",
        "CMCMCMC",
        "MCMCMCM",
    };

    constexpr std::uint32_t world_seed = 5489U;
    const auto population = pe16015::dungeon::population_stream(world_seed, 7, -3, 1U);
    if (population.front() != 0x5a1d027bU) {
        std::cerr << "FAIL: population-seed derivation\n";
        return false;
    }

    std::vector<std::uint32_t> synthetic_stream(59U, 0U);
    synthetic_stream[5] = 2U; // X = chunk 0 base + 8 + 2
    synthetic_stream[6] = 7U;
    synthetic_stream[7] = 5U; // Z = chunk 0 base + 8 + 5
    synthetic_stream[8] = 0U; // X radius 2
    synthetic_stream[9] = 0U; // Z radius 2
    std::size_t word = 10U;
    for (std::size_t x = 0; x < 7U; ++x) {
        for (std::size_t z = 0; z < 7U; ++z) {
            synthetic_stream[word++] = observation.floor_rows[z][x] == 'M' ? 1U : 0U;
        }
    }
    if (!observation.matches_stream(synthetic_stream, 5U)) {
        std::cerr << "FAIL: spawner/radius/floor constraint ordering\n";
        return false;
    }
    DungeonObservation wildcard_observation = observation;
    wildcard_observation.floor_rows[0][0] = '?';
    synthetic_stream[10] = 0U; // Deliberately contradict the original M cell.
    if (!wildcard_observation.matches_stream(synthetic_stream, 5U)) {
        std::cerr << "FAIL: unknown floor tile must retain but not constrain its RNG roll\n";
        return false;
    }

    // The room-attempt X/Z offsets are 8..23, not 0..15.  These edges prove
    // the observation converts a spawner coordinate back to the *owner*
    // population chunk correctly, including floor division below zero.
    DungeonObservation chunk_edges{};
    chunk_edges.spawner = {8, 0, 23};
    if (chunk_edges.population_chunk_x() != 0 || chunk_edges.population_chunk_z() != 0) {
        std::cerr << "FAIL: positive owner population chunk\n";
        return false;
    }
    chunk_edges.spawner = {7, 0, -9};
    if (chunk_edges.population_chunk_x() != -1 || chunk_edges.population_chunk_z() != -2) {
        std::cerr << "FAIL: negative owner population chunk\n";
        return false;
    }

#ifdef PE16015_HAVE_AVX2
    if (pe16015::dungeon::avx2::is_available()) {
        constexpr std::uint32_t avx_seed = 5489U;
        const auto avx_stream = pe16015::dungeon::population_stream(avx_seed, 0, 0, 86U);
        DungeonObservation avx_observation{};
        avx_observation.spawner = {
            static_cast<std::int32_t>(8U + (avx_stream[0] & 15U)),
            static_cast<std::int32_t>(avx_stream[1] & 127U),
            static_cast<std::int32_t>(8U + (avx_stream[2] & 15U)),
        };
        avx_observation.x_radius = static_cast<std::int32_t>(avx_stream[3] & 1U) + 2;
        avx_observation.z_radius = static_cast<std::int32_t>(avx_stream[4] & 1U) + 2;
        avx_observation.floor_rows.assign(
            static_cast<std::size_t>(avx_observation.z_radius * 2 + 3),
            std::string(static_cast<std::size_t>(avx_observation.x_radius * 2 + 3), 'C'));
        std::size_t avx_word = 5U;
        for (std::size_t x = 0; x < avx_observation.floor_rows.front().size(); ++x) {
            for (std::size_t z = 0; z < avx_observation.floor_rows.size(); ++z) {
                avx_observation.floor_rows[z][x] = (avx_stream[avx_word++] & 3U) == 0U
                    ? 'C'
                    : 'M';
            }
        }
        const auto avx_matches = pe16015::dungeon::avx2::scan_unprefixed_attempts(
            {avx_seed, 1U}, avx_observation);
        if (avx_matches.size() != 1U || avx_matches.front().seed != avx_seed
            || avx_matches.front().stream_word != 0U) {
            std::cerr << "FAIL: AVX2 native-attempt scanner\n";
            return false;
        }
    }
#endif

    std::cout << "Self-test passed: exact MT, reduced MT prefix, population seed, and floor constraints.\n";
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        Options options = parse_options(argc, argv);
        if (options.help) {
            print_usage();
            return 0;
        }
        if (argc == 1) {
            configure_interactive_observation(options);
        }
        if (options.self_test) {
            return self_test() ? 0 : 1;
        }

        const DungeonObservation observation = require_observation(options);
        if (options.native_attempts_only
            && (options.word_start.has_value() || options.word_end.has_value()
                || options.prefix_limit.has_value())) {
            throw std::invalid_argument(
                "--native-attempts cannot be combined with prefix-window controls");
        }
        const auto [word_start, word_end] = options.native_attempts_only
            ? std::pair<std::size_t, std::size_t>{0U, 35U}
            : resolve_word_range(options);

        std::cout << "Population chunk: " << observation.population_chunk_x() << ','
                  << observation.population_chunk_z() << "\n"
                  << "Room radii: x=" << observation.x_radius
                  << " z=" << observation.z_radius
                  << "; observed floor rolls=" << observation.floor_roll_count() << "\n"
                  << "Candidate stream words: " << word_start << ".." << word_end << "\n";
        if (options.native_attempts_only) {
            std::cout << "Mode: eight unprefixed MonsterRoom attempts (words 0, 5, ..., 35).\n";
        }

        if (options.verify_seed.has_value()) {
            if (options.range_start.has_value() || options.range_count.has_value()) {
                throw std::invalid_argument("--verify cannot be combined with a scan range");
            }
            const auto matches = options.native_attempts_only
                ? find_unprefixed_attempt_matches(*options.verify_seed, observation)
                : find_matches(*options.verify_seed, observation, word_start, word_end);
            std::cout << "Seed unsigned=" << *options.verify_seed
                      << " signed=" << static_cast<std::int32_t>(*options.verify_seed)
                      << " population_seed=" << pe16015::dungeon::population_seed(
                             *options.verify_seed,
                             observation.population_chunk_x(),
                             observation.population_chunk_z())
                      << "\n";
            if (matches.empty()) {
                std::cout << "No match in stream words " << word_start << ".." << word_end << ".\n";
                return 1;
            }
            for (const Match& match : matches) {
                std::cout << "MATCH stream_word=" << match.stream_word << "\n";
            }
            return 0;
        }

        const SeedRange range = resolve_seed_range(options);
        const std::uint64_t range_start = range.start;
        const std::uint64_t range_count = range.count;
        if (range.is_complete_domain) {
            std::cout << "Scanning all 4,294,967,296 unsigned-32-bit seed values.\n";
        } else {
            std::cout << "Scanning " << range_count << " seed values starting at "
                      << range_start << ".\n";
        }

#ifdef PE16015_HAS_OPENMP
        if (options.threads.has_value()) {
            omp_set_num_threads(*options.threads);
        }
#else
        if (options.threads.has_value()) {
            std::cerr << "Warning: this build has no OpenMP support; --threads is ignored.\n";
        }
#endif

        if (options.native_attempts_only) {
            std::cout << "Scanning the selected seed domain with native direct-attempt constraints.\n";
            auto matches = scan_unprefixed_attempts(range, observation);
            std::sort(matches.begin(), matches.end());
            const std::filesystem::path output = result_file_path();
            write_matches(matches, output);
            std::cout << "Matches: " << matches.size() << "\n"
                      << "Saved: " << output.string() << "\n";
            for (const Match& match : matches) {
                std::cout << "unsigned=" << match.seed
                          << " signed=" << static_cast<std::int32_t>(match.seed)
                          << " stream_word=" << match.stream_word << "\n";
            }
            return 0;
        }

        std::atomic<std::uint64_t> checked{0U};
        std::atomic<bool> scan_finished{false};
        std::mutex progress_mutex;
        std::condition_variable progress_wake;
        const auto scan_started = std::chrono::steady_clock::now();
        std::jthread progress_reporter([&checked, &scan_finished, &progress_mutex,
                                        &progress_wake, range_count, scan_started] {
            using namespace std::chrono_literals;
            std::unique_lock lock(progress_mutex);
            while (!progress_wake.wait_for(lock, 5s, [&scan_finished] {
                return scan_finished.load(std::memory_order_relaxed);
            })) {
                const std::uint64_t completed = checked.load(std::memory_order_relaxed);
                if (completed == 0U) {
                    continue;
                }
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - scan_started).count();
                const double rate = static_cast<double>(completed) / elapsed;
                const double percent = 100.0 * static_cast<double>(completed)
                    / static_cast<double>(range_count);
                std::osyncstream(std::cout)
                    << "Checked " << completed << "/" << range_count << " ("
                    << std::fixed << std::setprecision(2) << percent << "%, "
                    << std::setprecision(0) << rate << " seeds/s)\n";
            }
        });

        std::vector<Match> matches;
#ifdef PE16015_HAS_OPENMP
#pragma omp parallel
        {
            std::vector<Match> local_matches;
            std::uint64_t local_checked{};
#pragma omp for schedule(static)
            for (std::int64_t offset = 0; offset < static_cast<std::int64_t>(range_count); ++offset) {
                const std::uint32_t seed = static_cast<std::uint32_t>(
                    range_start + static_cast<std::uint64_t>(offset));
                auto found = find_matches(seed, observation, word_start, word_end);
                local_matches.insert(local_matches.end(), found.begin(), found.end());
                ++local_checked;
                if ((local_checked & ((std::uint64_t{1} << 20U) - 1U)) == 0U) {
                    checked.fetch_add(std::uint64_t{1} << 20U, std::memory_order_relaxed);
                }
            }
            checked.fetch_add(local_checked & ((std::uint64_t{1} << 20U) - 1U),
                              std::memory_order_relaxed);
#pragma omp critical
            matches.insert(matches.end(), local_matches.begin(), local_matches.end());
        }
#else
        for (std::uint64_t offset = 0; offset < range_count; ++offset) {
            const std::uint32_t seed = static_cast<std::uint32_t>(range_start + offset);
            auto found = find_matches(seed, observation, word_start, word_end);
            matches.insert(matches.end(), found.begin(), found.end());
            if (((offset + 1U) & ((std::uint64_t{1} << 20U) - 1U)) == 0U) {
                checked.fetch_add(std::uint64_t{1} << 20U, std::memory_order_relaxed);
            }
        }
        checked.fetch_add(range_count & ((std::uint64_t{1} << 20U) - 1U),
                          std::memory_order_relaxed);
#endif
        scan_finished.store(true, std::memory_order_relaxed);
        progress_wake.notify_one();
        progress_reporter.join();

        std::sort(matches.begin(), matches.end());
        const std::filesystem::path output = result_file_path();
        write_matches(matches, output);
        std::cout << "Matches: " << matches.size() << "\n"
                  << "Saved: " << output.string() << "\n";
        for (const Match& match : matches) {
            std::cout << "unsigned=" << match.seed
                      << " signed=" << static_cast<std::int32_t>(match.seed)
                      << " stream_word=" << match.stream_word << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\nUse --help for usage.\n";
        return 2;
    }
}
