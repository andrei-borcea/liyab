// Tests for the zero-copy loader: file mapping, paging hints (madvise),
// GGUF parsing and validation, layer-window prefetching and the NEON INT4
// unpacking kernel.
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "liyab/mmap_loader.h"
#include "liyab/triple_buffer_loader.h"
#include "test_model.h"
#include "test_util.h"

using namespace liyab;

namespace {

std::string temp_path(const char* name) { return test::temp_dir() + "/liyab_" + std::to_string(getpid()) + "_" + name; }

void write_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()),
                                                static_cast<std::streamsize>(bytes.size()));
}

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Reference for unpack_int4_to_int8_neon: element 2i = low nibble of byte i,
// element 2i+1 = high nibble, both offset-binary (nibble - 8).
std::vector<int8_t> unpack_reference(const std::vector<uint8_t>& packed, size_t n) {
    std::vector<int8_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t b = packed[i / 2];
        out[i] = static_cast<int8_t>(((i % 2 == 0) ? (b & 0x0F) : (b >> 4)) - 8);
    }
    return out;
}

}  // namespace

TEST_CASE("MappedFile maps a file read-only and returns its bytes") {
    const std::string path = temp_path("raw.bin");
    std::vector<uint8_t> bytes(3 * 65536 + 123);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>(i * 31 + 7);
    write_bytes(path, bytes);

    auto file = MappedFile::open(path);
    REQUIRE(file.has_value());
    CHECK(file.value()->size() == bytes.size());
    CHECK(std::memcmp(file.value()->data(), bytes.data(), bytes.size()) == 0);
    std::remove(path.c_str());
}

TEST_CASE("MappedFile reports missing and empty files") {
    auto missing = MappedFile::open(temp_path("does-not-exist.bin"));
    CHECK(!missing.has_value());
    CHECK(missing.status().code() == ErrorCode::IoError);

    const std::string empty = temp_path("empty.bin");
    write_bytes(empty, {});
    auto e = MappedFile::open(empty);
    CHECK(!e.has_value());
    std::remove(empty.c_str());
}

TEST_CASE("madvise WILLNEED / DONTNEED / SEQUENTIAL are accepted and DONTNEED keeps data intact") {
    const std::string path = temp_path("advise.bin");
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    std::vector<uint8_t> bytes(8 * page + 17);
    std::mt19937 rng(42);
    for (auto& b : bytes) b = static_cast<uint8_t>(rng());
    write_bytes(path, bytes);

    auto opened = MappedFile::open(path);
    REQUIRE(opened.has_value());
    const MappedFile& file = *opened.value();

    CHECK(file.advise_sequential());
    // Unaligned ranges are widened to whole pages internally.
    CHECK(file.advise_willneed(page + 5, 3 * page));
    file.touch(page + 5, 3 * page);
    CHECK(std::memcmp(file.data() + page, bytes.data() + page, 4 * page) == 0);

    // Release processed pages, then read them again: they must re-fault from
    // the file with identical contents (read-only mapping, nothing is lost).
    CHECK(file.advise_dontneed(0, 6 * page));
    CHECK(std::memcmp(file.data(), bytes.data(), bytes.size()) == 0);

    // Ranges are clamped to the file; out-of-range or empty ranges are rejected.
    CHECK(file.advise_willneed(7 * page, 100 * page));
    CHECK(!file.advise_willneed(bytes.size() + page, page));
    CHECK(!file.advise_dontneed(0, 0));
    std::remove(path.c_str());
}

TEST_CASE("unpack_int4_to_int8_neon matches the scalar definition") {
    // Hand-checked vector: 0x80 -> (0-8, 8-8) = (-8, 0); 0x7F -> (15-8, 7-8) = (7, -1).
    const uint8_t packed[] = {0x80, 0x7F, 0x00, 0xFF};
    int8_t out[8] = {};
    MmapLoader::unpack_int4_to_int8_neon(packed, out, 8);
    const int8_t expected[] = {-8, 0, 7, -1, -8, -8, 7, 7};
    CHECK(std::memcmp(out, expected, sizeof expected) == 0);

    // Lengths around the 32-element NEON stride, including odd tails.
    std::mt19937 rng(7);
    for (const size_t n : {size_t{1}, size_t{2}, size_t{31}, size_t{32}, size_t{33}, size_t{64}, size_t{95},
                           size_t{1000}, size_t{4097}}) {
        std::vector<uint8_t> src((n + 1) / 2);
        for (auto& b : src) b = static_cast<uint8_t>(rng());
        std::vector<int8_t> dst(n + 16, 0x55);  // guard bytes must stay untouched
        MmapLoader::unpack_int4_to_int8_neon(src.data(), dst.data(), n);
        const auto ref = unpack_reference(src, n);
        CHECK(std::memcmp(dst.data(), ref.data(), n) == 0);
        CHECK(std::all_of(dst.begin() + static_cast<std::ptrdiff_t>(n), dst.end(), [](int8_t v) { return v == 0x55; }));
        CHECK(std::all_of(dst.begin(), dst.begin() + static_cast<std::ptrdiff_t>(n),
                          [](int8_t v) { return v >= -8 && v <= 7; }));
    }
}

TEST_CASE("MmapLoader parses GGUF metadata and tensors without copying") {
    const std::string path = temp_path("tiny.gguf");
    test::write_tiny_model(path);
    auto loader = MmapLoader::open(path);
    REQUIRE(loader.has_value());
    const MmapLoader& m = *loader.value();

    CHECK(m.gguf_version() == 3);
    CHECK(m.get_string("general.architecture") == std::optional<std::string_view>("llama"));
    CHECK(m.get_int("llama.block_count") == std::optional<int64_t>(2));
    CHECK(m.get_float("llama.rope.freq_base").value_or(0) == 10000.0);
    const GgufValue* tokens = m.metadata("tokenizer.ggml.tokens");
    REQUIRE(tokens != nullptr);
    CHECK(tokens->strings.size() == test::tiny_vocab().size());
    CHECK(tokens->strings[1] == "<s>");

    const TensorView* q = m.tensor("blk.0.attn_q.weight");
    REQUIRE(q != nullptr);
    CHECK(q->type == DType::Q8_0);
    CHECK(q->cols() == 128 && q->rows() == 128);
    // Zero-copy: the view points inside the mapping, 32-byte aligned.
    CHECK(q->data >= m.file().data() && q->data + q->nbytes <= m.file().data() + m.file_size());
    CHECK(q->file_offset % 32 == 0);
    CHECK(m.tensor("blk.0.ffn_up.weight")->type == DType::Q4_0);  // mixed precision preserved
    CHECK(m.tensor("missing.weight") == nullptr);
    std::remove(path.c_str());
}

TEST_CASE("MmapLoader rejects malformed files") {
    const std::string good = temp_path("good.gguf");
    test::write_tiny_model(good);
    const std::vector<uint8_t> bytes = read_bytes(good);
    const std::string bad = temp_path("bad.gguf");

    auto expect_invalid = [&](std::vector<uint8_t> data, const char* what) {
        write_bytes(bad, data);
        auto r = MmapLoader::open(bad);
        if (r.has_value()) std::printf("  accepted malformed file: %s\n", what);
        CHECK(!r.has_value());
    };

    std::vector<uint8_t> magic = bytes;
    magic[0] = 'X';
    expect_invalid(magic, "bad magic");

    std::vector<uint8_t> version = bytes;
    version[4] = 99;
    expect_invalid(version, "bad version");

    expect_invalid(std::vector<uint8_t>(bytes.begin(), bytes.begin() + 200), "truncated metadata");
    expect_invalid(std::vector<uint8_t>(bytes.begin(), bytes.end() - 1000), "truncated tensor data");

    std::vector<uint8_t> huge_count = bytes;
    const uint64_t absurd = uint64_t{1} << 60;
    std::memcpy(huge_count.data() + 8, &absurd, sizeof absurd);  // n_tensors
    expect_invalid(huge_count, "absurd tensor count");

    std::remove(good.c_str());
    std::remove(bad.c_str());
}

TEST_CASE("Streaming prefetcher pages in the next layers and releases old ones") {
    const std::string path = temp_path("stream.gguf");
    test::TinyModelSpec spec;
    spec.n_layers = 4;
    test::write_tiny_model(path, spec);
    LoaderOptions options;
    options.streaming = true;
    auto loader = MmapLoader::open(path, options);
    REQUIRE(loader.has_value());
    MmapLoader& m = *loader.value();
    CHECK(m.streaming());

    m.configure_layers(spec.n_layers);
    for (int32_t token = 0; token < 2; ++token) {
        for (int32_t layer = 0; layer <= spec.n_layers; ++layer) m.begin_layer(layer);  // slot n = output head
    }
    m.wait_prefetch_idle();
    CHECK(m.prefetched_bytes() > 0);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Triple-buffered loader
// ---------------------------------------------------------------------------
namespace {

struct BlockFile {
    std::string path;
    std::vector<uint8_t> bytes;
    std::vector<TripleBufferLoader::Item> items;
};

// File of `n` blocks of uneven sizes at unaligned offsets.
BlockFile make_block_file(int n) {
    BlockFile f;
    f.path = temp_path("blocks.bin");
    std::mt19937 rng(99);
    uint64_t offset = 123;
    f.bytes.resize(123);
    for (int i = 0; i < n; ++i) {
        const uint64_t len = 20000 + 7919 * static_cast<uint64_t>(i);
        f.items.push_back({offset, len});
        for (uint64_t k = 0; k < len; ++k) f.bytes.push_back(static_cast<uint8_t>(rng()));
        offset += len;
    }
    write_bytes(f.path, f.bytes);
    return f;
}

TripleBufferLoader::FetchFn pread_fetch(const std::string& path, int delay_ms = 0) {
    auto file = std::make_shared<std::unique_ptr<MappedFile>>(std::move(MappedFile::open(path).value()));
    return [file, delay_ms](uint64_t offset, size_t length, uint8_t* dst) {
        if (delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        const MappedFile& f = **file;
        const size_t n = offset < f.size() ? std::min(length, f.size() - static_cast<size_t>(offset)) : 0;
        std::memcpy(dst, f.data() + offset, n);
        return Status::ok();
    };
}

}  // namespace

TEST_CASE("TripleBufferLoader delivers every block in cyclic order and after resyncs") {
    const BlockFile f = make_block_file(5);
    auto loader = TripleBufferLoader::create(f.items, pread_fetch(f.path));
    REQUIRE(loader.has_value());
    TripleBufferLoader& tb = *loader.value();
    for (int round = 0; round < 3; ++round) {  // three "tokens" over five blocks
        for (int32_t i = 0; i < 5; ++i) {
            auto data = tb.acquire(i);
            REQUIRE(data.has_value());
            CHECK(std::memcmp(data.value(), f.bytes.data() + f.items[i].offset, f.items[i].length) == 0);
            tb.release(i);
        }
    }
    // Out of order (e.g. early exit jumping back to block 0): resynchronizes.
    REQUIRE(tb.acquire(3).has_value());
    tb.release(3);
    auto first = tb.acquire(0);
    REQUIRE(first.has_value());
    CHECK(std::memcmp(first.value(), f.bytes.data() + f.items[0].offset, f.items[0].length) == 0);
    CHECK(tb.stats().resyncs >= 1);
    CHECK(!tb.acquire(0).has_value());  // double acquire is rejected
    tb.release(0);
    CHECK(!tb.acquire(7).has_value());
    std::remove(f.path.c_str());
}

TEST_CASE("TripleBufferLoader stage 2 runs SIMD INT4 unpacking off the critical path") {
    // Each block stores packed INT4 in its first half; stage 2 expands it to
    // INT8 over the whole block, so the consumer sees ready-to-use bytes.
    const BlockFile f = make_block_file(4);
    auto unpack = [](int32_t, uint8_t* data, size_t length) {
        thread_local std::vector<uint8_t> packed;
        packed.assign(data, data + length / 2);
        MmapLoader::unpack_int4_to_int8_neon(packed.data(), reinterpret_cast<int8_t*>(data), length / 2 * 2);
        return Status::ok();
    };
    auto loader = TripleBufferLoader::create(f.items, pread_fetch(f.path), unpack);
    REQUIRE(loader.has_value());
    for (int32_t i = 0; i < 4; ++i) {
        auto data = loader.value()->acquire(i);
        REQUIRE(data.has_value());
        const uint8_t* src = f.bytes.data() + f.items[i].offset;
        const auto* out = reinterpret_cast<const int8_t*>(data.value());
        bool ok = true;
        for (size_t k = 0; k < f.items[i].length / 2; ++k) {
            ok = ok && out[2 * k] == (src[k] & 0x0F) - 8 && out[2 * k + 1] == (src[k] >> 4) - 8;
        }
        CHECK(ok);
        loader.value()->release(i);
    }
    CHECK(loader.value()->stats().transform_ms > 0.0);
    std::remove(f.path.c_str());
}

TEST_CASE("TripleBufferLoader keeps the executor fed when compute outlasts I/O, and reports stalls otherwise") {
    const BlockFile f = make_block_file(6);
    auto run = [&](int fetch_ms, int compute_ms) {
        auto loader = TripleBufferLoader::create(f.items, pread_fetch(f.path, fetch_ms));
        for (int round = 0; round < 3; ++round) {
            for (int32_t i = 0; i < 6; ++i) {
                (void)loader.value()->acquire(i);
                std::this_thread::sleep_for(std::chrono::milliseconds(compute_ms));  // "execution"
                loader.value()->release(i);
            }
        }
        return loader.value()->stats();
    };
    const auto fed = run(2, 8);      // I/O 2 ms per block, compute 8 ms
    const auto starved = run(8, 1);  // I/O slower than compute
    std::printf("  fast I/O: %llu stalls (%.1f ms waiting); slow I/O: %llu stalls (%.1f ms waiting)\n",
                static_cast<unsigned long long>(fed.stalls), fed.consumer_wait_ms,
                static_cast<unsigned long long>(starved.stalls), starved.consumer_wait_ms);
    CHECK(fed.stalls <= 1);  // only the cold first block
    CHECK(starved.stalls > 10);
    CHECK(fed.items_fetched >= 18);
    std::remove(f.path.c_str());
}

TEST_CASE("TripleBufferLoader surfaces fetch errors to the consumer") {
    auto loader = TripleBufferLoader::create({{0, 100}, {100, 100}},
                                             [](uint64_t, size_t, uint8_t*) { return Status(ErrorCode::IoError, "flash gone"); });
    REQUIRE(loader.has_value());
    auto r = loader.value()->acquire(0);
    CHECK(!r.has_value());
    CHECK(r.status().code() == ErrorCode::IoError);
    CHECK(!TripleBufferLoader::create({}, [](uint64_t, size_t, uint8_t*) { return Status::ok(); }).has_value());
}

TEST_MAIN()
