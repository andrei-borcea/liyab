// Liyab — zero-copy model loading.
//
// MmapLoader maps a GGUF file read-only and exposes its metadata and tensors
// as views into the mapping: no weight byte is ever copied by the loader.
//
// For models larger than available RAM it switches to streaming mode: the
// mapping is advised MADV_SEQUENTIAL and a background prefetcher keeps a
// three-layer window resident (the layer being computed, plus the next two
// being faulted in) and releases layers behind the compute cursor. Pages are
// faulted on the prefetch thread so the compute threads never block on flash.
#ifndef LIYAB_MMAP_LOADER_H
#define LIYAB_MMAP_LOADER_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "liyab/types.h"

namespace liyab {

// RAII read-only file mapping.
class LIYAB_API MappedFile {
public:
    static Result<std::unique_ptr<MappedFile>> open(const std::string& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    [[nodiscard]] const uint8_t* data() const noexcept { return data_; }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] const std::string& path() const noexcept { return path_; }

    // Paging hints over [offset, offset + length), widened to whole pages and
    // clamped to the file. They return whether madvise() accepted the hint;
    // callers may ignore the result since every hint is advisory.
    //
    // Streaming: aggressive read-ahead, pages behind the cursor reclaimed first.
    bool advise_sequential() const noexcept;
    // Start reading pages in asynchronously (prefetch upcoming layers).
    bool advise_willneed(size_t offset, size_t length) const noexcept;
    // Drop our page-table entries now. The mapping is read-only and file
    // backed, so this frees RSS immediately and the next access re-reads the
    // page from the page cache or flash; no data can be lost.
    bool advise_dontneed(size_t offset, size_t length) const noexcept;
    // Faults [offset, offset + length) in by touching one byte per page.
    void touch(size_t offset, size_t length) const noexcept;

private:
    bool advise(size_t offset, size_t length, int flag) const noexcept;

    MappedFile(std::string path, const uint8_t* data, size_t size)
        : path_(std::move(path)), data_(data), size_(size) {}

    std::string path_;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

// One GGUF metadata value. Strings and arrays point into the mapping.
struct GgufValue {
    enum class Type : uint32_t {
        U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, Bool = 7,
        String = 8, Array = 9, U64 = 10, I64 = 11, F64 = 12,
    };
    Type type = Type::U8;
    std::variant<uint64_t, int64_t, double, bool, std::string_view> scalar;
    // Arrays only: element type, count and raw element storage.
    Type array_type = Type::U8;
    uint64_t array_size = 0;
    const uint8_t* array_data = nullptr;
    std::vector<std::string_view> strings;  // filled for string arrays

    [[nodiscard]] std::optional<int64_t> as_int() const;
    [[nodiscard]] std::optional<double> as_float() const;
    [[nodiscard]] std::optional<bool> as_bool() const;
    [[nodiscard]] std::optional<std::string_view> as_string() const;
};

struct LoaderOptions {
    // nullopt: stream iff the file is larger than 80% of available memory.
    std::optional<bool> streaming;
};

class LIYAB_API MmapLoader {
public:
    static Result<std::unique_ptr<MmapLoader>> open(const std::string& path, const LoaderOptions& options = {});
    ~MmapLoader();
    MmapLoader(const MmapLoader&) = delete;
    MmapLoader& operator=(const MmapLoader&) = delete;

    [[nodiscard]] uint32_t gguf_version() const noexcept { return version_; }
    [[nodiscard]] size_t file_size() const noexcept { return file_->size(); }
    [[nodiscard]] bool streaming() const noexcept { return streaming_; }
    [[nodiscard]] const MappedFile& file() const noexcept { return *file_; }

    [[nodiscard]] const GgufValue* metadata(std::string_view key) const;
    [[nodiscard]] std::optional<int64_t> get_int(std::string_view key) const;
    [[nodiscard]] std::optional<double> get_float(std::string_view key) const;
    [[nodiscard]] std::optional<std::string_view> get_string(std::string_view key) const;
    [[nodiscard]] size_t metadata_count() const noexcept { return metadata_.size(); }

    [[nodiscard]] const TensorView* tensor(std::string_view name) const;
    [[nodiscard]] const std::vector<TensorView>& tensors() const noexcept { return tensors_; }

    // --- Layer-window prefetching (streaming mode) -------------------------
    // Groups tensors named "blk.<i>.*" into per-layer byte ranges. Must be
    // called once before begin_layer(); `n_layers` comes from the model header.
    void configure_layers(int32_t n_layers);
    // Called by the compute thread when it starts layer `layer`: schedules
    // layer+1 and layer+2 (wrapping to the next token) and releases layer-1.
    void begin_layer(int32_t layer);
    // Byte range [first, second) of the file covering block `slot`'s tensors
    // (slot n_layers = output head). Requires configure_layers().
    [[nodiscard]] std::pair<size_t, size_t> layer_range(int32_t slot) const;
    // Blocks until all queued prefetches completed (tests, benchmarks).
    void wait_prefetch_idle();
    [[nodiscard]] uint64_t prefetched_bytes() const noexcept { return prefetched_bytes_.load(); }

    // --- JIT bit-unpacking --------------------------------------------------
    // Expands packed INT4 into INT8 for integer engines that consume one
    // value per byte. Byte i holds element 2i in its low nibble and element
    // 2i+1 in its high nibble; nibbles are offset-binary (value = nibble - 8,
    // the GGML Q4_0 convention), so outputs lie in [-8, 7]. An odd
    // `num_elements` uses only the low nibble of the last byte.
    // On ARM NEON 32 elements are produced per iteration entirely in 128-bit
    // registers (vld1q_u8 → vandq_u8/vshrq_n_u8 → vzipq_u8 → vst1q_s8) with no
    // heap allocation; other targets run the equivalent scalar loop.
    static void unpack_int4_to_int8_neon(const uint8_t* __restrict src_packed, int8_t* __restrict dst_unpacked,
                                         size_t num_elements);

private:
    MmapLoader() = default;
    Status parse();
    void prefetch_loop();

    std::unique_ptr<MappedFile> file_;
    uint32_t version_ = 0;
    bool streaming_ = false;
    std::unordered_map<std::string_view, GgufValue> metadata_;
    std::vector<TensorView> tensors_;
    std::unordered_map<std::string_view, size_t> tensor_index_;

    struct Range { size_t begin = 0; size_t end = 0; };
    std::vector<Range> layer_ranges_;

    std::thread prefetch_thread_;
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_cv_;
    std::condition_variable prefetch_idle_cv_;
    std::deque<int32_t> prefetch_queue_;
    std::vector<int32_t> resident_;  // layers currently inside the window
    bool prefetch_busy_ = false;
    bool prefetch_stop_ = false;
    std::atomic<uint64_t> prefetched_bytes_{0};
};

// Physical memory the process can still use without being killed, in bytes.
LIYAB_API uint64_t available_memory_bytes() noexcept;

}  // namespace liyab

#endif  // LIYAB_MMAP_LOADER_H
