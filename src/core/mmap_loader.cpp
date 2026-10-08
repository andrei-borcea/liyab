#include "liyab/mmap_loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <limits>

#include "core/direct_io.h"
#include "core/log.h"
#include "core/quant.h"

#if defined(LIYAB_USE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#if TARGET_OS_IPHONE
#include <os/proc.h>
#endif
#endif

namespace liyab {

namespace {

size_t page_size() noexcept {
    static const size_t size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    return size;
}

Status io_error(const std::string& what, const std::string& path) {
    return {ErrorCode::IoError, what + " '" + path + "': " + std::strerror(errno)};
}

}  // namespace

// ---------------------------------------------------------------------------
// MappedFile
// ---------------------------------------------------------------------------
Result<std::unique_ptr<MappedFile>> MappedFile::open(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return io_error("cannot open", path);

    struct stat st {};
    if (fstat(fd, &st) != 0) {
        Status s = io_error("cannot stat", path);
        ::close(fd);
        return s;
    }
    if (!S_ISREG(st.st_mode) || st.st_size <= 0) {
        ::close(fd);
        return Status(ErrorCode::IoError, "not a non-empty regular file: '" + path + "'");
    }
    const auto size = static_cast<size_t>(st.st_size);
    void* addr = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    Status map_error = addr == MAP_FAILED ? io_error("mmap failed for", path) : Status::ok();
    ::close(fd);  // the mapping keeps its own reference to the file
    if (!map_error.is_ok()) return map_error;

    return std::unique_ptr<MappedFile>(new MappedFile(path, static_cast<const uint8_t*>(addr), size));
}

MappedFile::~MappedFile() {
    if (data_ != nullptr) munmap(const_cast<uint8_t*>(data_), size_);
}

bool MappedFile::advise(size_t offset, size_t length, int flag) const noexcept {
    if (offset >= size_ || length == 0) return false;
    const size_t page = page_size();
    const size_t begin = offset & ~(page - 1);
    const size_t end = offset + std::min(length, size_ - offset);
    return madvise(const_cast<uint8_t*>(data_) + begin, end - begin, flag) == 0;
}

bool MappedFile::advise_sequential() const noexcept { return advise(0, size_, MADV_SEQUENTIAL); }

bool MappedFile::advise_willneed(size_t offset, size_t length) const noexcept {
    return advise(offset, length, MADV_WILLNEED);
}

bool MappedFile::advise_dontneed(size_t offset, size_t length) const noexcept {
    return advise(offset, length, MADV_DONTNEED);
}

void MappedFile::touch(size_t offset, size_t length) const noexcept {
    if (offset >= size_) return;
    const size_t page = page_size();
    const size_t end = offset + std::min(length, size_ - offset);
    uint8_t sink = 0;
    for (size_t p = offset & ~(page - 1); p < end; p += page) {
        sink ^= *static_cast<const volatile uint8_t*>(data_ + std::max(p, offset));
    }
    static std::atomic<uint8_t> keep_alive{0};
    keep_alive.fetch_xor(sink, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// GGUF metadata values
// ---------------------------------------------------------------------------
std::optional<int64_t> GgufValue::as_int() const {
    if (type == Type::Array) return std::nullopt;
    if (const auto* u = std::get_if<uint64_t>(&scalar)) {
        if (*u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) return std::nullopt;
        return static_cast<int64_t>(*u);
    }
    if (const auto* i = std::get_if<int64_t>(&scalar)) return *i;
    return std::nullopt;
}

std::optional<double> GgufValue::as_float() const {
    if (type == Type::Array) return std::nullopt;
    if (const auto* d = std::get_if<double>(&scalar)) return *d;
    if (auto i = as_int()) return static_cast<double>(*i);
    return std::nullopt;
}

std::optional<bool> GgufValue::as_bool() const {
    if (const auto* b = std::get_if<bool>(&scalar); b && type == Type::Bool) return *b;
    return std::nullopt;
}

std::optional<std::string_view> GgufValue::as_string() const {
    if (const auto* s = std::get_if<std::string_view>(&scalar); s && type == Type::String) return *s;
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// GGUF parsing. The file is untrusted input: every length, count and offset
// is validated against the mapping before use.
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t kGgufMagic = 0x46554747;  // "GGUF" little-endian
constexpr uint64_t kMaxArrayElements = uint64_t{1} << 28;

class Reader {
public:
    Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    [[nodiscard]] size_t pos() const noexcept { return pos_; }
    [[nodiscard]] size_t remaining() const noexcept { return size_ - pos_; }

    template <typename T>
    bool get(T& out) noexcept {
        if (remaining() < sizeof(T)) return false;
        std::memcpy(&out, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return true;
    }

    bool get_string(std::string_view& out) noexcept {
        uint64_t len = 0;
        if (!get(len) || len > remaining()) return false;
        out = std::string_view(reinterpret_cast<const char*>(data_ + pos_), static_cast<size_t>(len));
        pos_ += static_cast<size_t>(len);
        return true;
    }

    bool skip(uint64_t n) noexcept {
        if (n > remaining()) return false;
        pos_ += static_cast<size_t>(n);
        return true;
    }

    [[nodiscard]] const uint8_t* cursor() const noexcept { return data_ + pos_; }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
};

size_t scalar_size(GgufValue::Type t) noexcept {
    using T = GgufValue::Type;
    switch (t) {
        case T::U8: case T::I8: case T::Bool: return 1;
        case T::U16: case T::I16: return 2;
        case T::U32: case T::I32: case T::F32: return 4;
        case T::U64: case T::I64: case T::F64: return 8;
        case T::String: case T::Array: return 0;
    }
    return 0;
}

bool read_scalar(Reader& r, GgufValue::Type t, GgufValue& v) {
    using T = GgufValue::Type;
    switch (t) {
        case T::U8: { uint8_t x; if (!r.get(x)) return false; v.scalar = uint64_t{x}; return true; }
        case T::I8: { int8_t x; if (!r.get(x)) return false; v.scalar = int64_t{x}; return true; }
        case T::U16: { uint16_t x; if (!r.get(x)) return false; v.scalar = uint64_t{x}; return true; }
        case T::I16: { int16_t x; if (!r.get(x)) return false; v.scalar = int64_t{x}; return true; }
        case T::U32: { uint32_t x; if (!r.get(x)) return false; v.scalar = uint64_t{x}; return true; }
        case T::I32: { int32_t x; if (!r.get(x)) return false; v.scalar = int64_t{x}; return true; }
        case T::U64: { uint64_t x; if (!r.get(x)) return false; v.scalar = x; return true; }
        case T::I64: { int64_t x; if (!r.get(x)) return false; v.scalar = x; return true; }
        case T::F32: { float x; if (!r.get(x)) return false; v.scalar = double{x}; return true; }
        case T::F64: { double x; if (!r.get(x)) return false; v.scalar = x; return true; }
        case T::Bool: {
            uint8_t x;
            if (!r.get(x) || x > 1) return false;
            v.scalar = x != 0;
            return true;
        }
        case T::String: {
            std::string_view s;
            if (!r.get_string(s)) return false;
            v.scalar = s;
            return true;
        }
        case T::Array: return false;
    }
    return false;
}

bool valid_type(uint32_t t) noexcept { return t <= static_cast<uint32_t>(GgufValue::Type::F64); }

bool read_value(Reader& r, GgufValue& v, std::string& error) {
    uint32_t raw = 0;
    if (!r.get(raw) || !valid_type(raw)) {
        error = "invalid metadata value type";
        return false;
    }
    v.type = static_cast<GgufValue::Type>(raw);
    if (v.type != GgufValue::Type::Array) {
        if (!read_scalar(r, v.type, v)) error = "truncated or malformed metadata value";
        return error.empty();
    }
    uint32_t elem = 0;
    if (!r.get(elem) || !valid_type(elem) || !r.get(v.array_size)) {
        error = "malformed metadata array header";
        return false;
    }
    v.array_type = static_cast<GgufValue::Type>(elem);
    if (v.array_type == GgufValue::Type::Array) {
        error = "nested metadata arrays are not supported";
        return false;
    }
    if (v.array_size > kMaxArrayElements) {
        error = "metadata array too large";
        return false;
    }
    v.array_data = r.cursor();
    if (v.array_type == GgufValue::Type::String) {
        // Each string needs at least its 8-byte length prefix.
        if (v.array_size > r.remaining() / 8) {
            error = "truncated string array";
            return false;
        }
        v.strings.resize(static_cast<size_t>(v.array_size));
        for (auto& s : v.strings) {
            if (!r.get_string(s)) {
                error = "truncated string array";
                return false;
            }
        }
        return true;
    }
    const size_t width = scalar_size(v.array_type);
    if (v.array_size > r.remaining() / width || !r.skip(v.array_size * width)) {
        error = "truncated metadata array";
        return false;
    }
    return true;
}

Status invalid(const std::string& path, const std::string& what) {
    return {ErrorCode::InvalidModel, "'" + path + "': " + what};
}

}  // namespace

Result<std::unique_ptr<MmapLoader>> MmapLoader::open(const std::string& path, const LoaderOptions& options) {
    auto file = MappedFile::open(path);
    if (!file) return file.status();

    std::unique_ptr<MmapLoader> loader(new MmapLoader());
    loader->files_.push_back(std::move(file).value());
    LIYAB_RETURN_IF_ERROR(loader->parse(0));
    LIYAB_RETURN_IF_ERROR(loader->open_other_parts());

    const uint64_t available = usable_memory_bytes(options.memory_budget_bytes);
    loader->streaming_ = options.streaming.value_or(
        available > 0 && static_cast<double>(loader->file_size()) > 0.8 * static_cast<double>(available));

    for (const auto& f : loader->files_) {
        if (loader->streaming_) f->advise_sequential();
        else f->advise_willneed(0, f->size());  // fits in RAM: warm the page cache once
    }
    LIYAB_LOG_INFO("mapped %s%s: %.2f GiB, %zu tensors, gguf v%u, %s mode", path.c_str(),
                   loader->files_.size() > 1 ? (" + " + std::to_string(loader->files_.size() - 1) + " parts").c_str() : "",
                   static_cast<double>(loader->file_size()) / (1024.0 * 1024.0 * 1024.0), loader->tensors_.size(),
                   loader->version_, loader->streaming_ ? "streaming" : "resident");
    return loader;
}

size_t MmapLoader::file_size() const noexcept {
    size_t total = 0;
    for (const auto& f : files_) total += f->size();
    return total;
}

Status MmapLoader::open_other_parts() {
    const auto count = get_int("split.count").value_or(1);
    if (count <= 1) return Status::ok();
    const auto part = get_int("split.no").value_or(0);
    const std::string& path = files_.front()->path();
    if (count > 9999 || part < 0 || part >= count) return invalid(path, "invalid split.no / split.count");
    if (part != 0) {
        return Status(ErrorCode::Unsupported, "this file is part " + std::to_string(part + 1) + " of " +
                                                  std::to_string(count) +
                                                  " of a split GGUF model: open part 1 (-00001-of-...), with the "
                                                  "other parts in the same folder");
    }
    // gguf-split naming: <prefix>-00001-of-0000N.gguf
    char suffix[32];
    std::snprintf(suffix, sizeof suffix, "-00001-of-%05lld.gguf", static_cast<long long>(count));
    const std::string_view first_suffix(suffix);
    if (path.size() <= first_suffix.size() || path.compare(path.size() - first_suffix.size(), first_suffix.size(),
                                                           first_suffix) != 0) {
        return Status(ErrorCode::Unsupported, "split GGUF model (" + std::to_string(count) +
                                                  " parts) opened without its file name, so the other parts cannot "
                                                  "be found: open part 1 by path from the folder holding every part");
    }
    const std::string prefix = path.substr(0, path.size() - first_suffix.size());
    for (int64_t i = 1; i < count; ++i) {
        std::snprintf(suffix, sizeof suffix, "-%05lld-of-%05lld.gguf", static_cast<long long>(i + 1),
                      static_cast<long long>(count));
        auto part_file = MappedFile::open(prefix + suffix);
        if (!part_file) {
            return Status(ErrorCode::IoError, "split model part " + std::to_string(i + 1) + " of " +
                                                  std::to_string(count) + " is missing: " +
                                                  part_file.status().message());
        }
        files_.push_back(std::move(part_file).value());
        LIYAB_RETURN_IF_ERROR(parse(static_cast<uint32_t>(i)));
    }
    return Status::ok();
}

MmapLoader::~MmapLoader() {
    {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        prefetch_stop_ = true;
    }
    prefetch_cv_.notify_all();
    if (prefetch_thread_.joinable()) prefetch_thread_.join();
    for (const auto& [p, bytes] : converted_) munmap(p, bytes);
}

Status MmapLoader::parse(uint32_t shard) {
    const MappedFile& file = *files_[shard];
    const std::string& path = file.path();
    Reader r(file.data(), file.size());

    uint32_t magic = 0;
    if (!r.get(magic) || magic != kGgufMagic) return invalid(path, "not a GGUF file (bad magic)");
    uint32_t version = 0;
    if (!r.get(version) || (version != 2 && version != 3)) {
        return invalid(path, "unsupported GGUF version " + std::to_string(version) + " (need 2 or 3)");
    }
    if (shard == 0) version_ = version;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;
    if (!r.get(n_tensors) || !r.get(n_kv)) return invalid(path, "truncated header");
    // Lower bounds on encoded sizes reject absurd counts before allocating.
    if (n_kv > r.remaining() / 12 || n_tensors > r.remaining() / 24) {
        return invalid(path, "header counts exceed file size");
    }

    // Later parts carry only split bookkeeping: validated, checked, dropped.
    std::unordered_map<std::string_view, GgufValue> part_metadata;
    auto& meta = shard == 0 ? metadata_ : part_metadata;
    meta.reserve(static_cast<size_t>(n_kv));
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string_view key;
        if (!r.get_string(key)) return invalid(path, "truncated metadata key");
        GgufValue value;
        std::string error;
        if (!read_value(r, value, error)) return invalid(path, "metadata '" + std::string(key) + "': " + error);
        if (!meta.emplace(key, std::move(value)).second) {
            return invalid(path, "duplicate metadata key '" + std::string(key) + "'");
        }
    }
    if (shard > 0) {
        const auto it = part_metadata.find("split.no");
        if (it == part_metadata.end() || it->second.as_int() != int64_t{shard}) {
            return invalid(path, "not part " + std::to_string(shard + 1) + " of this split model");
        }
    }

    uint64_t alignment = 32;
    const auto align_it = meta.find("general.alignment");
    if (const GgufValue* a = align_it == meta.end() ? nullptr : &align_it->second) {
        const auto v = a->as_int();
        if (!v || *v <= 0 || (*v & (*v - 1)) != 0) return invalid(path, "general.alignment must be a power of two");
        alignment = static_cast<uint64_t>(*v);
    }

    struct RawInfo { uint32_t ggml_type; uint64_t offset; };
    std::vector<RawInfo> raw(static_cast<size_t>(n_tensors));
    const size_t first = tensors_.size();
    tensors_.resize(first + static_cast<size_t>(n_tensors));
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorView& t = tensors_[first + i];
        uint32_t n_dims = 0;
        if (!r.get_string(t.name) || !r.get(n_dims)) return invalid(path, "truncated tensor info");
        if (n_dims == 0 || n_dims > 4) return invalid(path, "tensor '" + std::string(t.name) + "' has unsupported rank");
        t.n_dims = static_cast<int32_t>(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint64_t ne = 0;
            if (!r.get(ne)) return invalid(path, "truncated tensor dims");
            if (ne == 0 || ne > (uint64_t{1} << 40)) return invalid(path, "tensor '" + std::string(t.name) + "' has invalid dims");
            t.ne[d] = static_cast<int64_t>(ne);
        }
        if (!r.get(raw[i].ggml_type) || !r.get(raw[i].offset)) return invalid(path, "truncated tensor info");
    }

    const uint64_t data_start = (static_cast<uint64_t>(r.pos()) + alignment - 1) / alignment * alignment;
    if (data_start > file.size()) return invalid(path, "tensor data section starts past end of file");
    const uint64_t data_size = file.size() - data_start;

    tensor_index_.reserve(tensors_.size());
    for (size_t i = 0; i < static_cast<size_t>(n_tensors); ++i) {
        TensorView& t = tensors_[first + i];
        const std::string name(t.name);
        if (!dtype_from_ggml(raw[i].ggml_type, t.type)) {
            return Status(ErrorCode::Unsupported,
                          "tensor '" + name + "' uses " + std::string(ggml_type_name(raw[i].ggml_type)) +
                              ", which Liyab does not execute (supported: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, "
                              "Q8_0, Q2_K..Q6_K, IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS, TQ1_0, TQ2_0, MXFP4, NVFP4)");
        }
        const DTypeTraits traits = dtype_traits(t.type);
        if (t.ne[0] % traits.block_size != 0) return invalid(path, "tensor '" + name + "' row is not a whole number of blocks");

        uint64_t elements = 1;
        for (int d = 0; d < t.n_dims; ++d) {
            if (__builtin_mul_overflow(elements, static_cast<uint64_t>(t.ne[d]), &elements)) {
                return invalid(path, "tensor '" + name + "' size overflows");
            }
        }
        const uint64_t nbytes = elements / static_cast<uint64_t>(traits.block_size) *
                                static_cast<uint64_t>(traits.block_bytes);
        if (raw[i].offset > data_size || nbytes > data_size - raw[i].offset) {
            return invalid(path, "tensor '" + name + "' data lies outside the file");
        }
        t.file_offset = data_start + raw[i].offset;
        t.shard = shard;
        t.data = file.data() + t.file_offset;
        t.nbytes = static_cast<size_t>(nbytes);
        if (!tensor_index_.emplace(t.name, first + i).second) return invalid(path, "duplicate tensor '" + name + "'");
    }
    return Status::ok();
}

const GgufValue* MmapLoader::metadata(std::string_view key) const {
    const auto it = metadata_.find(key);
    return it == metadata_.end() ? nullptr : &it->second;
}

std::optional<int64_t> MmapLoader::get_int(std::string_view key) const {
    const GgufValue* v = metadata(key);
    return v ? v->as_int() : std::nullopt;
}

std::optional<double> MmapLoader::get_float(std::string_view key) const {
    const GgufValue* v = metadata(key);
    return v ? v->as_float() : std::nullopt;
}

std::optional<std::string_view> MmapLoader::get_string(std::string_view key) const {
    const GgufValue* v = metadata(key);
    return v ? v->as_string() : std::nullopt;
}

const TensorView* MmapLoader::tensor(std::string_view name) const {
    const auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

// ---------------------------------------------------------------------------
// Layer-window prefetching
// ---------------------------------------------------------------------------
void MmapLoader::configure_layers(int32_t n_layers) {
    if (n_layers <= 0 || !layer_ranges_.empty()) return;
    // Slots 0..n-1 are transformer blocks; slot n holds the output head, which
    // is read after the last block of every token.
    layer_ranges_.assign(static_cast<size_t>(n_layers) + 1, {});
    auto extend = [](std::vector<Range>& ranges, const TensorView& t) {
        auto it = std::find_if(ranges.begin(), ranges.end(), [&](const Range& r) { return r.shard == t.shard; });
        if (it == ranges.end()) {
            ranges.push_back(Range{t.shard, static_cast<size_t>(t.file_offset), static_cast<size_t>(t.file_offset) + t.nbytes});
            return;
        }
        it->begin = std::min(it->begin, static_cast<size_t>(t.file_offset));
        it->end = std::max(it->end, static_cast<size_t>(t.file_offset) + t.nbytes);
    };
    for (const TensorView& t : tensors_) {
        if (t.name.substr(0, 4) == "blk.") {
            int32_t layer = -1;
            const char* first = t.name.data() + 4;
            const char* last = t.name.data() + t.name.size();
            const auto [ptr, ec] = std::from_chars(first, last, layer);
            if (ec == std::errc() && ptr != last && *ptr == '.' && layer >= 0 && layer < n_layers) {
                extend(layer_ranges_[static_cast<size_t>(layer)], t);
            }
        } else if (t.name == "output.weight" || t.name == "output_norm.weight") {
            extend(layer_ranges_.back(), t);
        }
    }
    if (tensor("output.weight") == nullptr) {
        if (const TensorView* emb = tensor("token_embd.weight")) extend(layer_ranges_.back(), *emb);
    }
    if (streaming_) prefetch_thread_ = std::thread([this] { prefetch_loop(); });
}

Result<size_t> MmapLoader::relocate(const std::vector<std::pair<size_t, size_t>>& ranges,
                                    const std::function<uint8_t*(size_t bytes)>& allocate) {
    auto reader = DirectFile::open(files_.front()->path());
    if (!reader) return reader.status();
    constexpr size_t kAlign = DirectFile::kAlign;
    size_t moved = 0;
    for (const auto& [begin, end] : ranges) {
        if (end <= begin) continue;
        const size_t aligned = begin / kAlign * kAlign;
        const size_t length = (end - aligned + kAlign - 1) / kAlign * kAlign;
        uint8_t* dst = allocate(length);
        if (dst == nullptr) return Status(ErrorCode::OutOfMemory, "cannot allocate memory for relocated weights");
        if (reinterpret_cast<uintptr_t>(dst) % kAlign == 0) {
            LIYAB_RETURN_IF_ERROR(reader.value()->read_parallel(aligned, length, dst, 4));
        } else {  // not aligned for direct I/O: copy from the mapping instead
            const MappedFile& f = *files_.front();
            std::memcpy(dst, f.data() + aligned, std::min(length, f.size() - aligned));
        }
        for (TensorView& t : tensors_) {
            if (t.shard == 0 && t.file_offset >= begin && t.file_offset + t.nbytes <= end) {
                t.data = dst + (t.file_offset - aligned);
            }
        }
        files_.front()->advise_dontneed(begin, end - begin);
        moved += end - begin;
    }
    return moved;
}

Result<size_t> MmapLoader::requantize(const std::function<bool(const TensorView&)>& select, DType target) {
    // Repacked layouts: their source type and rows per group.
    DType source = DType::Q8_0;
    int64_t group = 0;
    switch (target) {
        case DType::Q4_K: case DType::Q5_K: break;
        case DType::Q4_K_R8: source = DType::Q4_K; group = 8; break;
        case DType::Q5_K_R8: source = DType::Q5_K; group = 8; break;
        case DType::Q6_K_R8: source = DType::Q6_K; group = 8; break;
        case DType::Q8_0_R4: source = DType::Q8_0; group = 4; break;
        default:
            return Status(ErrorCode::InvalidArgument,
                          "requantize() converts to Q4_K, Q5_K, Q4_K_R8, Q5_K_R8, Q6_K_R8 or Q8_0_R4 only");
    }
    const int64_t col_multiple = target == DType::Q8_0_R4 ? quant::kBlock : quant::kSuperBlock;
    size_t saved = 0;
    for (TensorView& t : tensors_) {
        if (!select(t)) continue;
        if (t.type != source || (group != 0 && t.rows() % group != 0) || t.cols() % col_multiple != 0) {
            return Status(ErrorCode::InvalidArgument, "cannot requantize " + std::string(t.name));
        }
        TensorView out = t;
        out.type = target;
        const size_t row_in = t.row_bytes();
        const size_t row_out = out.row_bytes();
        const auto rows = static_cast<size_t>(t.rows());
        const size_t bytes = row_out * rows;
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) return Status(ErrorCode::OutOfMemory, "cannot allocate requantized weights");
        converted_.emplace_back(p, bytes);
        auto* dst = static_cast<uint8_t*>(p);
        // Repacking: the same bytes, rows interleaved (a rearrangement).
        if (target == DType::Q4_K_R8) {
            quant::repack_q4_K_r8(reinterpret_cast<const quant::BlockQ4_K*>(t.data), t.rows(), t.cols(),
                                  reinterpret_cast<quant::BlockQ4_Kx8*>(dst));
        } else if (target == DType::Q5_K_R8) {
            quant::repack_q5_K_r8(reinterpret_cast<const quant::BlockQ5_K*>(t.data), t.rows(), t.cols(),
                                  reinterpret_cast<quant::BlockQ5_Kx8*>(dst));
        } else if (target == DType::Q6_K_R8) {
            quant::repack_q6_K_r8(reinterpret_cast<const quant::BlockQ6_K*>(t.data), t.rows(), t.cols(),
                                  reinterpret_cast<quant::BlockQ6_Kx8*>(dst));
        } else if (target == DType::Q8_0_R4) {
            quant::repack_q8_0_r4(reinterpret_cast<const quant::BlockQ8_0*>(t.data), t.rows(), t.cols(),
                                  reinterpret_cast<quant::BlockQ8_0x4*>(dst));
        } else {
        // Rows are independent; the reference quantizer is slow (a weighted
        // search per 32 values), so every core takes a share.
        const auto n_threads = static_cast<size_t>(std::max(1u, std::thread::hardware_concurrency()));
        std::vector<std::thread> workers;
        for (size_t w = 0; w < n_threads; ++w) {
            workers.emplace_back([&, w] {
                std::vector<float> row(static_cast<size_t>(t.cols()));
                for (size_t r = rows * w / n_threads; r < rows * (w + 1) / n_threads; ++r) {
                    quant::dequantize_row(DType::Q8_0, t.data + r * row_in, row.data(), t.cols());
                    quant::quantize_row(target, row.data(), dst + r * row_out, t.cols());
                }
            });
        }
        for (std::thread& w : workers) w.join();
        }
        if (t.data >= files_[t.shard]->data() && t.data < files_[t.shard]->data() + files_[t.shard]->size()) {
            files_[t.shard]->advise_dontneed(t.file_offset, t.nbytes);
        }
        saved += t.nbytes - bytes;
        t.type = target;
        t.data = dst;
        t.nbytes = bytes;
    }
    return saved;
}

bool MmapLoader::converted(const TensorView& t) const noexcept {
    return std::any_of(converted_.begin(), converted_.end(), [&](const auto& region) {
        const auto* begin = static_cast<const uint8_t*>(region.first);
        return t.data >= begin && t.data < begin + region.second;
    });
}

size_t MmapLoader::keep_resident(const std::function<bool(const TensorView&)>& resident) {
    streaming_ = false;
    size_t bytes = 0;
    for (const TensorView& t : tensors_) {
        const MappedFile& f = *files_[t.shard];
        if (resident(t)) {
            f.advise_willneed(t.file_offset, t.nbytes);
            bytes += t.nbytes;
        } else {
            f.advise_dontneed(t.file_offset, t.nbytes);  // drop anything read so far
            const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
            const size_t begin = static_cast<size_t>(t.file_offset) & ~(page - 1);
            madvise(const_cast<uint8_t*>(f.data()) + begin, static_cast<size_t>(t.file_offset) + t.nbytes - begin,
                    MADV_RANDOM);
        }
    }
    return bytes;
}

void MmapLoader::begin_layer(int32_t layer) {
    if (!streaming_ || layer_ranges_.empty()) return;
    const auto slots = static_cast<int32_t>(layer_ranges_.size());
    if (layer < 0 || layer >= slots) return;

    // Triple-buffer window: [current, next, next+1].
    std::array<int32_t, 3> window{layer, (layer + 1) % slots, (layer + 2) % slots};
    auto in_window = [&](int32_t l) { return std::find(window.begin(), window.end(), l) != window.end(); };

    {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        // Release slots that fell behind the compute cursor.
        for (auto it = resident_.begin(); it != resident_.end();) {
            if (!in_window(*it)) {
                for (const Range& r : layer_ranges_[static_cast<size_t>(*it)]) {
                    files_[r.shard]->advise_dontneed(r.begin, r.end - r.begin);
                }
                prefetch_queue_.erase(std::remove(prefetch_queue_.begin(), prefetch_queue_.end(), *it),
                                      prefetch_queue_.end());
                it = resident_.erase(it);
            } else {
                ++it;
            }
        }
        if (std::find(resident_.begin(), resident_.end(), layer) == resident_.end()) {
            resident_.push_back(layer);  // the compute thread faults it in itself
        }
        for (size_t i = 1; i < window.size(); ++i) {
            const int32_t next = window[i];
            if (std::find(resident_.begin(), resident_.end(), next) == resident_.end()) {
                resident_.push_back(next);
                prefetch_queue_.push_back(next);
            }
        }
    }
    prefetch_cv_.notify_one();
}

std::pair<size_t, size_t> MmapLoader::layer_range(int32_t slot) const {
    if (slot < 0 || static_cast<size_t>(slot) >= layer_ranges_.size()) return {0, 0};
    const std::vector<Range>& ranges = layer_ranges_[static_cast<size_t>(slot)];
    if (ranges.size() != 1 || files_.size() != 1) return {0, 0};
    return {ranges.front().begin, ranges.front().end};
}

void MmapLoader::wait_prefetch_idle() {
    std::unique_lock<std::mutex> lock(prefetch_mutex_);
    prefetch_idle_cv_.wait(lock, [this] { return prefetch_queue_.empty() && !prefetch_busy_; });
}

void MmapLoader::prefetch_loop() {
    for (;;) {
        int32_t layer = 0;
        {
            std::unique_lock<std::mutex> lock(prefetch_mutex_);
            prefetch_cv_.wait(lock, [this] { return prefetch_stop_ || !prefetch_queue_.empty(); });
            if (prefetch_stop_) return;
            layer = prefetch_queue_.front();
            prefetch_queue_.pop_front();
            prefetch_busy_ = true;
        }
        for (const Range& r : layer_ranges_[static_cast<size_t>(layer)]) {
            const MappedFile& f = *files_[r.shard];
            f.advise_willneed(r.begin, r.end - r.begin);
            f.touch(r.begin, r.end - r.begin);
            prefetched_bytes_.fetch_add(r.end - r.begin, std::memory_order_relaxed);
        }
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            prefetch_busy_ = false;
            if (prefetch_queue_.empty()) prefetch_idle_cv_.notify_all();
        }
    }
}

// ---------------------------------------------------------------------------
// JIT bit-unpacking
// ---------------------------------------------------------------------------
void MmapLoader::unpack_int4_to_int8_neon(const uint8_t* __restrict src_packed, int8_t* __restrict dst_unpacked,
                                          size_t num_elements) {
    size_t i = 0;  // elements produced
#if defined(LIYAB_USE_NEON) && defined(__ARM_NEON)
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);
    const int8x16_t offset = vdupq_n_s8(8);
    for (; i + 32 <= num_elements; i += 32) {
        const uint8x16_t packed = vld1q_u8(src_packed + i / 2);
        const uint8x16_t lo = vandq_u8(packed, low_mask);   // elements 0, 2, 4, ...
        const uint8x16_t hi = vshrq_n_u8(packed, 4);        // elements 1, 3, 5, ...
        const uint8x16x2_t interleaved = vzipq_u8(lo, hi);  // restore element order
        vst1q_s8(dst_unpacked + i, vsubq_s8(vreinterpretq_s8_u8(interleaved.val[0]), offset));
        vst1q_s8(dst_unpacked + i + 16, vsubq_s8(vreinterpretq_s8_u8(interleaved.val[1]), offset));
    }
#endif
    for (; i + 2 <= num_elements; i += 2) {
        const uint8_t b = src_packed[i / 2];
        dst_unpacked[i] = static_cast<int8_t>((b & 0x0F) - 8);
        dst_unpacked[i + 1] = static_cast<int8_t>((b >> 4) - 8);
    }
    if (i < num_elements) dst_unpacked[i] = static_cast<int8_t>((src_packed[i / 2] & 0x0F) - 8);
}

// ---------------------------------------------------------------------------
// Memory budget
// ---------------------------------------------------------------------------
uint64_t usable_memory_bytes(uint64_t budget_bytes) noexcept {
    const uint64_t available = available_memory_bytes();
    if (budget_bytes == 0) return available;
    return available == 0 ? budget_bytes : std::min(available, budget_bytes);
}

uint64_t available_memory_bytes() noexcept {
#if defined(__APPLE__) && TARGET_OS_IPHONE
    return static_cast<uint64_t>(os_proc_available_memory());
#elif defined(__APPLE__)
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) !=
        KERN_SUCCESS) {
        return 0;
    }
    const uint64_t pages = static_cast<uint64_t>(vm.free_count) + vm.inactive_count + vm.purgeable_count;
    return pages * static_cast<uint64_t>(vm_page_size);
#elif defined(__linux__)
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (std::fgets(line, sizeof line, f)) {
        unsigned long long v = 0;
        if (std::sscanf(line, "MemAvailable: %llu kB", &v) == 1) {
            kb = v;
            break;
        }
    }
    std::fclose(f);
    return kb * 1024;
#else
    return 0;
#endif
}

}  // namespace liyab
