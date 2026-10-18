// Liyab — LoRA adapters over a loaded model (see lora.h).
#include "core/lora.h"

#include <algorithm>
#include <cstring>
#include <map>

#include "core/quant.h"
#include "core/thread_pool.h"

namespace liyab {

namespace {

constexpr std::string_view kSuffixA = ".lora_a";
constexpr std::string_view kSuffixB = ".lora_b";

bool float_type(DType t) { return t == DType::F32 || t == DType::F16 || t == DType::BF16; }

// Slice `e` of a stack of 2-D matrices (ne[2] = count) as a 2-D view.
TensorView slice(const TensorView& stack, int32_t e) {
    TensorView v = stack;
    v.n_dims = 2;
    v.ne = {stack.ne[0], stack.ne[1], 1, 1};
    const size_t bytes = static_cast<size_t>(stack.ne[1]) * stack.row_bytes();
    v.data = stack.data + static_cast<size_t>(e) * bytes;
    v.nbytes = bytes;
    return v;
}

// Bᵀ of every expert of `b` (ne {rank, rows, experts}) into `out`, in b's
// type when it is a float type, else in F32; returns that type.
DType transpose_b(const TensorView& b, std::vector<uint8_t>& out) {
    const auto rank = static_cast<size_t>(b.ne[0]);
    const auto rows = static_cast<size_t>(b.ne[1]);
    const auto experts = static_cast<size_t>(b.ne[2]);
    const DType type = float_type(b.type) ? b.type : DType::F32;
    const size_t elem = type == DType::F32 ? 4 : 2;
    out.resize(experts * rank * rows * elem);
    std::vector<float> row(rank);
    for (size_t e = 0; e < experts; ++e) {
        const uint8_t* src = b.data + e * rows * b.row_bytes();
        uint8_t* dst = out.data() + e * rank * rows * elem;
        for (size_t o = 0; o < rows; ++o) {
            const uint8_t* in = src + o * b.row_bytes();
            if (type != b.type) {  // a block type: decoded first
                quant::dequantize_row(b.type, in, row.data(), static_cast<int64_t>(rank));
                in = reinterpret_cast<const uint8_t*>(row.data());
            }
            for (size_t r = 0; r < rank; ++r) std::memcpy(dst + (r * rows + o) * elem, in + r * elem, elem);
        }
    }
    return type;
}

}  // namespace

TensorView LoraTarget::a_of(int32_t e) const noexcept { return experts > 0 ? slice(a, e) : a; }
TensorView LoraTarget::bt_of(int32_t e) const noexcept { return experts > 0 ? slice(bt, e) : bt; }

Result<std::unique_ptr<LoraAdapter>> LoraAdapter::load(const std::string& path) {
    LoaderOptions options;
    options.streaming = false;  // small, and read whole at every token: never streamed
    auto file = MmapLoader::open(path, options);
    if (!file) return file.status();
    std::unique_ptr<LoraAdapter> a(new LoraAdapter());
    a->path_ = path;
    MmapLoader& f = **file;
    const auto general_type = f.get_string("general.type");
    const auto adapter_type = f.get_string("adapter.type");
    if (!adapter_type || (general_type && *general_type != "adapter")) {
        return Status(ErrorCode::InvalidModel, path + ": not a LoRA adapter (no adapter.type)");
    }
    if (*adapter_type != "lora") {
        return Status(ErrorCode::Unsupported, path + ": adapter type '" + std::string(*adapter_type) +
                                                  "' (only 'lora' is supported)");
    }
    if (f.metadata("adapter.alora.invocation_tokens") != nullptr) {
        return Status(ErrorCode::Unsupported,
                      path + ": activated LoRA (adapter.alora.invocation_tokens) is not supported");
    }
    if (const auto arch = f.get_string("general.architecture")) a->architecture_ = std::string(*arch);
    if (const auto alpha = f.get_float("adapter.lora.alpha")) a->alpha_ = static_cast<float>(*alpha);

    // Pair the halves by base name; ordered, so targets keep a stable order.
    std::map<std::string, std::pair<const TensorView*, const TensorView*>> pairs;
    for (const TensorView& t : f.tensors()) {
        const std::string_view name = t.name;
        const bool is_a = name.ends_with(kSuffixA);
        if (!is_a && !name.ends_with(kSuffixB)) {
            return Status(ErrorCode::InvalidModel,
                          path + ": tensor " + std::string(name) + " is neither a lora_a nor a lora_b");
        }
        const std::string base(name.substr(0, name.size() - kSuffixA.size()));
        auto& slot = pairs[base];
        (is_a ? slot.first : slot.second) = &t;
    }
    if (pairs.empty()) return Status(ErrorCode::InvalidModel, path + ": the adapter has no tensors");
    for (const auto& [base, pair] : pairs) {
        const auto [ta, tb] = pair;
        if (ta == nullptr || tb == nullptr) {
            return Status(ErrorCode::InvalidModel, path + ": " + base + " has only one of lora_a / lora_b");
        }
        if (base.starts_with("token_embd.")) {
            return Status(ErrorCode::Unsupported,
                          path + ": adapters of the token embeddings (" + base + ") are not supported");
        }
        const bool stacked = ta->n_dims == 3 || tb->n_dims == 3;
        if (ta->n_dims < 2 || tb->n_dims < 2 || ta->n_dims > 3 || tb->n_dims > 3 || ta->ne[3] != 1 ||
            tb->ne[3] != 1 || ta->ne[1] != tb->ne[0] || ta->ne[2] != tb->ne[2] || ta->ne[1] <= 0 ||
            ta->ne[1] > 4096) {
            return Status(ErrorCode::InvalidModel, path + ": " + base + ": lora_a and lora_b do not fit together "
                                                          "(expected ne {cols, rank[, experts]} and {rank, rows[, experts]})");
        }
        for (const TensorView* v : {ta, tb}) {
            const DTypeTraits traits = dtype_traits(v->type);
            if (!float_type(v->type) &&
                (traits.block_size <= 0 || v->ne[0] % traits.block_size != 0 || quant::is_extended(v->type))) {
                return Status(ErrorCode::Unsupported, path + ": " + std::string(v->name) + ": tensor type " +
                                                          std::string(traits.name) + " is not supported in adapters");
            }
        }
        LoraTarget target;
        target.base = base;
        target.a = *ta;
        target.rank = static_cast<int32_t>(ta->ne[1]);
        target.rows = tb->ne[1];
        target.experts = stacked ? static_cast<int32_t>(ta->ne[2]) : 0;
        // A: read in place by the matmul kernels when a float type, else
        // converted once (an adapter is small).
        if (!float_type(target.a.type)) {
            auto& copy = a->owned_.emplace_back(static_cast<size_t>(target.a.elements()) * sizeof(float));
            quant::dequantize_row(target.a.type, target.a.data, reinterpret_cast<float*>(copy.data()),
                                  target.a.elements());
            target.a.type = DType::F32;
            target.a.data = copy.data();
            target.a.nbytes = copy.size();
        }
        // Bᵀ: the update's own layout (see the header comment).
        auto& bt = a->owned_.emplace_back();
        target.bt = *tb;
        target.bt.type = transpose_b(*tb, bt);
        target.bt.ne = {tb->ne[1], tb->ne[0], tb->ne[2], 1};
        target.bt.data = bt.data();
        target.bt.nbytes = bt.size();
        a->bytes_ += target.a.nbytes + target.bt.nbytes;
        a->max_rank_ = std::max(a->max_rank_, target.rank);
        a->targets_.push_back(std::move(target));
    }
    for (size_t i = 0; i < a->targets_.size(); ++i) a->by_base_.emplace(a->targets_[i].base, i);
    // The whole file: tensors, metadata and alignment, so two files that
    // differ in any byte key their contexts apart.
    const MappedFile& mapped = f.file();
    a->id_ = xxhash64(mapped.data(), mapped.size(), static_cast<uint64_t>(mapped.size()));
    a->file_ = std::move(file).value();
    return a;
}

const LoraTarget* LoraAdapter::find(std::string_view name) const {
    const auto it = by_base_.find(name);
    return it == by_base_.end() ? nullptr : &targets_[it->second];
}

void LoraRunner::add_a_matmuls(std::span<const Item> items, std::vector<Backend::MatmulItem>& out) {
    size_t total = 0;
    t_at_.resize(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        t_at_[i] = total;
        total += static_cast<size_t>(items[i].n) * static_cast<size_t>(items[i].target->rank);
    }
    t_.resize(total);
    views_.resize(items.size());  // sized before any pointer into it is taken
    for (size_t i = 0; i < items.size(); ++i) {
        views_[i] = items[i].target->a_of(items[i].expert);
        out.push_back({&views_[i], items[i].x, t_.data() + t_at_[i], items[i].n});
    }
}

void LoraRunner::add_updates(std::span<const Item> items, ThreadPool* pool) {
    if (items.empty()) return;
    // Tasks: column chunks of each item's output. A chunk's Bᵀ columns are
    // decoded once (rank x kChunk floats), then every token's update over
    // them is summed in r order and added to y once.
    constexpr int64_t kChunk = 256;
    std::vector<int64_t> task_end(items.size());
    int64_t tasks = 0, work = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const Item& it = items[i];
        tasks += (it.target->rows + kChunk - 1) / kChunk;
        task_end[i] = tasks;
        work += static_cast<int64_t>(it.n) * it.target->rank * it.target->rows;
    }
    auto run = [&](int64_t begin, int64_t end) {
        thread_local std::vector<float> cols, acc;
        for (int64_t task = begin; task < end; ++task) {
            const auto i = static_cast<size_t>(std::upper_bound(task_end.begin(), task_end.end(), task) - task_end.begin());
            const Item& it = items[i];
            const LoraTarget& target = *it.target;
            const int64_t c0 = (task - (i == 0 ? 0 : task_end[i - 1])) * kChunk;
            const int64_t len = std::min(kChunk, target.rows - c0);
            const auto rank = static_cast<size_t>(target.rank);
            const TensorView bt = target.bt_of(it.expert);
            const size_t elem = bt.type == DType::F32 ? 4 : 2;
            cols.resize(rank * static_cast<size_t>(kChunk));
            acc.resize(static_cast<size_t>(kChunk));
            for (size_t r = 0; r < rank; ++r) {
                quant::dequantize_row(bt.type, bt.data + r * bt.row_bytes() + static_cast<size_t>(c0) * elem,
                                      cols.data() + r * kChunk, len);
            }
            const float* t_rows = t_.data() + t_at_[i];
            for (int32_t tok = 0; tok < it.n; ++tok) {
                const float* tr = t_rows + static_cast<size_t>(tok) * rank;
                std::fill(acc.begin(), acc.begin() + len, 0.0f);
                for (size_t r = 0; r < rank; ++r) {
                    const float a = it.scale * tr[r];
                    const float* col = cols.data() + r * kChunk;
                    for (int64_t c = 0; c < len; ++c) acc[static_cast<size_t>(c)] += a * col[c];
                }
                float* y = it.y + static_cast<size_t>(tok) * static_cast<size_t>(target.rows) + static_cast<size_t>(c0);
                for (int64_t c = 0; c < len; ++c) y[c] += acc[static_cast<size_t>(c)];
            }
        }
    };
    // Below this many multiply-adds (about 10 us on one core, a fork/join's
    // cost) the calling thread does it alone: a decoding step's Q/K/V of a
    // 0.5B model at rank 16 is ~18k, its gate/up ~160k. Either way gives the
    // same bits.
    constexpr int64_t kParallelWork = int64_t{1} << 15;
    if (pool == nullptr || pool->active_threads() <= 1 || work < kParallelWork || tasks == 1) {
        run(0, tasks);
    } else {
        pool->parallel_for(tasks, run);
    }
}

Status LoraRunner::apply(Backend& backend, std::span<const Item> items, ThreadPool* pool) {
    if (items.empty()) return Status::ok();
    matmuls_.clear();
    add_a_matmuls(items, matmuls_);
    LIYAB_RETURN_IF_ERROR(backend.matmul_batch(matmuls_));
    add_updates(items, pool);
    return Status::ok();
}

}  // namespace liyab
