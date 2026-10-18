// Liyab — LoRA adapters over a loaded model (see lora.h).
#include "core/lora.h"

#include <algorithm>
#include <map>

#include "core/quant.h"

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

}  // namespace

TensorView LoraTarget::a_of(int32_t e) const noexcept { return experts > 0 ? slice(a, e) : a; }
TensorView LoraTarget::b_of(int32_t e) const noexcept { return experts > 0 ? slice(b, e) : b; }

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
        LoraTarget target;
        target.base = base;
        target.a = *ta;
        target.b = *tb;
        target.rank = static_cast<int32_t>(ta->ne[1]);
        target.experts = stacked ? static_cast<int32_t>(ta->ne[2]) : 0;
        // Types the matmul kernels read with a float input stay mapped; the
        // others are converted once (an adapter is small).
        for (TensorView* v : {&target.a, &target.b}) {
            if (float_type(v->type)) {
                a->bytes_ += v->nbytes;
                continue;
            }
            const DTypeTraits traits = dtype_traits(v->type);
            if (traits.block_size <= 0 || v->ne[0] % traits.block_size != 0 || quant::is_extended(v->type)) {
                return Status(ErrorCode::Unsupported, path + ": " + std::string(v->name) + ": tensor type " +
                                                          std::string(traits.name) + " is not supported in adapters");
            }
            auto& copy = a->converted_.emplace_back(static_cast<size_t>(v->elements()));
            quant::dequantize_row(v->type, v->data, copy.data(), v->elements());
            v->type = DType::F32;
            v->data = reinterpret_cast<const uint8_t*>(copy.data());
            v->nbytes = copy.size() * sizeof(float);
            a->bytes_ += v->nbytes;
        }
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

Status LoraRunner::apply(Backend& backend, std::span<const Item> items) {
    if (items.empty()) return Status::ok();
    // Scratch: t (n x rank) and the update d (n x rows) of each item.
    size_t t_total = 0, d_total = 0;
    t_at_.resize(items.size());
    d_at_.resize(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        const Item& it = items[i];
        t_at_[i] = t_total;
        d_at_[i] = d_total;
        t_total += static_cast<size_t>(it.n) * static_cast<size_t>(it.target->rank);
        d_total += static_cast<size_t>(it.n) * static_cast<size_t>(it.target->b.ne[1]);
    }
    t_.resize(t_total);
    d_.resize(d_total);
    views_.resize(2 * items.size());
    matmuls_.resize(items.size());
    // t = A·x for every item, in one pass.
    for (size_t i = 0; i < items.size(); ++i) {
        const Item& it = items[i];
        views_[2 * i] = it.target->a_of(it.expert);
        views_[2 * i + 1] = it.target->b_of(it.expert);
        matmuls_[i] = {&views_[2 * i], it.x, t_.data() + t_at_[i], it.n};
    }
    LIYAB_RETURN_IF_ERROR(backend.matmul_batch(matmuls_));
    // The scale on t (n x rank values) rather than on the update (n x rows).
    for (size_t i = 0; i < items.size(); ++i) {
        const Item& it = items[i];
        float* t = t_.data() + t_at_[i];
        const size_t count = static_cast<size_t>(it.n) * static_cast<size_t>(it.target->rank);
        for (size_t j = 0; j < count; ++j) t[j] *= it.scale;
    }
    // d = B·t for every item, in one pass; then y += d.
    for (size_t i = 0; i < items.size(); ++i) {
        matmuls_[i] = {&views_[2 * i + 1], t_.data() + t_at_[i], d_.data() + d_at_[i], items[i].n};
    }
    LIYAB_RETURN_IF_ERROR(backend.matmul_batch(matmuls_));
    for (size_t i = 0; i < items.size(); ++i) {
        const Item& it = items[i];
        const float* d = d_.data() + d_at_[i];
        const size_t count = static_cast<size_t>(it.n) * static_cast<size_t>(it.target->b.ne[1]);
        for (size_t j = 0; j < count; ++j) it.y[j] += d[j];
    }
    return Status::ok();
}

}  // namespace liyab
