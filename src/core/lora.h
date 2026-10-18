// Liyab — LoRA adapters over a loaded model (internal).
//
// What. A LoRA adapter changes some of a model's matrices by a low-rank
// update: W' = W + s · B·A, with A [rank x cols] and B [rows x rank], rank
// far below the matrix's size (8 to 64 against thousands). Instead of
// merging it into the weights (which would cost a copy of every adapted
// matrix per adapter and rule out switching), the forward pass computes
//     y = W·x + s · B·(A·x)
// for each adapted matrix: the base matmul as before, then two small
// matmuls (A·x has `rank` rows, B·t has the matrix's rows but only `rank`
// columns) and an add. So several adapters can be loaded next to one base
// model and the one in use chosen per generation, and a pass without an
// adapter costs exactly what it did (one branch per matmul).
//
// The file. A GGUF the common converters write from a PEFT adapter
// (general.type "adapter", adapter.type "lora", adapter.lora.alpha): for
// each adapted matrix "<name>" of the base model, "<name>.lora_a" with
// ne {cols, rank} and "<name>.lora_b" with ne {rank, rows} (A and B as PEFT
// stores them, row-major: lora_A.weight [rank, in], lora_B.weight
// [out, rank]). A stack of per-expert matrices (a mixture of experts'
// ffn_*_exps, ne[2] = experts) has stacked adapters, ne {cols, rank, experts}
// and {rank, rows, experts}: expert e's update uses slice e of each. The
// scale is s = user_scale · alpha / rank, or user_scale when the file has no
// alpha (alpha 0), with each matrix's own rank. Tensors are read in place
// from the mapping when F32, F16 or BF16; other types are converted to F32
// once at load.
//
// Not supported (load() or Transformer::set_adapter() refuse them, naming
// the tensor): adapters of the token embeddings (token_embd: they change
// the embedding lookup, not a matmul), of matrices the forward pass does not
// multiply as a whole (latent attention's per-head attn_k_b / attn_v_b), and
// activated LoRA (adapter.alora.invocation_tokens), which applies only after
// its invocation tokens.
#ifndef LIYAB_CORE_LORA_H
#define LIYAB_CORE_LORA_H

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "liyab/backend.h"
#include "liyab/mmap_loader.h"
#include "liyab/types.h"

namespace liyab {

// One adapted matrix of the base model.
struct LoraTarget {
    std::string base;       // the base model's tensor name (e.g. "blk.3.attn_q.weight")
    TensorView a;           // ne {cols, rank} (per expert: {cols, rank, experts})
    TensorView b;           // ne {rank, rows} (per expert: {rank, rows, experts})
    int32_t rank = 0;
    int32_t experts = 0;    // 0: one matrix; > 0: a stack of this many per-expert matrices
    // A and B of expert `e` (0 for a single matrix) as 2-D views.
    [[nodiscard]] TensorView a_of(int32_t e) const noexcept;
    [[nodiscard]] TensorView b_of(int32_t e) const noexcept;
};

// A LoRA adapter file, mapped read-only. Immutable after load(); its
// targets' views stay valid for the adapter's lifetime. Thread-compatible.
class LoraAdapter {
public:
    // Opens and checks `path` (see the header comment). InvalidModel for a
    // file that is not a LoRA adapter or whose pairs are inconsistent (a
    // missing half, ranks or expert counts that differ, a tensor that is
    // neither half), Unsupported for what the header comment lists, IoError
    // when it cannot be read.
    static Result<std::unique_ptr<LoraAdapter>> load(const std::string& path);

    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    // general.architecture of the base model it was trained on ("" when absent).
    [[nodiscard]] const std::string& architecture() const noexcept { return architecture_; }
    [[nodiscard]] float alpha() const noexcept { return alpha_; }
    // A hash of the whole file (xxHash64 of its bytes, mixed with its size):
    // what keys contexts computed with it.
    [[nodiscard]] uint64_t id() const noexcept { return id_; }
    // Bytes of its tensors as used (mapped, or converted to F32).
    [[nodiscard]] size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] int32_t max_rank() const noexcept { return max_rank_; }
    [[nodiscard]] const std::vector<LoraTarget>& targets() const noexcept { return targets_; }
    // The target adapting base tensor `name`, or nullptr.
    [[nodiscard]] const LoraTarget* find(std::string_view name) const;
    // s for `t` at a user scale (see the header comment).
    [[nodiscard]] float scale_of(const LoraTarget& t, float user_scale) const noexcept {
        return alpha_ != 0.0f ? user_scale * alpha_ / static_cast<float>(t.rank) : user_scale;
    }

private:
    LoraAdapter() = default;
    std::string path_;
    std::string architecture_;
    float alpha_ = 0.0f;
    uint64_t id_ = 0;
    size_t bytes_ = 0;
    int32_t max_rank_ = 0;
    std::unique_ptr<MmapLoader> file_;
    std::vector<LoraTarget> targets_;
    std::unordered_map<std::string_view, size_t> by_base_;  // keys view targets_[i].base
    std::vector<std::vector<float>> converted_;              // F32 copies of non-float tensors
};

// Adds adapter updates to matmul outputs: y += s · B·(A·x) for each item, as
// two batched matmuls on a backend (every item's A·x in one parallel pass,
// then every B·t in another) and the adds. Each output value is one dot
// product computed by one thread, so the result does not depend on the
// thread count. Keeps its scratch between calls. Thread-compatible.
class LoraRunner {
public:
    struct Item {
        const LoraTarget* target = nullptr;
        int32_t expert = 0;  // slice of a per-expert stack (0 for a matrix)
        float scale = 0.0f;  // s
        const float* x = nullptr;  // n rows of target->a.cols() values
        float* y = nullptr;        // n rows of the matrix's rows, updated in place
        int32_t n = 0;
    };
    // `backend` must read host memory (the CPU backend): the adapters live there.
    Status apply(Backend& backend, std::span<const Item> items);

private:
    std::vector<float> t_, d_;
    std::vector<TensorView> views_;
    std::vector<Backend::MatmulItem> matmuls_;
    std::vector<size_t> t_at_, d_at_;
};

}  // namespace liyab

#endif  // LIYAB_CORE_LORA_H
