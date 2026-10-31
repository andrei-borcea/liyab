// Liyab — what one decoded token reads of a model (internal).
//
// Decoding is bound by memory bandwidth: a token reads every weight once, so the bytes it reads, divided by the
// speed the device reads them at, bound the rate. This reads the bytes from a model file's tensor table, by format,
// without loading anything.
#ifndef LIYAB_CORE_MODEL_BYTES_H
#define LIYAB_CORE_MODEL_BYTES_H

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "liyab/mmap_loader.h"
#include "liyab/types.h"

namespace liyab {

// What one decoded token reads of a model, by format: every weight tensor once, except the embedding table when a
// separate output head exists (a token reads one row of it) and the experts of a mixture, of which a token reads
// the share the router picks.
struct ModelBytes {
    std::string architecture;
    uint64_t file_bytes = 0;
    int64_t blocks = 0;
    double weight_bytes = 0.0;                       // all tensors
    std::vector<std::pair<DType, double>> per_token;  // bytes read per token, by format, largest first
};

inline ModelBytes read_model_bytes(const MmapLoader& loader) {
    ModelBytes out;
    out.architecture = std::string(loader.get_string("general.architecture").value_or("?"));
    out.file_bytes = loader.file_size();
    out.blocks = loader.get_int(out.architecture + ".block_count").value_or(0);
    const double experts = static_cast<double>(loader.get_int(out.architecture + ".expert_count").value_or(0));
    const double used = static_cast<double>(loader.get_int(out.architecture + ".expert_used_count").value_or(0));
    const bool separate_head = loader.tensor("output.weight") != nullptr;
    std::vector<std::pair<DType, double>> by_format;
    for (const TensorView& t : loader.tensors()) {
        out.weight_bytes += static_cast<double>(t.nbytes);
        if (t.name == "token_embd.weight" && separate_head) continue;
        double bytes = static_cast<double>(t.nbytes);
        if (t.name.find("_exps") != std::string_view::npos && experts > 0 && used > 0) bytes *= used / experts;
        const auto it = std::find_if(by_format.begin(), by_format.end(), [&](const auto& e) { return e.first == t.type; });
        if (it == by_format.end()) by_format.emplace_back(t.type, bytes);
        else it->second += bytes;
    }
    std::sort(by_format.begin(), by_format.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    out.per_token = std::move(by_format);
    return out;
}

}  // namespace liyab

#endif  // LIYAB_CORE_MODEL_BYTES_H
