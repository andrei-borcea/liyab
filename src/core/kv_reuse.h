// Liyab — KV reused away from its prefix: segment reuse (internal).
//
// What. A caller marks segments of a prompt (a system block, tools,
// documents, tool results, conversation turns: PromptSegment). The KV rows
// a segment's tokens get in one prompt are kept in the engine's KV store (a
// segment is a context of its own there: core/kv_store.h), and a later
// prompt that holds the same tokens anywhere takes them from the store
// instead of processing them again. Exact prefix reuse comes first and
// stays lossless: only what follows the part a prompt shares with the live
// context or a stored context is taken from segments.
//
// Why it is lossy, and what repairs it. A token's keys and values depend on
// every token before it, and a segment's rows were computed after another
// prefix (other documents, in another order). Two repairs:
//  * Positions. RoPE turns each key by an angle proportional to its
//    position, so rows computed at other positions are turned to the new
//    ones (Transformer::move_keys): exact up to rounding.
//  * Attention to what now precedes them, by selective recompute: some of
//    the reused tokens are processed again, block by block, with attention
//    to everything before them (Transformer::run_block_at); the others keep
//    their reused rows. Which ones: every reused token runs the first
//    blocks (up to select_block) with the tokens around it, and those whose
//    keys and values there moved the most from their reused rows go on
//    through the rest of the model (the deviation criterion; KV rows that
//    move early keep moving through the later blocks). The fraction is not
//    a constant: calibrate_reuse() measures, per model, the smallest one
//    that keeps the next-token distributions of a fixed RAG-shaped prompt
//    within a KL target of a full prefill.
//
// What a reused segment's tokens cost: the blocks up to select_block, then
// the fraction of them through the remaining blocks; the tokens between
// reused segments (text not marked, or not stored yet) are processed as
// usual. Everything runs block by block over the whole span between the
// exact prefix and the last reused token, in passes of `chunk` tokens in
// order, so a token always sees every position before it at each block.
//
// Not for hybrid models (Transformer::reuse_support()): their recurrent
// blocks need every token in order, so reusing the attention blocks' rows
// would only skip their key and value projections.
#ifndef LIYAB_CORE_KV_REUSE_H
#define LIYAB_CORE_KV_REUSE_H

#include <atomic>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "core/kv_store.h"
#include "core/tokenizer.h"
#include "core/transformer.h"
#include "liyab/engine.h"

namespace liyab {

class ThreadPool;

// One marked segment of a prompt's tokens: [begin, end).
struct TokenSegment {
    int32_t begin = 0;
    int32_t end = 0;
};

// Whether `segments` are non-empty, ascending, apart and inside a prompt of
// `size` units: bytes of `text` when it is given (then a boundary must not
// cut a UTF-8 character), else tokens. InvalidArgument naming the first
// that is not.
Status check_segments(std::span<const PromptSegment> segments, size_t size, std::string_view text = {});

// Encodes text[from, end) after `tokens`, which spell text[0, from) (the
// context's known text, or nothing), split at the boundaries of the marked
// byte ranges inside it. Each piece is encoded on its own, so a segment's
// tokens are the same wherever it appears: Tokenizer::encode() for the
// first piece of an empty `tokens` (with BOS when `bos`), else
// Tokenizer::encode_continuation() after the token before it; a segment that
// starts the text gets BOS (when `bos` and the text does not spell it)
// before it, not in it. Appends the tokens, and the token ranges of the
// segments that start at or after `from` to `out` (one that starts before it
// is plain text from `from` on). A boundary where encoding the whole text
// would merge two tokens across it splits them instead. InvalidArgument for
// segments that are empty, overlap, are not ascending, end past the text or
// cut a UTF-8 character.
Status encode_segmented(const Tokenizer& tokenizer, std::string_view text, size_t from,
                        std::span<const PromptSegment> segments, bool bos, std::vector<int32_t>& tokens,
                        std::vector<TokenSegment>& out);

// How the reused tokens to recompute are chosen.
enum class ReuseCriterion : int32_t {
    Deviation,  // every reused token runs blocks [0, select_block]; those whose rows moved most go on
    First,      // the first `fraction` of each reused segment's tokens (no measurement)
    Random,     // a random `fraction` of the reused tokens (seeded; a baseline)
};

struct ReuseSettings {
    // Share of the reused tokens processed again through every block.
    float fraction = 0.15f;
    ReuseCriterion criterion = ReuseCriterion::Deviation;
    // Deviation: the block whose rows are compared (the first one whose keys
    // depend on other tokens is 1: block 0's are a function of the token and
    // its position). Clamped to the model's last block.
    int32_t select_block = 1;
    // Deviation: each row's change relative to its size (true) or absolute.
    bool relative = true;
    // Deviation: blocks after select_block that narrow the selection again
    // (gradual filtering): at select_block the share kept is fraction x 2^refine,
    // halved at each of the next `refine` blocks, measured anew at each.
    int32_t refine = 0;
    // The first tokens of each reused segment, recomputed whatever the
    // criterion picks (they lean the most on what precedes them).
    int32_t first_tokens = 0;
    // Segments shorter than this are processed, not reused or kept.
    int32_t min_tokens = 16;
    uint64_t seed = 1;  // Random
};

struct ReuseResult {
    int32_t end = 0;           // the context now holds [0, end) (the exact prefix when nothing was reused)
    int32_t reused = 0;        // positions whose rows came from stored segments
    int32_t recomputed = 0;    // of them, processed again through every block
    int32_t computed = 0;      // positions in [exact, end) processed as usual (not reused)
    int32_t segments = 0;      // segments reused
    int32_t first_reused = -1; // the first position whose rows are approximate; -1: none
    double rows_ms = 0.0;      // reading and moving the stored rows
    double blend_ms = 0.0;     // the block-by-block pass
};

// Reuses stored segments of `tokens` (the prompt; `segments` ascending) after
// its first `exact` positions, which the context holds (n_past() == exact),
// and before `limit` (the tokens the caller processes itself start there;
// for generation, the last prompt token). For each segment of at least
// min_tokens that reaches past `exact`, the store's copy computed at the
// same position or else the most recently used one; its rows moved to the
// segment's positions, then the span [exact, end) run block by block (see
// the header comment), `end` being the end of the last reused range. On
// return the context holds [0, end) (n_past() == end). Nothing reused (no
// copy stored): end == exact and nothing changed. A stored file found
// damaged drops that segment (processed instead). On any other failure the
// context is cut back to `exact` and the error returned (the caller
// processes the prompt as usual); `stop` (optional), when set, stops the
// pass between two blocks the same way (Cancelled). Unsupported per
// reuse_support().
Result<ReuseResult> reuse_segments(Transformer& model, KvStore& store, std::span<const int32_t> tokens,
                                   std::span<const TokenSegment> segments, int32_t exact, int32_t limit,
                                   const ReuseSettings& settings, const Route& route, ThreadPool& pool,
                                   int32_t chunk, const std::atomic<bool>* stop = nullptr);

// Stores the segments of the context (`tokens`, whose first n_past() rows
// the model holds) that are at least `min_tokens` long, lie within those
// rows and have no stored copy: each as kv_segment_tokens(segment, its
// first position), its rows as the context holds them. Returns how many
// were stored whole.
int32_t keep_segments(const Transformer& model, KvStore& store, std::span<const int32_t> tokens,
                      std::span<const TokenSegment> segments, int32_t min_tokens);

// ---- Calibration -------------------------------------------------------------
// The recompute fraction for one model and cache, measured instead of set:
// on a fixed RAG-shaped text (reuse_calibration_text: a system line, four
// documents, a question), the documents are stored from one order, then two
// other orders are answered both ways, with a full prefill and with the
// documents reused at each fraction of kReuseFractions in turn. The score is
// the mean KL divergence of the next-token distributions, KL(full ||
// reuse), at the first answer token and the next kCalibrationAnswer tokens
// of the full prefill's greedy answer (teacher-forced). The fraction kept is
// the smallest whose score is within `max_kl`; none: reuse would need about
// as much work as processing the documents, and is not worth it.
inline constexpr float kReuseFractions[] = {0.05f, 0.1f, 0.15f, 0.2f, 0.3f, 0.45f};
inline constexpr int32_t kCalibrationAnswer = 16;
// Changing the texts, the grid or the procedure changes every model's
// result: bump it so results kept in a cache directory are measured again.
inline constexpr uint32_t kReuseCalibrationVersion = 1;

struct ReuseCalibration {
    bool met = false;        // a fraction on the grid met the target
    float fraction = 0.0f;   // that fraction (met), else the last one tried
    double kl = 0.0;         // its score
    double top1 = 0.0;       // its top-1 agreement with the full prefill, in [0, 1]
    double ms = 0.0;         // time the calibration took
};

// Calibrates `settings` (its fraction is what is searched; the criterion and
// the rest are kept) for `model` against `max_kl`. Uses the model's context
// (left empty) and a store of its own in RAM. Unsupported per
// reuse_support(); other errors from the passes.
Result<ReuseCalibration> calibrate_reuse(Transformer& model, const Tokenizer& tokenizer, const ReuseSettings& settings,
                                         double max_kl, const Route& route, ThreadPool& pool, int32_t chunk);

}  // namespace liyab

#endif  // LIYAB_CORE_KV_REUSE_H
