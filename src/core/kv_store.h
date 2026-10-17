// Liyab — content-addressed store of KV chunks: the engine's prefix cache (internal).
//
// What it keeps. Contexts the engine sets aside (a conversation another
// prompt replaced, one an embedding call interrupted, the context left when
// the engine is destroyed) are cut into chunks of `chunk_tokens` positions,
// one page of the paged KV cache each (KvCache::read_rows). A chunk holds the
// KV rows of its positions and the tokens they encode; the last chunk of a
// context may be partial. Each chunk is keyed by a hash chained over every
// token before it and its own (kv_chain_key), rooted in a seed that covers
// what the bytes depend on: the model file, the cache layout and type, the
// key scales the keys were stored with, the kernel set and the backends that
// computed them. So identical prefixes (a system prompt, a tool block, the
// start of a conversation) are stored once whatever context they came from,
// and a chunk is never read back for another model or other settings.
//
// What a prompt gets back (plan()). The chunks of the prompt's tokens are
// looked up in order while they are stored; at the first one that is not,
// the stored chunks continuing the same prefix (siblings of that chunk) are
// compared with the prompt token by token, and the one sharing the most is
// used up to where they diverge. On attention-only models every position is
// a restore point. Recurrent (DeltaNet) states cannot be cut into chunks, so
// hybrid models keep the state images the prefix cache always kept (the
// end of the context, the end of its last prompt, the oldest snapshot: see
// Transformer::kept_states()), each keyed by the prefix it follows and
// attached to a chunk its position ends in. Such a chunk may be one of
// several (the system prompt's end lies inside the first chunk of every
// conversation that follows it): the image stays with the most recently used
// of them and moves to another when that one goes. A prompt is restored only
// up to the deepest stored state its own tokens lead to, and the images below
// that state come back as the model's snapshots, so a restored context
// rewinds where the live one would have.
//
// Stored contexts (records). Each context set aside also keeps a record: its
// tokens, the text they encode (the engine matches a prompt's text against
// it, since a reply's tokens need not be its text's encoding) and the states
// it refers to. A record ends at its last chunk and lives as long as that
// chunk (a state it refers to may go first: the record then restores less on
// a hybrid model). Setting a conversation aside again replaces its older
// records (any whose tokens it starts with); a state no record refers to any
// more goes, and so does the partial chunk such a record ended at once
// nothing else holds it.
//
// Tiers and caps. RAM (Config::ram_bytes) and storage (Config::dir,
// disk_bytes), each capped and never above its cap: room is made before
// anything is added. Every chunk and state has a place in one LRU order;
// using a context (setting it aside, restoring it) moves its chunks to the
// front, a chunk ahead of every chunk after it, so a chunk is never older than
// what continues it. RAM holds the most recently used; what it cannot hold
// moves to storage, and what storage cannot hold goes: a chunk with every
// chunk after it (they would be unreachable) and the records ending there,
// its states moving to another chunk that holds their prefix or going with
// it; a state image (62.8 MiB on a 35B against 0.7 MiB for a chunk) on its
// own, at its place in the order, next to the chunk it was used with. With
// one tier alone, that tier is the whole order. Reference counts: a chunk
// counts the chunks that continue it, a state the records that use it.
//
// Storage. One file per chunk (kv-<key>.chunk), state (kv-<key>.state) and
// record (kv-<leaf key>.ctx, written once its last chunk is on storage), each
// a 128-byte header (format version, kind, encoding, keys, seed, sizes, a
// hash of the header and metadata, a hash of the payload), the metadata
// (a chunk's tokens; a state's tokens since its chunk boundary; a record's
// tokens, text and states) and the payload. Files
// are written to a temporary name, then renamed: a crash leaves the old file
// or none, never a partial one under a valid name. The storage tier persists
// across engines and processes: a later engine with the same seed finds its
// chunks (LRU order from the files' modification times, which use updates),
// others are kept and counted against the cap. When it opens the directory,
// the store removes what it cannot trust: temporary files, files of another
// format version or kind, damaged headers or sizes, chunks whose chain is
// broken, states and records whose chunks are gone, prefix-cache files of
// older builds (ctx-*.state). A payload whose hash does not match is removed
// when it is read, and that read fails (the engine then processes the prompt).
// One store at a time uses a directory: it holds an OS lock on kv.lock there
// (os::FileLock) while open; another engine, in this process or another,
// that finds it held runs without the storage tier.
//
// Extension points (later branches build on these, nothing else changes):
//  * Compressed chunks on storage (feat/kv-compress): FileHeader::encoding
//    names how a chunk's payload is stored (KvEncoding); `raw_bytes` is its
//    decoded size and every cap counts the stored size (Item::file_bytes).
//    Callers only ever see raw images (put()'s rows, read_chunk()), so an
//    encoding is added in write_item_file() (encoding), read_chunk()
//    (decoding into the caller's image) and scan() (accepting it); a file of
//    an encoding a build does not know is removed when the directory is
//    opened, never misread.
//  * Chunks reused away from their prefix (feat/kv-reuse): every chunk also
//    carries a position-independent content key (kv_content_key: the seed
//    and its own tokens, not what precedes them) and its start position,
//    indexed by chunks_with_content(). Reusing one elsewhere needs its keys
//    rotated to the new positions and attention that tolerates the missing
//    cross-chunk context; until then chunks are only restored where they
//    were stored.
//
// Thread-compatible: the engine calls it under its own lock, one call at a time.
#ifndef LIYAB_CORE_KV_STORE_H
#define LIYAB_CORE_KV_STORE_H

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/os.h"
#include "liyab/types.h"

namespace liyab {

// A 128-bit content key: two xxHash64 sums with different seeds.
struct KvKey {
    uint64_t hi = 0;
    uint64_t lo = 0;
    friend bool operator==(const KvKey&, const KvKey&) = default;
};

struct KvKeyHash {
    size_t operator()(const KvKey& k) const noexcept { return static_cast<size_t>(k.hi ^ (k.lo * 0x9E3779B97F4A7C15ULL)); }
};

// The root of a store's keys: a hash of everything the stored bytes depend on.
KvKey kv_seed_key(std::span<const uint64_t> parts) noexcept;
// The key of a chunk holding `tokens` after the chunk keyed `parent` (the
// seed for the first chunk): chained, so it covers every token before.
KvKey kv_chain_key(const KvKey& parent, std::span<const int32_t> tokens) noexcept;
// A chunk's position-independent key: the seed and its own tokens only.
KvKey kv_content_key(const KvKey& seed, std::span<const int32_t> tokens) noexcept;

// Removes the snapshot files of KV dedup, the experimental persistent prefix
// cache this store replaced (<16 hex digits>.lykv and their temporary
// files), from `dir`, which older builds were given for it; other files stay.
// Returns how many were removed.
int32_t remove_kv_dedup_files(const std::string& dir);

// How a chunk's payload is stored on storage (see the header comment).
enum class KvEncoding : uint32_t {
    Raw = 0,  // the rows as KvCache::read_rows() writes them
};

class KvStore {
public:
    struct Config {
        KvKey seed;                 // see kv_seed_key(); chunks of other seeds are never returned
        int32_t chunk_tokens = 64;  // positions per chunk: one KV page
        size_t chunk_bytes = 0;     // bytes of a whole chunk's rows (KvCache::rows_bytes(chunk_tokens))
        size_t state_floats = 0;    // values of one recurrent-state image; 0: attention-only model
        size_t ram_bytes = 0;       // RAM cap; 0: no RAM tier
        std::string dir;            // storage tier directory (created when missing); empty: none
        size_t disk_bytes = 0;      // storage cap; 0: no storage tier
    };

    // Opens the tiers `config` asks for. Never fails: a directory that cannot
    // be created or is in use by another store leaves the storage tier off
    // (logged). nullptr when no tier is on.
    static std::unique_ptr<KvStore> open(const Config& config);
    ~KvStore();
    KvStore(const KvStore&) = delete;
    KvStore& operator=(const KvStore&) = delete;

    [[nodiscard]] int32_t chunk_tokens() const noexcept { return config_.chunk_tokens; }
    [[nodiscard]] bool has_storage() const noexcept { return disk_on_; }

    // A recurrent-state image of a context: its position and pinned flag
    // (Transformer::KeptState), with its key once stored.
    struct StateMark {
        int32_t pos = 0;
        bool pinned = false;
    };
    struct RecordState {
        int32_t pos = 0;
        bool pinned = false;
        KvKey key;
    };
    // A stored context (see the header comment). Valid until the next call
    // that changes the store (put, used, read_*: a damaged file is removed).
    struct Record {
        KvKey leaf;                       // its last chunk
        std::vector<int32_t> tokens;
        std::string text;                 // what its first text_tokens tokens encode (0: not known)
        size_t text_tokens = 0;
        std::vector<RecordState> states;  // ascending; empty on attention-only models
    };

    // A context to set aside: its tokens and text, a source of each chunk's
    // rows and, on hybrid models, of the state images to keep.
    struct ContextSource {
        std::span<const int32_t> tokens;
        std::string_view text;
        size_t text_tokens = 0;
        // Writes the rows of positions [chunk * chunk_tokens, + n) to `out`
        // (KvCache::read_rows of logical page `chunk`).
        std::function<Status(int32_t chunk, int32_t n, uint8_t* out)> rows;
        // Ascending, the last at tokens.size() (Transformer::kept_states()).
        std::vector<StateMark> states;
        // Writes the state image at `pos` (state_floats values) to `out`.
        std::function<Status(int32_t pos, float* out)> state;
    };
    struct PutResult {
        bool whole = false;          // every chunk and state is stored, with the record
        int32_t chunks = 0;          // chunks of the context
        int32_t chunks_written = 0;  // new ones
        int32_t chunks_reused = 0;   // stored already (a shared prefix, an earlier copy)
        int32_t states_written = 0;
        size_t bytes_ram = 0;        // added to RAM
        size_t bytes_disk = 0;       // written to storage (moves from RAM included)
    };
    // Stores `source` and makes it the most recently used context, replacing
    // older records of the same conversation. With `storage_only` (the engine
    // is being destroyed) everything it needs, chunks already in RAM
    // included, goes to storage, so a later engine finds it; nothing without
    // a storage tier. What does not fit is not kept: the chunks that fit (its
    // first ones) are, the record only when every chunk is.
    PutResult put(const ContextSource& source, bool storage_only = false);

    // What can be restored for a prompt (see the header comment).
    struct Plan {
        int32_t positions = 0;              // the prompt's tokens [0, positions) (0: nothing)
        std::vector<KvKey> chunks;          // chunk i holds positions from i * chunk_tokens
        std::vector<int32_t> chunk_sizes;   // its tokens (the last may hold more than `positions` needs)
        KvKey state;                        // hybrid: the state image at `positions`
        bool state_pinned = false;          // its pinned flag (the snapshot taken there)
        std::vector<RecordState> snapshots;  // hybrid: images below `positions`, restored as snapshots
        const Record* record = nullptr;     // a record whose tokens start with those positions (text_tokens > 0), or nullptr
    };
    // The longest restorable prefix of `tokens`, at most `limit` positions.
    [[nodiscard]] Plan plan(std::span<const int32_t> tokens, size_t limit) const;
    // Marks a plan's chunks and states used (the front of the LRU order).
    void used(const Plan& plan);
    // A chunk's whole image (rows_bytes of its token count) or a state
    // image, into `out` of exactly that size. IoError for a damaged or
    // unreadable file, which is removed (with what depends on it);
    // InvalidArgument for an unknown key or a wrong size.
    Status read_chunk(const KvKey& key, std::span<uint8_t> out);
    Status read_state(const KvKey& key, std::span<float> out);
    // The keys of the chunks holding `tokens` (all stored: else empty).
    [[nodiscard]] std::vector<KvKey> chunk_path(std::span<const int32_t> tokens) const;

    // The stored contexts of this seed.
    [[nodiscard]] std::vector<const Record*> records() const;
    // The longest record whose text `text` starts with (text_tokens > 0), or nullptr.
    [[nodiscard]] const Record* record_for_text(std::string_view text) const;

    // Chunks of this seed holding exactly these tokens, wherever they start
    // (kv_content_key); for reuse away from the prefix (feat/kv-reuse).
    struct ChunkInfo {
        KvKey key;
        KvKey parent;
        int32_t start = 0;
        int32_t tokens = 0;
        bool in_ram = false;
    };
    [[nodiscard]] std::vector<ChunkInfo> chunks_with_content(const KvKey& content) const;

    struct Stats {
        size_t ram_bytes = 0;      // chunk and state payloads in RAM
        size_t ram_cap = 0;
        size_t disk_bytes = 0;     // every store file in the directory, other seeds' included
        size_t disk_cap = 0;
        int32_t chunks_ram = 0;    // this seed's
        int32_t chunks_disk = 0;
        int32_t states = 0;
        int32_t records = 0;
        int32_t foreign_files = 0;  // other seeds' files kept on storage
        int32_t removed_files = 0;  // untrusted files removed when the directory was opened
    };
    [[nodiscard]] Stats stats() const;

private:
    explicit KvStore(const Config& config);
    enum class Kind : uint8_t { Chunk, State };
    // A chunk or a state: what the LRU orders.
    struct Item {
        KvKey key;
        KvKey seed;
        uint64_t clock = 0;
        bool placed = false;               // in a tier: counted against its cap, in its LRU
        bool on_disk = false;
        size_t payload_bytes = 0;          // raw payload size
        std::unique_ptr<uint8_t[]> data;   // in RAM: the payload
        size_t file_bytes = 0;             // on storage: the file's size,
        size_t payload_offset = 0;         //   where the payload starts in it
        uint64_t payload_hash = 0;         //   and its hash (FileHeader::payload_hash)
        int64_t mtime = 0;                 // when opened from storage: for the LRU order
    };
    struct Chunk : Item {
        KvKey parent;
        KvKey content;
        int32_t start = 0;
        std::vector<int32_t> tokens;
        std::vector<KvKey> kids;    // chunks continuing it
        std::vector<KvKey> states;  // states attached to it
    };
    struct State : Item {
        int32_t pos = 0;
        KvKey parent;                 // the prefix before the chunk its position ends in (the seed for the first)
        std::vector<int32_t> tokens;  // that chunk's tokens up to pos
        KvKey chunk;                  // the chunk holding that prefix it is attached to
        int32_t refs = 0;             // records using it
    };
    struct RecordEntry {
        Record record;
        KvKey seed;
        bool on_disk = false;
        size_t file_bytes = 0;
    };

    [[nodiscard]] bool ours(const Item& item) const noexcept { return item.seed == config_.seed; }
    [[nodiscard]] size_t rows_bytes(int32_t n) const noexcept;
    [[nodiscard]] std::string path_of(const KvKey& key, const char* ext) const;
    // The key of the prefix of tokens ending at `pos`, given the keys of the
    // chunks before it: a chunk's own key at a chunk boundary.
    [[nodiscard]] KvKey prefix_key(std::span<const int32_t> tokens, std::span<const KvKey> chunk_keys,
                                   int32_t pos) const noexcept;
    Item* item(Kind kind, const KvKey& key);
    void set_clock(Kind kind, Item& it, uint64_t clock);
    void touch_files(const std::vector<std::pair<Kind, KvKey>>& newest_first);

    // Placement and caps (see the header comment). make_room() frees `bytes`
    // in a tier by moving out (RAM) or evicting (storage) items older than
    // `below`; false when that is not enough.
    bool make_room(bool disk, size_t bytes, uint64_t below);
    // RAM -> storage. When it cannot be written: evicted when `or_evict`,
    // else left in RAM. True when it is on storage.
    bool demote(Kind kind, const KvKey& key, bool or_evict);
    bool place(Kind kind, Item& it, bool storage_only);  // a new item: RAM, else storage
    void evict(Kind kind, const KvKey& key);             // at chunk granularity
    void drop_chunk(const KvKey& key);                   // with every chunk after it
    void drop_one_chunk(const KvKey& key);
    // Attaches a state to `chunk` (one holding its prefix).
    void attach(State& state, const KvKey& chunk);
    // The most recently used chunk other than `except` holding a state's
    // prefix (kids of its parent starting with its tokens), or nullptr.
    [[nodiscard]] const Chunk* holder_of(const State& state, const KvKey& except) const;
    void drop_state(const KvKey& key);
    void drop_record(const KvKey& leaf, bool collect);
    void release(Kind kind, Item& it);  // frees its payload or file, out of its tier
    [[nodiscard]] size_t file_size(Kind kind, const Item& it) const noexcept;
    bool write_item_file(Kind kind, Item& it);  // the file of an item with its payload in `data`
    bool write_record_file(RecordEntry& entry);

    void scan();  // indexes the directory (see the header comment)

    Config config_;
    bool disk_on_ = false;
    os::FileLock lock_;
    uint64_t clock_ = 0;
    size_t ram_used_ = 0;
    size_t disk_used_ = 0;
    int32_t removed_files_ = 0;
    std::unordered_map<KvKey, Chunk, KvKeyHash> chunks_;
    std::unordered_map<KvKey, State, KvKeyHash> states_;
    std::unordered_map<KvKey, RecordEntry, KvKeyHash> records_;  // by leaf key
    std::vector<KvKey> roots_;                                   // this seed's first chunks
    std::unordered_multimap<KvKey, KvKey, KvKeyHash> by_content_;
    std::map<uint64_t, std::pair<Kind, KvKey>> lru_[2];  // [0] RAM, [1] storage; oldest first
    std::vector<uint8_t> scratch_;
};

}  // namespace liyab

#endif  // LIYAB_CORE_KV_STORE_H
