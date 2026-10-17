#include "core/kv_store.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <tuple>

#include "core/log.h"
#include "liyab/mmap_loader.h"  // xxhash64

namespace liyab {

namespace {

// Seeds of the two halves of a key, so they are two different hashes.
constexpr uint64_t kHalfHi = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kHalfLo = 0xC2B2AE3D27D4EB4FULL;
// Separates content keys from chain keys (a first chunk's would be equal).
constexpr uint64_t kContentDomain = 0x165667B19E3779F9ULL;

KvKey hash_after(const KvKey& parent, const void* data, size_t bytes) noexcept {
    // Each half hashes the whole parent key, then the data.
    return {xxhash64(data, bytes, xxhash64(&parent, sizeof parent, kHalfHi)),
            xxhash64(data, bytes, xxhash64(&parent, sizeof parent, kHalfLo))};
}

// --- Files -------------------------------------------------------------------------
// Little-endian hosts only (arm64, x86-64), as the engine's state files.
constexpr char kMagic[4] = {'L', 'Y', 'K', 'S'};
constexpr uint32_t kFormat = 1;
constexpr uint32_t kFileChunk = 1, kFileState = 2, kFileRecord = 3;
constexpr uint64_t kMaxMeta = uint64_t{1} << 30;
constexpr const char* kLockName = "kv.lock";

struct FileHeader {
    char magic[4];
    uint32_t version;
    uint32_t kind;
    uint32_t encoding;  // KvEncoding (chunks); 0 otherwise
    uint64_t key[2];
    uint64_t link[2];     // chunk: its parent (the seed for a first chunk); state: the prefix before its chunk; record: its last chunk
    uint64_t seed[2];
    uint64_t content[2];  // chunk: its content key
    int32_t start;        // chunk: first position; state: its position
    int32_t count;        // chunk, record: tokens
    uint64_t meta_bytes;  // after the header: chunk: tokens; state: its chunk's tokens up to it; record: see record_meta()
    uint64_t payload_bytes;
    uint64_t raw_bytes;   // the payload decoded (== payload_bytes for KvEncoding::Raw)
    uint64_t payload_hash;
    uint64_t header_hash;  // of the bytes above and the metadata
};
static_assert(sizeof(FileHeader) == 128, "the store's file header is 128 bytes");

void put_key(uint64_t (&to)[2], const KvKey& k) noexcept {
    to[0] = k.hi;
    to[1] = k.lo;
}
KvKey get_key(const uint64_t (&from)[2]) noexcept { return {from[0], from[1]}; }

uint64_t header_hash(const FileHeader& h, const uint8_t* meta, size_t meta_bytes) noexcept {
    return xxhash64(meta, meta_bytes, xxhash64(&h, offsetof(FileHeader, header_hash)));
}

std::string hex(const KvKey& k) {
    char name[33];
    std::snprintf(name, sizeof name, "%016llx%016llx", static_cast<unsigned long long>(k.hi),
                  static_cast<unsigned long long>(k.lo));
    return name;
}

bool parse_hex(std::string_view s, KvKey& k) {
    if (s.size() != 32) return false;
    uint64_t parts[2] = {0, 0};
    for (size_t i = 0; i < 32; ++i) {
        const char c = s[i];
        const int v = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (v < 0) return false;
        parts[i / 16] = parts[i / 16] << 4 | static_cast<uint64_t>(v);
    }
    k = {parts[0], parts[1]};
    return true;
}

// Writes header, metadata and payload to `path` through a temporary file.
bool write_file(const std::string& path, const FileHeader& h, std::span<const uint8_t> meta,
                std::span<const uint8_t> payload) {
    const std::string tmp = path + ".tmp";
    std::FILE* f = os::open_file(tmp, "wb");
    if (f == nullptr) return false;
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1;
    ok = ok && (meta.empty() || std::fwrite(meta.data(), 1, meta.size(), f) == meta.size());
    ok = ok && (payload.empty() || std::fwrite(payload.data(), 1, payload.size(), f) == payload.size());
    ok = std::fflush(f) == 0 && ok;
    ok = std::fclose(f) == 0 && ok;
    if (!ok || !os::replace_file(tmp, path)) {
        os::remove_file(tmp);
        return false;
    }
    return true;
}

// Reads `out.size()` payload bytes at `offset` of `path` and checks their hash.
bool read_payload(const std::string& path, size_t offset, uint64_t hash, std::span<uint8_t> out) {
    std::FILE* f = os::open_file(path, "rb");
    if (f == nullptr) return false;
    bool ok = std::fseek(f, static_cast<long>(offset), SEEK_SET) == 0 &&
              std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    return ok && xxhash64(out.data(), out.size()) == hash;
}

void touch_file(const std::string& path) {
    std::error_code ec;
    std::filesystem::last_write_time(os::utf8_path(path), std::filesystem::file_time_type::clock::now(), ec);
}

template <typename T>
void append(std::vector<uint8_t>& out, const T& v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof v);
}

// A record's metadata: tokens (count x int32), text_tokens, text length,
// state count (uint64 each), the states (pos, pinned as int32, key), the text.
std::vector<uint8_t> record_meta(const KvStore::Record& r) {
    std::vector<uint8_t> meta;
    meta.reserve(r.tokens.size() * sizeof(int32_t) + 24 + r.states.size() * 24 + r.text.size());
    const auto* t = reinterpret_cast<const uint8_t*>(r.tokens.data());
    meta.insert(meta.end(), t, t + r.tokens.size() * sizeof(int32_t));
    append(meta, static_cast<uint64_t>(r.text_tokens));
    append(meta, static_cast<uint64_t>(r.text.size()));
    append(meta, static_cast<uint64_t>(r.states.size()));
    for (const KvStore::RecordState& s : r.states) {
        append(meta, s.pos);
        append(meta, static_cast<int32_t>(s.pinned));
        append(meta, s.key.hi);
        append(meta, s.key.lo);
    }
    meta.insert(meta.end(), r.text.begin(), r.text.end());
    return meta;
}

bool parse_record_meta(std::span<const uint8_t> meta, int32_t count, KvStore::Record& r) {
    size_t at = 0;
    auto take = [&](void* to, size_t n) {
        if (n > meta.size() - at) return false;
        std::memcpy(to, meta.data() + at, n);
        at += n;
        return true;
    };
    if (count < 0) return false;
    r.tokens.resize(static_cast<size_t>(count));
    uint64_t text_tokens = 0, text_bytes = 0, states = 0;
    if (!take(r.tokens.data(), r.tokens.size() * sizeof(int32_t)) || !take(&text_tokens, 8) || !take(&text_bytes, 8) ||
        !take(&states, 8) || text_tokens > static_cast<uint64_t>(count) || states > 64) {
        return false;
    }
    r.text_tokens = static_cast<size_t>(text_tokens);
    r.states.resize(static_cast<size_t>(states));
    for (KvStore::RecordState& s : r.states) {
        int32_t pinned = 0;
        if (!take(&s.pos, 4) || !take(&pinned, 4) || !take(&s.key.hi, 8) || !take(&s.key.lo, 8)) return false;
        s.pinned = pinned != 0;
    }
    if (text_bytes != meta.size() - at) return false;
    r.text.assign(reinterpret_cast<const char*>(meta.data() + at), static_cast<size_t>(text_bytes));
    return true;
}

template <typename V>
void erase_value(std::vector<V>& v, const V& x) {
    v.erase(std::remove(v.begin(), v.end(), x), v.end());
}

}  // namespace

KvKey kv_seed_key(std::span<const uint64_t> parts) noexcept { return hash_after(KvKey{}, parts.data(), parts.size_bytes()); }

KvKey kv_chain_key(const KvKey& parent, std::span<const int32_t> tokens) noexcept {
    return hash_after(parent, tokens.data(), tokens.size_bytes());
}

KvKey kv_content_key(const KvKey& seed, std::span<const int32_t> tokens) noexcept {
    return hash_after(KvKey{seed.hi ^ kContentDomain, seed.lo ^ kContentDomain}, tokens.data(), tokens.size_bytes());
}

std::vector<int32_t> kv_segment_tokens(std::span<const int32_t> tokens, int32_t base) {
    std::vector<int32_t> out;
    out.reserve(tokens.size() + 2);
    out.push_back(kv_segment_base(base));
    out.insert(out.end(), tokens.begin(), tokens.end());
    out.push_back(kSegmentEnd);
    return out;
}

bool kv_is_segment(std::span<const int32_t> tokens) noexcept {
    return tokens.size() >= 2 && tokens.front() <= kv_segment_base(0) && tokens.back() == kSegmentEnd;
}

int32_t remove_kv_dedup_files(const std::string& dir) {
    int32_t removed = 0;
    std::vector<std::filesystem::path> found;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(os::utf8_path(dir), ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = os::path_utf8(it->path().filename());
        const size_t dot = name.find('.');
        const std::string_view rest = dot == std::string::npos ? std::string_view() : std::string_view(name).substr(dot);
        KvKey ignored;
        const bool hex16 = dot == 16 && parse_hex(std::string(32 - 16, '0') + name.substr(0, 16), ignored);
        if (hex16 && (rest == ".lykv" || rest.starts_with(".lykv.tmp"))) found.push_back(it->path());
    }
    for (const auto& path : found) {
        std::error_code rm;
        if (std::filesystem::remove(path, rm)) ++removed;
    }
    if (removed > 0) LIYAB_LOG_INFO("KV dedup is retired: %d snapshot file(s) removed from %s", removed, dir.c_str());
    return removed;
}

KvStore::KvStore(const Config& config) : config_(config) {}

KvStore::~KvStore() = default;  // the RAM tier is freed; files stay for later engines; the lock is released

std::unique_ptr<KvStore> KvStore::open(const Config& config) {
    if (config.chunk_tokens <= 0 || config.chunk_bytes == 0 ||
        config.chunk_bytes % static_cast<size_t>(config.chunk_tokens) != 0) {
        return nullptr;
    }
    std::unique_ptr<KvStore> store(new KvStore(config));
    if (!config.dir.empty() && config.disk_bytes > 0) {
        std::error_code ec;
        std::filesystem::create_directories(os::utf8_path(config.dir), ec);
        if (ec) {
            LIYAB_LOG_WARN("prefix cache: cannot use %s: %s", config.dir.c_str(), ec.message().c_str());
        } else if (store->lock_ = os::FileLock::acquire(config.dir + "/" + kLockName); !store->lock_.held()) {
            LIYAB_LOG_WARN("prefix cache: %s is in use by another engine or process; no storage tier",
                           config.dir.c_str());
        } else {
            store->disk_on_ = true;
            store->scan();
        }
    }
    if (config.ram_bytes == 0 && !store->disk_on_) return nullptr;
    return store;
}

size_t KvStore::rows_bytes(int32_t n) const noexcept {
    return config_.chunk_bytes / static_cast<size_t>(config_.chunk_tokens) * static_cast<size_t>(std::max(n, 0));
}

std::string KvStore::path_of(const KvKey& key, const char* ext) const { return config_.dir + "/kv-" + hex(key) + ext; }

KvKey KvStore::prefix_key(std::span<const int32_t> tokens, std::span<const KvKey> chunk_keys, int32_t pos) const noexcept {
    const int32_t c = config_.chunk_tokens;
    const int32_t full = pos / c;
    if (pos % c == 0) return full > 0 ? chunk_keys[static_cast<size_t>(full - 1)] : config_.seed;
    const KvKey& parent = full > 0 ? chunk_keys[static_cast<size_t>(full - 1)] : config_.seed;
    return kv_chain_key(parent, tokens.subspan(static_cast<size_t>(full) * static_cast<size_t>(c),
                                               static_cast<size_t>(pos - full * c)));
}

KvStore::Item* KvStore::item(Kind kind, const KvKey& key) {
    if (kind == Kind::Chunk) {
        auto it = chunks_.find(key);
        return it == chunks_.end() ? nullptr : &it->second;
    }
    auto it = states_.find(key);
    return it == states_.end() ? nullptr : &it->second;
}

void KvStore::set_clock(Kind kind, Item& it, uint64_t clock) {
    if (it.placed) lru_[it.on_disk].erase(it.clock);
    it.clock = clock;
    if (it.placed) lru_[it.on_disk][clock] = {kind, it.key};
}

void KvStore::touch_files(const std::vector<std::pair<Kind, KvKey>>& newest_first) {
    // Oldest first, so modification times follow the LRU order (ties are
    // broken by depth when the directory is opened again).
    for (auto it = newest_first.rbegin(); it != newest_first.rend(); ++it) {
        const Item* i = item(it->first, it->second);
        if (i != nullptr && i->placed && i->on_disk) touch_file(path_of(i->key, it->first == Kind::Chunk ? ".chunk" : ".state"));
    }
}

size_t KvStore::file_size(Kind kind, const Item& it) const noexcept {
    const size_t tokens = kind == Kind::Chunk ? static_cast<const Chunk&>(it).tokens.size()
                                              : static_cast<const State&>(it).tokens.size();
    return sizeof(FileHeader) + tokens * sizeof(int32_t) + it.payload_bytes;
}

bool KvStore::write_item_file(Kind kind, Item& it) {
    FileHeader h{};
    std::memcpy(h.magic, kMagic, sizeof kMagic);
    h.version = kFormat;
    h.encoding = static_cast<uint32_t>(KvEncoding::Raw);
    put_key(h.key, it.key);
    put_key(h.seed, it.seed);
    std::span<const uint8_t> meta;
    if (kind == Kind::Chunk) {
        const Chunk& c = static_cast<const Chunk&>(it);
        h.kind = kFileChunk;
        put_key(h.link, c.parent);
        put_key(h.content, c.content);
        h.start = c.start;
        h.count = static_cast<int32_t>(c.tokens.size());
        meta = {reinterpret_cast<const uint8_t*>(c.tokens.data()), c.tokens.size() * sizeof(int32_t)};
    } else {
        const State& s = static_cast<const State&>(it);
        h.kind = kFileState;
        put_key(h.link, s.parent);
        h.start = s.pos;
        h.count = static_cast<int32_t>(s.tokens.size());
        meta = {reinterpret_cast<const uint8_t*>(s.tokens.data()), s.tokens.size() * sizeof(int32_t)};
    }
    const std::span<const uint8_t> payload(it.data.get(), it.payload_bytes);
    h.meta_bytes = meta.size();
    h.payload_bytes = h.raw_bytes = payload.size();
    h.payload_hash = xxhash64(payload.data(), payload.size());
    h.header_hash = header_hash(h, meta.data(), meta.size());
    if (!write_file(path_of(it.key, kind == Kind::Chunk ? ".chunk" : ".state"), h, meta, payload)) {
        LIYAB_LOG_WARN("prefix cache: cannot write to %s", config_.dir.c_str());
        return false;
    }
    it.file_bytes = sizeof h + meta.size() + payload.size();
    it.payload_offset = sizeof h + meta.size();
    it.payload_hash = h.payload_hash;
    return true;
}

bool KvStore::write_record_file(RecordEntry& entry) {
    if (!disk_on_) return false;
    const Record& r = entry.record;
    const std::vector<uint8_t> meta = record_meta(r);
    const size_t bytes = sizeof(FileHeader) + meta.size();
    const auto leaf = chunks_.find(r.leaf);
    if (leaf == chunks_.end()) return false;
    // A rewrite replaces the old file: its bytes are free again.
    const size_t old = entry.on_disk ? entry.file_bytes : 0;
    disk_used_ -= old;
    entry.on_disk = false;
    entry.file_bytes = 0;
    // Room only from what is older than everything the record uses (its
    // deepest state images are older than its last chunk).
    uint64_t below = leaf->second.clock;
    for (const RecordState& rs : r.states) {
        if (auto st = states_.find(rs.key); st != states_.end()) below = std::min(below, st->second.clock);
    }
    if (!make_room(true, bytes, below) || records_.find(r.leaf) == records_.end()) {
        os::remove_file(path_of(r.leaf, ".ctx"));
        return false;
    }
    FileHeader h{};
    std::memcpy(h.magic, kMagic, sizeof kMagic);
    h.version = kFormat;
    h.kind = kFileRecord;
    put_key(h.key, r.leaf);
    put_key(h.link, r.leaf);
    put_key(h.seed, entry.seed);
    h.count = static_cast<int32_t>(r.tokens.size());
    h.meta_bytes = meta.size();
    h.payload_hash = xxhash64(nullptr, 0);
    h.header_hash = header_hash(h, meta.data(), meta.size());
    if (!write_file(path_of(r.leaf, ".ctx"), h, meta, {})) {
        os::remove_file(path_of(r.leaf, ".ctx"));
        return false;
    }
    entry.on_disk = true;
    entry.file_bytes = bytes;
    disk_used_ += bytes;
    return true;
}

// --- Caps ------------------------------------------------------------------------

bool KvStore::make_room(bool disk, size_t bytes, uint64_t below) {
    const size_t cap = disk ? config_.disk_bytes : config_.ram_bytes;
    if ((disk && !disk_on_) || bytes > cap) return false;
    size_t& used = disk ? disk_used_ : ram_used_;
    while (used + bytes > cap) {
        auto& lru = lru_[disk ? 1 : 0];
        if (lru.empty() || lru.begin()->first >= below) return false;
        const auto [kind, key] = lru.begin()->second;
        if (disk) {
            evict(kind, key);
        } else {
            demote(kind, key, /*or_evict=*/true);
        }
    }
    return true;
}

bool KvStore::demote(Kind kind, const KvKey& key, bool or_evict) {
    Item* it = item(kind, key);
    if (it == nullptr || !it->placed) return false;
    if (it->on_disk) return true;
    if (disk_on_ && make_room(true, file_size(kind, *it), it->clock)) {
        it = item(kind, key);  // making room never evicts newer items, but stay safe
        if (it != nullptr && write_item_file(kind, *it)) {
            lru_[0].erase(it->clock);
            ram_used_ -= it->payload_bytes;
            it->data.reset();
            it->on_disk = true;
            disk_used_ += it->file_bytes;
            lru_[1][it->clock] = {kind, key};
            if (kind == Kind::Chunk) {
                if (auto r = records_.find(key); r != records_.end()) write_record_file(r->second);
            }
            return true;
        }
    }
    if (or_evict) evict(kind, key);
    return false;
}

bool KvStore::place(Kind kind, Item& it, bool storage_only) {
    if (!storage_only && config_.ram_bytes > 0 && make_room(false, it.payload_bytes, it.clock)) {
        it.placed = true;
        it.on_disk = false;
        ram_used_ += it.payload_bytes;
        lru_[0][it.clock] = {kind, it.key};
        return true;
    }
    if (disk_on_ && make_room(true, file_size(kind, it), it.clock) && write_item_file(kind, it)) {
        it.data.reset();
        it.placed = true;
        it.on_disk = true;
        disk_used_ += it.file_bytes;
        lru_[1][it.clock] = {kind, it.key};
        return true;
    }
    return false;
}

void KvStore::release(Kind kind, Item& it) {
    if (!it.placed) return;
    if (it.on_disk) {
        os::remove_file(path_of(it.key, kind == Kind::Chunk ? ".chunk" : ".state"));
        disk_used_ -= it.file_bytes;
        lru_[1].erase(it.clock);
    } else {
        ram_used_ -= it.payload_bytes;
        lru_[0].erase(it.clock);
        it.data.reset();
    }
    it.placed = false;
}

void KvStore::evict(Kind kind, const KvKey& key) {
    if (kind == Kind::Chunk) {
        drop_chunk(key);
    } else {
        drop_state(key);  // the records using it then restore less on a hybrid model
    }
}

void KvStore::attach(State& state, const KvKey& chunk) {
    if (state.chunk == chunk) return;
    if (auto old = chunks_.find(state.chunk); old != chunks_.end()) erase_value(old->second.states, state.key);
    state.chunk = chunk;
    chunks_.at(chunk).states.push_back(state.key);
}

const KvStore::Chunk* KvStore::holder_of(const State& state, const KvKey& except) const {
    const std::vector<KvKey>* kids = nullptr;
    if (state.parent == config_.seed && state.seed == config_.seed) {
        kids = &roots_;
    } else if (auto parent = chunks_.find(state.parent); parent != chunks_.end()) {
        kids = &parent->second.kids;
    }
    const Chunk* best = nullptr;
    if (kids == nullptr) return best;
    for (const KvKey& key : *kids) {
        const Chunk& ch = chunks_.at(key);
        if (key == except || ch.seed != state.seed || ch.tokens.size() < state.tokens.size() ||
            !std::equal(state.tokens.begin(), state.tokens.end(), ch.tokens.begin())) {
            continue;
        }
        if (best == nullptr || ch.clock > best->clock) best = &ch;
    }
    return best;
}

void KvStore::drop_chunk(const KvKey& key) {
    // The chunk and every chunk after it, deepest first (no recursion: a
    // chain of a 32k-token context is 512 chunks deep).
    std::vector<KvKey> subtree{key};
    for (size_t i = 0; i < subtree.size(); ++i) {
        if (auto c = chunks_.find(subtree[i]); c != chunks_.end()) {
            subtree.insert(subtree.end(), c->second.kids.begin(), c->second.kids.end());
        }
    }
    for (auto it = subtree.rbegin(); it != subtree.rend(); ++it) drop_one_chunk(*it);
}

void KvStore::drop_one_chunk(const KvKey& key) {
    auto it = chunks_.find(key);
    if (it == chunks_.end()) return;
    drop_record(key, /*collect=*/false);
    // Its states move to another chunk holding their prefix, if one is left.
    const std::vector<KvKey> states = it->second.states;
    for (const KvKey& sk : states) {
        State& st = states_.at(sk);
        if (const Chunk* other = holder_of(st, key); other != nullptr) {
            attach(st, other->key);
        } else {
            drop_state(sk);
        }
    }
    Chunk& c = it->second;
    if (auto parent = chunks_.find(c.parent); parent != chunks_.end()) {
        erase_value(parent->second.kids, key);
    } else if (c.parent == config_.seed) {
        erase_value(roots_, key);
    }
    for (auto [a, b] = by_content_.equal_range(c.content); a != b; ++a) {
        if (a->second == key) {
            by_content_.erase(a);
            break;
        }
    }
    release(Kind::Chunk, c);
    chunks_.erase(it);
}

void KvStore::drop_state(const KvKey& key) {
    auto it = states_.find(key);
    if (it == states_.end()) return;
    State& s = it->second;
    if (auto c = chunks_.find(s.chunk); c != chunks_.end()) erase_value(c->second.states, key);
    for (auto& [leaf, entry] : records_) {
        std::erase_if(entry.record.states, [&](const RecordState& r) { return r.key == key; });
    }
    release(Kind::State, s);
    states_.erase(it);
}

void KvStore::drop_record(const KvKey& leaf, bool collect) {
    auto it = records_.find(leaf);
    if (it == records_.end()) return;
    RecordEntry entry = std::move(it->second);
    records_.erase(it);
    if (entry.on_disk) {
        os::remove_file(path_of(leaf, ".ctx"));
        disk_used_ -= entry.file_bytes;
    }
    for (const RecordState& rs : entry.record.states) {
        if (auto s = states_.find(rs.key); s != states_.end() && --s->second.refs <= 0) drop_state(rs.key);
    }
    // The partial chunk a replaced record ended at: what continues it holds
    // the same rows, so nothing needs it once nothing refers to it.
    if (collect) {
        if (auto c = chunks_.find(leaf); c != chunks_.end() && ours(c->second) &&
                                          c->second.tokens.size() < static_cast<size_t>(config_.chunk_tokens) &&
                                          c->second.kids.empty() && c->second.states.empty()) {
            drop_chunk(leaf);
        }
    }
}

// --- Setting aside ------------------------------------------------------------------

KvStore::PutResult KvStore::put(const ContextSource& source, bool storage_only) {
    PutResult result;
    const auto n = static_cast<int32_t>(source.tokens.size());
    if (n <= 0 || (storage_only && !disk_on_) || !source.rows) return result;
    const int32_t c = config_.chunk_tokens;
    const int32_t k = (n + c - 1) / c;
    result.chunks = k;
    auto chunk_tokens = [&](int32_t j) {
        return source.tokens.subspan(static_cast<size_t>(j) * static_cast<size_t>(c),
                                     static_cast<size_t>(std::min(c, n - j * c)));
    };
    std::vector<KvKey> keys(static_cast<size_t>(k));
    for (int32_t j = 0; j < k; ++j) keys[static_cast<size_t>(j)] = kv_chain_key(j > 0 ? keys[static_cast<size_t>(j - 1)] : config_.seed, chunk_tokens(j));
    struct Mark {
        StateMark mark;
        KvKey key;
        int32_t chunk;
    };
    std::vector<Mark> marks;
    if (config_.state_floats > 0 && source.state) {
        for (const StateMark& m : source.states) {
            if (m.pos > 0 && m.pos <= n) marks.push_back({m, prefix_key(source.tokens, keys, m.pos), (m.pos - 1) / c});
        }
    }
    // Newest first: each chunk, then its states (the deepest first), so a
    // chunk is always newer than what continues it.
    std::vector<std::pair<Kind, size_t>> order;
    for (int32_t j = 0; j < k; ++j) {
        order.emplace_back(Kind::Chunk, static_cast<size_t>(j));
        std::vector<size_t> own;
        for (size_t m = 0; m < marks.size(); ++m) {
            if (marks[m].chunk == j) own.push_back(m);
        }
        std::sort(own.begin(), own.end(), [&](size_t a, size_t b) { return marks[a].mark.pos > marks[b].mark.pos; });
        for (const size_t m : own) order.emplace_back(Kind::State, m);
    }
    const uint64_t base = clock_;
    clock_ += order.size();
    auto clock_of = [&](size_t i) { return base + order.size() - i; };
    auto key_of = [&](size_t i) { return order[i].first == Kind::Chunk ? keys[order[i].second] : marks[order[i].second].key; };

    // What is stored already moves to the front, before anything is added.
    std::vector<bool> present(order.size(), false);
    for (size_t i = 0; i < order.size(); ++i) {
        const auto [kind, index] = order[i];
        if (kind == Kind::Chunk) {
            const auto j = static_cast<int32_t>(index);
            auto it = chunks_.find(keys[index]);
            const auto tokens = chunk_tokens(j);
            if (it == chunks_.end() || !ours(it->second) || it->second.start != j * c ||
                !std::equal(tokens.begin(), tokens.end(), it->second.tokens.begin(), it->second.tokens.end())) {
                continue;
            }
            set_clock(kind, it->second, clock_of(i));
            ++result.chunks_reused;
        } else {
            auto it = states_.find(marks[index].key);
            if (it == states_.end() || !ours(it->second) || it->second.pos != marks[index].mark.pos) continue;
            set_clock(kind, it->second, clock_of(i));
        }
        present[i] = true;
    }
    // Being destroyed: what is in RAM goes to storage, or a later engine
    // would find the chunks after it without them.
    if (storage_only) {
        for (size_t i = 0; i < order.size(); ++i) {
            if (!present[i]) continue;
            const Item* it = item(order[i].first, key_of(i));
            if (it != nullptr && !it->on_disk && demote(order[i].first, key_of(i), /*or_evict=*/false)) {
                result.bytes_disk += item(order[i].first, key_of(i))->file_bytes;
            }
        }
    }
    // The rest, newest first: a chunk whose parent did not fit is not kept.
    bool chain = true;
    for (size_t i = 0; i < order.size(); ++i) {
        if (present[i]) continue;
        const auto [kind, index] = order[i];
        if (kind == Kind::Chunk) {
            const auto j = static_cast<int32_t>(index);
            if (!chain || (j > 0 && chunks_.find(keys[index - 1]) == chunks_.end()) || chunks_.count(keys[index]) != 0) {
                chain = false;
                continue;
            }
            Chunk chunk;
            chunk.key = keys[index];
            chunk.seed = config_.seed;
            chunk.clock = clock_of(i);
            chunk.parent = j > 0 ? keys[index - 1] : config_.seed;
            chunk.start = j * c;
            const auto tokens = chunk_tokens(j);
            chunk.tokens.assign(tokens.begin(), tokens.end());
            chunk.content = kv_content_key(config_.seed, tokens);
            chunk.payload_bytes = rows_bytes(static_cast<int32_t>(tokens.size()));
            chunk.data = std::make_unique<uint8_t[]>(chunk.payload_bytes);
            if (Status s = source.rows(j, static_cast<int32_t>(tokens.size()), chunk.data.get()); !s.is_ok()) {
                LIYAB_LOG_WARN("prefix cache: %s", s.to_string().c_str());
                chain = false;
                continue;
            }
            const KvKey key = chunk.key;
            auto [it, inserted] = chunks_.emplace(key, std::move(chunk));
            if (!place(Kind::Chunk, it->second, storage_only)) {
                chunks_.erase(it);
                chain = false;
                continue;
            }
            (it->second.on_disk ? result.bytes_disk : result.bytes_ram) +=
                it->second.on_disk ? it->second.file_bytes : it->second.payload_bytes;
            if (j > 0) {
                chunks_.at(keys[index - 1]).kids.push_back(it->first);
            } else {
                roots_.push_back(it->first);
            }
            by_content_.emplace(it->second.content, it->first);
            ++result.chunks_written;
        } else {
            const Mark& m = marks[index];
            auto owner = chunks_.find(keys[static_cast<size_t>(m.chunk)]);
            if (owner == chunks_.end() || states_.count(m.key) != 0) continue;
            State state;
            state.key = m.key;
            state.seed = config_.seed;
            state.clock = clock_of(i);
            state.chunk = owner->first;
            state.pos = m.mark.pos;
            state.parent = m.chunk > 0 ? keys[static_cast<size_t>(m.chunk - 1)] : config_.seed;
            const auto prefix = source.tokens.subspan(static_cast<size_t>(m.chunk) * static_cast<size_t>(c),
                                                      static_cast<size_t>(m.mark.pos - m.chunk * c));
            state.tokens.assign(prefix.begin(), prefix.end());
            state.payload_bytes = config_.state_floats * sizeof(float);
            state.data = std::make_unique<uint8_t[]>(state.payload_bytes);
            if (Status s = source.state(m.mark.pos, reinterpret_cast<float*>(state.data.get())); !s.is_ok()) {
                LIYAB_LOG_WARN("prefix cache: %s", s.to_string().c_str());
                continue;
            }
            const KvKey key = state.key;
            auto [it, inserted] = states_.emplace(key, std::move(state));
            if (!place(Kind::State, it->second, storage_only)) {
                states_.erase(it);
                continue;
            }
            (it->second.on_disk ? result.bytes_disk : result.bytes_ram) +=
                it->second.on_disk ? it->second.file_bytes : it->second.payload_bytes;
            if (auto o = chunks_.find(it->second.chunk); o != chunks_.end()) o->second.states.push_back(it->first);
            ++result.states_written;
        }
    }
    // States stay with the newest chunk holding their prefix: this context's.
    for (const Mark& m : marks) {
        const KvKey& own = keys[static_cast<size_t>(m.chunk)];
        if (auto st = states_.find(m.key); st != states_.end() && ours(st->second) && chunks_.count(own) != 0) {
            attach(st->second, own);
        }
    }
    // Every file of the context, in LRU order (new files were created newest first).
    touch_files([&] {
        std::vector<std::pair<Kind, KvKey>> touched;
        for (size_t i = 0; i < order.size(); ++i) touched.emplace_back(order[i].first, key_of(i));
        return touched;
    }());

    // The record, once every chunk is stored: the states that are, too.
    const KvKey leaf = keys.back();
    bool all_chunks = true;
    for (const KvKey& key : keys) all_chunks = all_chunks && chunks_.count(key) != 0;
    if (!all_chunks) return result;
    RecordEntry entry;
    entry.seed = config_.seed;
    entry.record.leaf = leaf;
    entry.record.tokens.assign(source.tokens.begin(), source.tokens.end());
    entry.record.text.assign(source.text);
    entry.record.text_tokens = std::min(source.text_tokens, source.tokens.size());
    bool all_states = true;
    for (const Mark& m : marks) {
        if (states_.count(m.key) != 0) {
            entry.record.states.push_back({m.mark.pos, m.mark.pinned, m.key});
        } else {
            all_states = false;
        }
    }
    for (const RecordState& rs : entry.record.states) ++states_.at(rs.key).refs;
    if (auto old = records_.find(leaf); old != records_.end()) {
        // The same context again: the new record replaces it (its states are
        // counted first, so the shared ones stay).
        const Record& was = old->second.record;
        const bool same = was.text == entry.record.text && was.text_tokens == entry.record.text_tokens &&
                          was.states.size() == entry.record.states.size() &&
                          std::equal(was.states.begin(), was.states.end(), entry.record.states.begin(),
                                     [](const RecordState& a, const RecordState& b) {
                                         return a.key == b.key && a.pos == b.pos && a.pinned == b.pinned;
                                     });
        if (same && (old->second.on_disk || !chunks_.at(leaf).on_disk)) {
            for (const RecordState& rs : entry.record.states) --states_.at(rs.key).refs;
            result.whole = all_states;
            return result;
        }
        entry.on_disk = old->second.on_disk;
        entry.file_bytes = old->second.file_bytes;
        // Released once the record is replaced: dropping a state edits every
        // record's list, this one's included.
        const std::vector<RecordState> released = std::move(old->second.record.states);
        old->second = std::move(entry);
        for (const RecordState& rs : released) {
            if (auto s = states_.find(rs.key); s != states_.end() && --s->second.refs <= 0) drop_state(rs.key);
        }
    } else {
        records_.emplace(leaf, std::move(entry));
    }
    RecordEntry& stored = records_.at(leaf);
    if (chunks_.at(leaf).on_disk) {
        write_record_file(stored);
    } else if (stored.on_disk) {  // a file of the replaced record: the leaf is in RAM now
        os::remove_file(path_of(leaf, ".ctx"));
        disk_used_ -= stored.file_bytes;
        stored.on_disk = false;
        stored.file_bytes = 0;
    }
    // Older records of the same conversation (any whose tokens it starts with).
    std::vector<KvKey> older;
    for (const auto& [key, e] : records_) {
        const std::vector<int32_t>& t = e.record.tokens;
        if (key != leaf && e.seed == config_.seed && t.size() < source.tokens.size() &&
            std::equal(t.begin(), t.end(), source.tokens.begin())) {
            older.push_back(key);
        }
    }
    for (const KvKey& key : older) drop_record(key, /*collect=*/true);
    result.whole = all_states && records_.count(leaf) != 0;
    return result;
}

// --- Restoring ------------------------------------------------------------------------

KvStore::Plan KvStore::plan(std::span<const int32_t> tokens, size_t limit) const {
    Plan plan;
    limit = std::min(limit, tokens.size());
    const auto c = static_cast<size_t>(config_.chunk_tokens);
    // Whole chunks of the prompt, while they are stored.
    std::vector<KvKey> path;
    KvKey parent = config_.seed;
    size_t pos = 0;
    while (pos + c <= limit) {
        const auto t = tokens.subspan(pos, c);
        const KvKey key = kv_chain_key(parent, t);
        const auto it = chunks_.find(key);
        if (it == chunks_.end() || !ours(it->second) || it->second.start != static_cast<int32_t>(pos) ||
            !std::equal(t.begin(), t.end(), it->second.tokens.begin(), it->second.tokens.end())) {
            break;
        }
        path.push_back(key);
        parent = key;
        pos += c;
    }
    // The stored chunks continuing that prefix: how far each follows the prompt.
    const std::vector<KvKey>* kids = nullptr;
    if (path.empty()) {
        kids = &roots_;
    } else {
        kids = &chunks_.at(path.back()).kids;
    }
    std::vector<std::pair<const Chunk*, size_t>> next;  // (chunk, tokens in common)
    for (const KvKey& key : config_.state_floats == 0 ? *kids : std::vector<KvKey>{}) {
        const Chunk& ch = chunks_.at(key);
        const size_t most = std::min(ch.tokens.size(), limit - pos);
        size_t common = 0;
        while (common < most && ch.tokens[common] == tokens[pos + common]) ++common;
        if (common > 0) next.emplace_back(&ch, common);
    }
    auto chunks_up_to = [&](const Chunk* last) {
        plan.chunks = path;
        for (const KvKey& key : path) plan.chunk_sizes.push_back(static_cast<int32_t>(chunks_.at(key).tokens.size()));
        if (last != nullptr) {
            plan.chunks.push_back(last->key);
            plan.chunk_sizes.push_back(static_cast<int32_t>(last->tokens.size()));
        }
    };
    if (config_.state_floats == 0) {
        // Attention only: any position restores.
        const std::pair<const Chunk*, size_t>* best = nullptr;
        for (const auto& candidate : next) {
            if (best == nullptr || candidate.second > best->second) best = &candidate;
        }
        plan.positions = static_cast<int32_t>(pos + (best != nullptr ? best->second : 0));
        if (plan.positions == 0) return plan;
        if (best != nullptr && best->second > 0) {
            chunks_up_to(best->first);
        } else {
            chunks_up_to(nullptr);
        }
    } else {
        // Hybrid: the deepest stored state the prompt's own prefix leads to,
        // whichever chunk holding that prefix it is attached to. The keys of
        // the prompt's whole chunks, as far as a state can lie.
        std::vector<KvKey> prompt_keys = path;
        for (size_t at = prompt_keys.size() * c; at + c <= limit; at += c) {
            prompt_keys.push_back(kv_chain_key(prompt_keys.empty() ? config_.seed : prompt_keys.back(), tokens.subspan(at, c)));
        }
        const State* best = nullptr;
        for (const auto& [key, st] : states_) {
            if (!ours(st) || st.pos <= 0 || static_cast<size_t>(st.pos) > limit ||
                (best != nullptr && st.pos <= best->pos)) {
                continue;
            }
            const auto j = static_cast<size_t>(st.pos - 1) / c;  // the chunk its position ends in
            const auto t = tokens.subspan(j * c, static_cast<size_t>(st.pos) - j * c);
            if (st.parent != (j > 0 ? prompt_keys[j - 1] : config_.seed) || kv_chain_key(st.parent, t) != st.key ||
                !std::equal(t.begin(), t.end(), st.tokens.begin(), st.tokens.end())) {
                continue;
            }
            best = &st;
        }
        if (best == nullptr) return plan;
        plan.positions = best->pos;
        // The prompt's chunks before it (stored: they lead to its chunk), then its chunk.
        const auto j = static_cast<size_t>(best->pos - 1) / c;
        path.assign(prompt_keys.begin(), prompt_keys.begin() + static_cast<std::ptrdiff_t>(j));
        chunks_up_to(&chunks_.at(best->chunk));
        plan.state = best->key;
        // Its snapshots: the images below it of the most recently used record
        // that holds it, with that record's pinned flags.
        const RecordEntry* holder = nullptr;
        uint64_t holder_clock = 0;
        for (const auto& [leaf, e] : records_) {
            for (const RecordState& rs : e.record.states) {
                if (rs.key != best->key || e.seed != config_.seed) continue;
                const uint64_t clock = chunks_.at(leaf).clock;
                if (holder == nullptr || clock > holder_clock) {
                    holder = &e;
                    holder_clock = clock;
                }
            }
        }
        plan.state_pinned = true;
        if (holder != nullptr) {
            for (const RecordState& rs : holder->record.states) {
                if (rs.key == best->key) {
                    plan.state_pinned = rs.pinned;
                } else if (rs.pos < best->pos && states_.count(rs.key) != 0) {
                    plan.snapshots.push_back(rs);
                }
            }
        }
    }
    // The record whose text the restored context then has: the one sharing
    // the most tokens with the prompt, at least the positions restored.
    size_t most = 0;
    for (const auto& [leaf, e] : records_) {
        const std::vector<int32_t>& t = e.record.tokens;
        if (e.seed != config_.seed || e.record.text_tokens == 0 || t.size() < static_cast<size_t>(plan.positions)) continue;
        const size_t n = std::min(t.size(), tokens.size());
        size_t common = 0;
        while (common < n && t[common] == tokens[common]) ++common;
        if (common >= static_cast<size_t>(plan.positions) && common > most) {
            most = common;
            plan.record = &e.record;
        }
    }
    return plan;
}

void KvStore::used(const Plan& plan) {
    // The images restored stay with the chunks they were restored with.
    const auto c = static_cast<size_t>(config_.chunk_tokens);
    auto follow = [&](const KvKey& key) {
        auto st = states_.find(key);
        if (st == states_.end()) return;
        const size_t j = static_cast<size_t>(st->second.pos - 1) / c;
        if (j < plan.chunks.size() && chunks_.count(plan.chunks[j]) != 0) attach(st->second, plan.chunks[j]);
    };
    if (plan.state != KvKey{}) follow(plan.state);
    for (const RecordState& rs : plan.snapshots) follow(rs.key);
    std::vector<std::pair<Kind, KvKey>> order;  // newest first
    for (const KvKey& key : plan.chunks) {
        order.emplace_back(Kind::Chunk, key);
        std::vector<const State*> own;
        auto add = [&](const KvKey& state) {
            if (auto s = states_.find(state); s != states_.end() && s->second.chunk == key) own.push_back(&s->second);
        };
        if (plan.state != KvKey{}) add(plan.state);
        for (const RecordState& rs : plan.snapshots) add(rs.key);
        std::sort(own.begin(), own.end(), [](const State* a, const State* b) { return a->pos > b->pos; });
        for (const State* s : own) order.emplace_back(Kind::State, s->key);
    }
    const uint64_t base = clock_;
    clock_ += order.size();
    for (size_t i = 0; i < order.size(); ++i) {
        if (Item* it = item(order[i].first, order[i].second)) set_clock(order[i].first, *it, base + order.size() - i);
    }
    touch_files(order);
}

Status KvStore::read_chunk(const KvKey& key, std::span<uint8_t> out) {
    auto it = chunks_.find(key);
    if (it == chunks_.end() || !ours(it->second) || !it->second.placed) {
        return Status(ErrorCode::InvalidArgument, "prefix cache: no such chunk");
    }
    Chunk& ch = it->second;
    if (out.size() != ch.payload_bytes) return Status(ErrorCode::InvalidArgument, "prefix cache: wrong chunk size");
    if (!ch.on_disk) {
        std::memcpy(out.data(), ch.data.get(), out.size());
        return Status::ok();
    }
    if (read_payload(path_of(key, ".chunk"), ch.payload_offset, ch.payload_hash, out)) return Status::ok();
    LIYAB_LOG_WARN("prefix cache: a chunk file is damaged or unreadable; removed");
    evict(Kind::Chunk, key);
    return Status(ErrorCode::IoError, "prefix cache: a chunk file is damaged or unreadable");
}

Status KvStore::read_state(const KvKey& key, std::span<float> out) {
    auto it = states_.find(key);
    if (it == states_.end() || !ours(it->second) || !it->second.placed) {
        return Status(ErrorCode::InvalidArgument, "prefix cache: no such state");
    }
    State& s = it->second;
    const std::span<uint8_t> bytes(reinterpret_cast<uint8_t*>(out.data()), out.size_bytes());
    if (bytes.size() != s.payload_bytes) return Status(ErrorCode::InvalidArgument, "prefix cache: wrong state size");
    if (!s.on_disk) {
        std::memcpy(bytes.data(), s.data.get(), bytes.size());
        return Status::ok();
    }
    if (read_payload(path_of(key, ".state"), s.payload_offset, s.payload_hash, bytes)) return Status::ok();
    LIYAB_LOG_WARN("prefix cache: a state file is damaged or unreadable; removed");
    evict(Kind::State, key);
    return Status(ErrorCode::IoError, "prefix cache: a state file is damaged or unreadable");
}

std::vector<KvKey> KvStore::chunk_path(std::span<const int32_t> tokens) const {
    std::vector<KvKey> keys;
    KvKey parent = config_.seed;
    const auto c = static_cast<size_t>(config_.chunk_tokens);
    for (size_t pos = 0; pos < tokens.size(); pos += c) {
        const auto t = tokens.subspan(pos, std::min(c, tokens.size() - pos));
        const KvKey key = kv_chain_key(parent, t);
        const auto it = chunks_.find(key);
        if (it == chunks_.end() || !ours(it->second) ||
            !std::equal(t.begin(), t.end(), it->second.tokens.begin(), it->second.tokens.end())) {
            return {};
        }
        keys.push_back(key);
        parent = key;
    }
    return keys;
}

std::vector<const KvStore::Record*> KvStore::records() const {
    std::vector<const Record*> out;
    for (const auto& [leaf, e] : records_) {
        if (e.seed == config_.seed && !kv_is_segment(e.record.tokens)) out.push_back(&e.record);
    }
    return out;
}

std::vector<KvStore::SegmentCopy> KvStore::segments(std::span<const int32_t> tokens) const {
    std::vector<std::pair<uint64_t, SegmentCopy>> found;  // (recency, copy)
    for (const auto& [leaf, e] : records_) {
        const std::vector<int32_t>& t = e.record.tokens;
        if (e.seed != config_.seed || t.size() != tokens.size() + 2 || !kv_is_segment(t) ||
            !std::equal(tokens.begin(), tokens.end(), t.begin() + 1)) {
            continue;
        }
        SegmentCopy copy;
        copy.base = kv_segment_base(0) - t.front();
        copy.chunks = chunk_path(t);
        if (copy.chunks.empty()) continue;  // cannot happen while the record lives; stay safe
        for (const KvKey& key : copy.chunks) copy.chunk_sizes.push_back(static_cast<int32_t>(chunks_.at(key).tokens.size()));
        found.emplace_back(chunks_.at(leaf).clock, std::move(copy));
    }
    std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<SegmentCopy> out;
    out.reserve(found.size());
    for (auto& f : found) out.push_back(std::move(f.second));
    return out;
}

const KvStore::Record* KvStore::record_for_text(std::string_view text) const {
    const Record* best = nullptr;
    for (const auto& [leaf, e] : records_) {
        const Record& r = e.record;
        if (e.seed == config_.seed && r.text_tokens > 0 && text.starts_with(r.text) &&
            (best == nullptr || r.text_tokens > best->text_tokens)) {
            best = &r;
        }
    }
    return best;
}

std::vector<KvStore::ChunkInfo> KvStore::chunks_with_content(const KvKey& content) const {
    std::vector<ChunkInfo> out;
    for (auto [a, b] = by_content_.equal_range(content); a != b; ++a) {
        const Chunk& ch = chunks_.at(a->second);
        out.push_back({ch.key, ch.parent, ch.start, static_cast<int32_t>(ch.tokens.size()), !ch.on_disk});
    }
    return out;
}

KvStore::Stats KvStore::stats() const {
    Stats s;
    s.ram_bytes = ram_used_;
    s.ram_cap = config_.ram_bytes;
    s.disk_bytes = disk_used_;
    s.disk_cap = disk_on_ ? config_.disk_bytes : 0;
    s.removed_files = removed_files_;
    for (const auto& [key, ch] : chunks_) {
        if (!ours(ch)) {
            ++s.foreign_files;
        } else if (ch.on_disk) {
            ++s.chunks_disk;
        } else {
            ++s.chunks_ram;
        }
    }
    for (const auto& [key, st] : states_) {
        if (ours(st)) {
            ++s.states;
        } else {
            ++s.foreign_files;
        }
    }
    for (const auto& [key, e] : records_) {
        if (e.seed != config_.seed) {
            if (e.on_disk) ++s.foreign_files;
        } else if (kv_is_segment(e.record.tokens)) {
            ++s.segments;
        } else {
            ++s.records;
        }
    }
    return s;
}

// --- Opening a directory ----------------------------------------------------------------

void KvStore::scan() {
    namespace fs = std::filesystem;
    struct Found {
        std::string path;
        FileHeader h{};
        std::vector<uint8_t> meta;
        size_t size = 0;
        int64_t mtime = 0;
    };
    std::vector<Found> found;
    std::vector<std::string> untrusted;  // removed once the directory has been read
    auto remove = [&](const std::string& path) { untrusted.push_back(path); };
    std::error_code ec;
    for (fs::directory_iterator it(os::utf8_path(config_.dir), ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = os::path_utf8(it->path().filename());
        const std::string path = os::path_utf8(it->path());
        if (name.starts_with("ctx-") && name.ends_with(".state")) {  // the prefix cache of older builds
            remove(path);
            continue;
        }
        if (!name.starts_with("kv-") || name == kLockName) continue;
        if (name.ends_with(".tmp")) {  // a write a crash cut short (the lock is ours: nobody is writing)
            remove(path);
            continue;
        }
        uint32_t kind = 0;
        if (name.ends_with(".chunk")) kind = kFileChunk;
        else if (name.ends_with(".state")) kind = kFileState;
        else if (name.ends_with(".ctx")) kind = kFileRecord;
        else continue;  // not a name the store writes: not ours to touch
        KvKey key;
        const size_t dot = name.find('.');
        std::error_code size_ec;
        Found f;
        f.path = path;
        f.size = static_cast<size_t>(it->file_size(size_ec));
        const auto time = it->last_write_time(size_ec);
        f.mtime = static_cast<int64_t>(time.time_since_epoch().count());
        bool ok = !size_ec && parse_hex(std::string_view(name).substr(3, dot - 3), key) && f.size >= sizeof(FileHeader);
        if (ok) {
            std::FILE* file = os::open_file(path, "rb");
            ok = file != nullptr && std::fread(&f.h, sizeof f.h, 1, file) == 1 &&
                 std::memcmp(f.h.magic, kMagic, sizeof kMagic) == 0 && f.h.version == kFormat && f.h.kind == kind &&
                 get_key(f.h.key) == key && f.h.meta_bytes <= kMaxMeta &&
                 f.h.meta_bytes + f.h.payload_bytes + sizeof(FileHeader) == f.size;
            if (ok) {
                f.meta.resize(static_cast<size_t>(f.h.meta_bytes));
                ok = f.meta.empty() || std::fread(f.meta.data(), 1, f.meta.size(), file) == f.meta.size();
            }
            if (file != nullptr) std::fclose(file);
            ok = ok && header_hash(f.h, f.meta.data(), f.meta.size()) == f.h.header_hash;
        }
        if (!ok) {  // damaged, or another format version: never read
            remove(path);
            continue;
        }
        found.push_back(std::move(f));
    }
    for (const std::string& path : untrusted) {
        if (os::remove_file(path)) ++removed_files_;
    }
    untrusted.clear();
    auto remove_now = [&](const std::string& path) {
        if (os::remove_file(path)) ++removed_files_;
    };

    // Chunks, parents first: a chunk is kept only when its chain is intact.
    std::vector<Found*> chunk_files, state_files, record_files;
    for (Found& f : found) {
        (f.h.kind == kFileChunk ? chunk_files : f.h.kind == kFileState ? state_files : record_files).push_back(&f);
    }
    std::sort(chunk_files.begin(), chunk_files.end(), [](const Found* a, const Found* b) { return a->h.start < b->h.start; });
    const auto c = static_cast<size_t>(config_.chunk_tokens);
    for (Found* f : chunk_files) {
        Chunk ch;
        ch.key = get_key(f->h.key);
        ch.seed = get_key(f->h.seed);
        ch.parent = get_key(f->h.link);
        ch.content = get_key(f->h.content);
        ch.start = f->h.start;
        const auto count = static_cast<size_t>(std::max(f->h.count, 0));
        bool ok = f->h.count > 0 && f->meta.size() == count * sizeof(int32_t) &&
                  f->h.encoding == static_cast<uint32_t>(KvEncoding::Raw) && f->h.raw_bytes == f->h.payload_bytes;
        if (ok) {
            ch.tokens.resize(count);
            std::memcpy(ch.tokens.data(), f->meta.data(), f->meta.size());
            if (ch.start == 0) {
                ok = ch.parent == ch.seed;
            } else {
                const auto p = chunks_.find(ch.parent);
                ok = p != chunks_.end() && p->second.seed == ch.seed &&
                     p->second.start + static_cast<int32_t>(p->second.tokens.size()) == ch.start;
            }
        }
        if (ok && ch.seed == config_.seed) {
            ok = count <= c && ch.start % config_.chunk_tokens == 0 && f->h.payload_bytes == rows_bytes(static_cast<int32_t>(count)) &&
                 kv_chain_key(ch.parent, ch.tokens) == ch.key && kv_content_key(ch.seed, ch.tokens) == ch.content &&
                 (ch.start == 0 || chunks_.at(ch.parent).tokens.size() == c);
        }
        if (!ok || chunks_.count(ch.key) != 0) {
            remove_now(f->path);
            continue;
        }
        ch.placed = ch.on_disk = true;
        ch.payload_bytes = static_cast<size_t>(f->h.payload_bytes);
        ch.file_bytes = f->size;
        ch.payload_offset = sizeof(FileHeader) + f->meta.size();
        ch.payload_hash = f->h.payload_hash;
        ch.mtime = f->mtime;
        const KvKey key = ch.key;
        auto [it, inserted] = chunks_.emplace(key, std::move(ch));
        if (it->second.start > 0) {
            chunks_.at(it->second.parent).kids.push_back(key);
        } else if (it->second.seed == config_.seed) {
            roots_.push_back(key);
        }
        if (it->second.seed == config_.seed) by_content_.emplace(it->second.content, key);
    }
    // States: their prefix must be stored (a chunk continuing their parent
    // that starts with their tokens); attached to the most recent such chunk.
    for (Found* f : state_files) {
        State st;
        st.key = get_key(f->h.key);
        st.seed = get_key(f->h.seed);
        st.parent = get_key(f->h.link);
        st.pos = f->h.start;
        const auto count = static_cast<size_t>(std::max(f->h.count, 0));
        bool ok = f->h.count > 0 && f->meta.size() == count * sizeof(int32_t) && f->h.raw_bytes == f->h.payload_bytes;
        if (ok) {
            st.tokens.resize(count);
            std::memcpy(st.tokens.data(), f->meta.data(), f->meta.size());
        }
        if (ok && st.seed == config_.seed) {
            const auto j = static_cast<size_t>(std::max(st.pos - 1, 0)) / c;
            ok = st.pos > 0 && count == static_cast<size_t>(st.pos) - j * c &&
                 f->h.payload_bytes == config_.state_floats * sizeof(float) && kv_chain_key(st.parent, st.tokens) == st.key;
        }
        const Chunk* holder = nullptr;
        if (ok) {
            const std::vector<KvKey>* kids = nullptr;
            if (st.seed == config_.seed && st.parent == config_.seed) {
                kids = &roots_;
            } else if (auto parent = chunks_.find(st.parent); parent != chunks_.end() && parent->second.seed == st.seed) {
                kids = &parent->second.kids;
            } else if (st.parent == st.seed) {  // another seed's first chunks: any of them may hold it
                kids = nullptr;
                for (const auto& [key, ch] : chunks_) {
                    if (ch.seed == st.seed && ch.start == 0 && ch.tokens.size() >= count &&
                        std::equal(st.tokens.begin(), st.tokens.end(), ch.tokens.begin()) &&
                        (holder == nullptr || ch.mtime > holder->mtime)) {
                        holder = &ch;
                    }
                }
            }
            for (const KvKey& key : kids != nullptr ? *kids : std::vector<KvKey>{}) {
                const Chunk& ch = chunks_.at(key);
                if (ch.tokens.size() >= count && std::equal(st.tokens.begin(), st.tokens.end(), ch.tokens.begin()) &&
                    (holder == nullptr || ch.mtime > holder->mtime)) {
                    holder = &ch;
                }
            }
            ok = holder != nullptr;
        }
        if (!ok || states_.count(st.key) != 0) {
            remove_now(f->path);
            continue;
        }
        st.chunk = holder->key;
        st.placed = st.on_disk = true;
        st.payload_bytes = static_cast<size_t>(f->h.payload_bytes);
        st.file_bytes = f->size;
        st.payload_offset = sizeof(FileHeader) + f->meta.size();
        st.payload_hash = f->h.payload_hash;
        st.mtime = f->mtime;
        const KvKey key = st.key;
        chunks_.at(holder->key).states.push_back(key);
        states_.emplace(key, std::move(st));
    }
    // Records: their last chunk and every state they use must be there.
    for (Found* f : record_files) {
        RecordEntry e;
        e.seed = get_key(f->h.seed);
        e.on_disk = true;
        e.file_bytes = f->size;
        e.record.leaf = get_key(f->h.link);
        const auto leaf = chunks_.find(e.record.leaf);
        bool ok = get_key(f->h.key) == e.record.leaf && leaf != chunks_.end() && leaf->second.seed == e.seed &&
                  f->h.payload_bytes == 0 && records_.count(e.record.leaf) == 0 &&
                  parse_record_meta(f->meta, f->h.count, e.record);
        if (ok) {
            const Chunk& ch = leaf->second;
            ok = e.record.tokens.size() == static_cast<size_t>(ch.start) + ch.tokens.size();
            // A state that went before the record (evicted on its own) is just not restored.
            std::erase_if(e.record.states, [&](const RecordState& rs) {
                const auto st = states_.find(rs.key);
                return st == states_.end() || st->second.pos != rs.pos || st->second.seed != e.seed;
            });
            if (ok && e.seed == config_.seed) {
                KvKey key = config_.seed;
                for (size_t pos = 0; pos < e.record.tokens.size(); pos += c) {
                    key = kv_chain_key(key, std::span<const int32_t>(e.record.tokens).subspan(pos, std::min(c, e.record.tokens.size() - pos)));
                }
                ok = key == e.record.leaf;
            }
        }
        if (!ok) {
            remove_now(f->path);
            continue;
        }
        if (e.seed != config_.seed) {  // kept for its own engine; its contents are not needed here
            e.record.tokens.clear();
            e.record.tokens.shrink_to_fit();
            e.record.text.clear();
        }
        for (const RecordState& rs : e.record.states) ++states_.at(rs.key).refs;
        records_.emplace(e.record.leaf, std::move(e));
    }
    // What a crash left half done: states no record uses, partial chunks
    // nothing holds.
    for (auto it = states_.begin(); it != states_.end();) {
        if (it->second.refs > 0) {
            ++it;
            continue;
        }
        if (auto owner = chunks_.find(it->second.chunk); owner != chunks_.end()) erase_value(owner->second.states, it->first);
        os::remove_file(path_of(it->first, ".state"));
        ++removed_files_;
        it = states_.erase(it);
    }
    std::vector<KvKey> garbage;
    for (const auto& [key, ch] : chunks_) {
        if (ours(ch) && ch.tokens.size() < c && ch.kids.empty() && ch.states.empty() && records_.count(key) == 0) {
            garbage.push_back(key);
        }
    }
    for (const KvKey& key : garbage) {
        const Chunk& ch = chunks_.at(key);
        if (auto p = chunks_.find(ch.parent); p != chunks_.end()) erase_value(p->second.kids, key);
        erase_value(roots_, key);
        for (auto [a, b] = by_content_.equal_range(ch.content); a != b; ++a) {
            if (a->second == key) {
                by_content_.erase(a);
                break;
            }
        }
        os::remove_file(path_of(key, ".chunk"));
        ++removed_files_;
        chunks_.erase(key);
    }

    // The LRU order from modification times. A chunk counts as recent as
    // anything after it (a crash while times were updated cannot put it
    // behind them); at equal times the deeper goes first, a state before its
    // chunk.
    std::vector<const Chunk*> deepest_first;
    for (const auto& [key, ch] : chunks_) deepest_first.push_back(&ch);
    std::sort(deepest_first.begin(), deepest_first.end(), [](const Chunk* a, const Chunk* b) { return a->start > b->start; });
    std::unordered_map<KvKey, int64_t, KvKeyHash> recency;
    for (const Chunk* ch : deepest_first) {
        int64_t t = ch->mtime;
        for (const KvKey& kid : ch->kids) t = std::max(t, recency.at(kid));
        recency[ch->key] = t;
    }
    struct Order {
        int64_t time;
        int32_t start;
        int kind;  // 0 state, 1 chunk
        int32_t pos;
        Kind which;
        KvKey key;
    };
    std::vector<Order> order;
    for (const auto& [key, ch] : chunks_) order.push_back({recency.at(key), ch.start, 1, ch.start, Kind::Chunk, key});
    for (const auto& [key, st] : states_) {
        const Chunk& ch = chunks_.at(st.chunk);
        order.push_back({std::min(st.mtime, recency.at(st.chunk)), ch.start, 0, st.pos, Kind::State, key});
    }
    std::sort(order.begin(), order.end(), [](const Order& a, const Order& b) {
        return std::tie(a.time, b.start, a.kind, a.pos) < std::tie(b.time, a.start, b.kind, b.pos);
    });
    for (const Order& o : order) {
        Item* it = item(o.which, o.key);
        it->clock = ++clock_;
        lru_[1][it->clock] = {o.which, o.key};
        disk_used_ += it->file_bytes;
    }
    for (const auto& [key, e] : records_) disk_used_ += e.file_bytes;
    // A cap lowered since: the least recently used go.
    while (disk_used_ > config_.disk_bytes && !lru_[1].empty()) {
        const auto [kind, key] = lru_[1].begin()->second;
        evict(kind, key);
    }
    if (removed_files_ > 0) {
        LIYAB_LOG_INFO("prefix cache: %d untrusted file(s) removed from %s", removed_files_, config_.dir.c_str());
    }
}

}  // namespace liyab
