// Liyab — operating-system services shared by the engine's modules (internal).
//
// The OS calls the expert store, the triple buffer, the engine's cache files,
// the power manager and the experimental modules need beyond standard C++,
// behind one interface. On macOS, Linux and Android each function makes the
// POSIX call those modules made before (sysconf, posix_memalign, mmap, mlock,
// getpid, rename, fopen, stat, access; flock for the KV store's lock), so
// their behaviour there is unchanged;
// on Windows each maps to its Win32 or C-runtime counterpart, and where the
// semantics differ the function says how. The loader (mmap_loader.cpp), the
// direct reader (direct_io.cpp) and the thread pool keep their own Windows
// code, next to the POSIX code it replaces.
//
// Paths are UTF-8 strings everywhere in Liyab (the C ABI's contract). Windows'
// narrow file APIs read a path in the process's ANSI code page instead (UTF-8
// only when the executable's manifest declares it), so every path the library
// opens on Windows goes through utf8_path(), open_file() or widen(), which use
// the UTF-16 APIs. Thread-safe: every function may be called from any thread.
#ifndef LIYAB_CORE_OS_H
#define LIYAB_CORE_OS_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>

namespace liyab::os {

// --- Memory -----------------------------------------------------------------------

// Bytes per virtual-memory page: 4 KiB on x86-64 and most phones, 16 KiB on
// Apple silicon and some Android devices.
size_t page_size() noexcept;

// `bytes` of heap memory aligned to `alignment` (a power of two, at least
// sizeof(void*)); nullptr when it cannot be allocated. Release it with
// aligned_free(), never std::free(): Windows' _aligned_malloc keeps a header
// in front of the block.
void* aligned_alloc(size_t alignment, size_t bytes) noexcept;
void aligned_free(void* p) noexcept;

// `bytes` of zero-filled, page-aligned read-write memory straight from the OS
// (mmap MAP_ANONYMOUS; VirtualAlloc, committed at once, so Windows counts it
// against the commit limit from the start), released by unmap_anonymous()
// with the same size; nullptr when the OS refuses.
void* map_anonymous(size_t bytes) noexcept;
void unmap_anonymous(void* p, size_t bytes) noexcept;

// Keeps [p, p + bytes) in RAM (mlock); false when the OS refuses, and the
// memory then stays pageable. Always false on Windows: VirtualLock is capped by
// the process's minimum working set (a few hundred KiB unless raised), and
// raising that by gigabytes of expert cache would hold the RAM away from every
// other process even while the engine idles.
bool lock_memory(void* p, size_t bytes) noexcept;
// Undoes lock_memory() (munlock); nothing to undo on Windows.
void unlock_memory(void* p, size_t bytes) noexcept;

// --- Processes ----------------------------------------------------------------------

// This process's id (unique names for temporary files).
int process_id() noexcept;

// --- Files -------------------------------------------------------------------------

// `utf8` as a filesystem path. On Windows the bytes are decoded as UTF-8 into
// the path's UTF-16 form, where std::filesystem::path(std::string) would
// decode them in the ANSI code page.
std::filesystem::path utf8_path(std::string_view utf8);
// `path` as UTF-8. On Windows path::string() converts to the ANSI code page
// instead, and throws for a character that page cannot represent.
std::string path_utf8(const std::filesystem::path& path);
// std::fopen() of a UTF-8 path (_wfopen on Windows); nullptr on failure.
std::FILE* open_file(const std::string& path, const char* mode);
// Renames `from` to `to`, replacing `to` when it exists, atomically on one
// volume, as POSIX rename() does. Windows' rename() refuses an existing
// target: MoveFileExW with MOVEFILE_REPLACE_EXISTING there.
bool replace_file(const std::string& from, const std::string& to);
// Removes the file at `path` (std::remove; DeleteFileW on Windows, which
// removes a file other processes still have open only if they allowed it, as
// Liyab's own mappings do); false when it could not.
bool remove_file(const std::string& path);
// Whether anything exists at `path` (access F_OK).
bool path_exists(const std::string& path);
// Whether `path` is a directory this process may create files in (stat +
// access W_OK). Windows keeps no write bit on directories (its access()
// reports every directory as writable): there, an existing directory, and
// creating a file in it reports what its ACL allows.
bool writable_directory(const std::string& path);

// An exclusive advisory lock on the file at `path` (created when missing),
// held until the object is destroyed or the process ends, a crash included
// (flock; LockFileEx on Windows). Any other holder fails to take it
// meanwhile: another process, and another FileLock of this one (both locks
// belong to an open file, not to a process). Owners of a directory of shared
// files (the engine's KV store) take it so one of them alone changes it.
class FileLock {
public:
    FileLock() = default;
    ~FileLock();
    FileLock(FileLock&& other) noexcept;
    FileLock& operator=(FileLock&& other) noexcept;
    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;
    // Takes the lock without waiting; an empty FileLock when another holder
    // has it or the file cannot be opened.
    static FileLock acquire(const std::string& path);
    [[nodiscard]] bool held() const noexcept { return handle_ != kNone; }

private:
    static constexpr intptr_t kNone = -1;
    intptr_t handle_ = kNone;  // a POSIX descriptor or a Windows HANDLE
};

#if defined(_WIN32)
// UTF-8 <-> UTF-16 for Win32 calls. Invalid UTF-8 becomes U+FFFD.
std::wstring widen(std::string_view utf8);
std::string narrow(std::wstring_view utf16);
// The system's description of Win32 error `code` (GetLastError()), as UTF-8
// with the code appended.
std::string error_text(unsigned long code);
#endif

}  // namespace liyab::os

#endif  // LIYAB_CORE_OS_H
