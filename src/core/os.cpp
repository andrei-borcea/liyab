#include "core/os.h"

#include <algorithm>
#include <cstdlib>
#include <utility>

#if defined(_WIN32)
#include <malloc.h>   // _aligned_malloc, _aligned_free
#include <process.h>  // _getpid

#include "core/windows_api.h"
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace liyab::os {

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------
size_t page_size() noexcept {
#if defined(_WIN32)
    static const size_t size = [] {
        SYSTEM_INFO info{};
        GetSystemInfo(&info);
        return static_cast<size_t>(info.dwPageSize);
    }();
#else
    static const size_t size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
    return size;
}

void* aligned_alloc(size_t alignment, size_t bytes) noexcept {
#if defined(_WIN32)
    return _aligned_malloc(bytes, alignment);
#else
    void* p = nullptr;
    return posix_memalign(&p, alignment, bytes) == 0 ? p : nullptr;
#endif
}

void aligned_free(void* p) noexcept {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

void* map_anonymous(size_t bytes) noexcept {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}

void unmap_anonymous(void* p, size_t bytes) noexcept {
    if (p == nullptr) return;
#if defined(_WIN32)
    (void)bytes;
    VirtualFree(p, 0, MEM_RELEASE);  // MEM_RELEASE frees the whole allocation and takes size 0
#else
    munmap(p, bytes);
#endif
}

bool lock_memory(void* p, size_t bytes) noexcept {
#if defined(_WIN32)
    (void)p;
    (void)bytes;
    return false;
#else
    return mlock(p, bytes) == 0;
#endif
}

void unlock_memory(void* p, size_t bytes) noexcept {
#if defined(_WIN32)
    (void)p;
    (void)bytes;
#else
    munlock(p, bytes);
#endif
}

// ---------------------------------------------------------------------------
// Processes
// ---------------------------------------------------------------------------
int process_id() noexcept {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<int>(getpid());
#endif
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------
std::filesystem::path utf8_path(std::string_view utf8) {
#if defined(_WIN32)
    return std::filesystem::path(widen(utf8));
#else
    return std::filesystem::path(utf8);
#endif
}

std::string path_utf8(const std::filesystem::path& path) {
#if defined(_WIN32)
    return narrow(path.native());
#else
    return path.string();
#endif
}

std::FILE* open_file(const std::string& path, const char* mode) {
#if defined(_WIN32)
    return _wfopen(widen(path).c_str(), widen(mode).c_str());
#else
    return std::fopen(path.c_str(), mode);
#endif
}

bool replace_file(const std::string& from, const std::string& to) {
#if defined(_WIN32)
    return MoveFileExW(widen(from).c_str(), widen(to).c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    return std::rename(from.c_str(), to.c_str()) == 0;
#endif
}

bool remove_file(const std::string& path) {
#if defined(_WIN32)
    return DeleteFileW(widen(path).c_str()) != 0;
#else
    return std::remove(path.c_str()) == 0;
#endif
}

bool path_exists(const std::string& path) {
#if defined(_WIN32)
    return GetFileAttributesW(widen(path).c_str()) != INVALID_FILE_ATTRIBUTES;
#else
    return access(path.c_str(), F_OK) == 0;
#endif
}

bool writable_directory(const std::string& path) {
#if defined(_WIN32)
    const DWORD attributes = GetFileAttributesW(widen(path).c_str());
    return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode) && access(path.c_str(), W_OK) == 0;
#endif
}

FileLock::~FileLock() {
    if (handle_ == kNone) return;
#if defined(_WIN32)
    // Closing the handle releases its lock.
    CloseHandle(reinterpret_cast<HANDLE>(handle_));
#else
    close(static_cast<int>(handle_));  // releases the flock of this open file
#endif
}

FileLock::FileLock(FileLock&& other) noexcept : handle_(other.handle_) { other.handle_ = kNone; }

FileLock& FileLock::operator=(FileLock&& other) noexcept {
    if (this != &other) {
        FileLock released(std::move(*this));
        handle_ = other.handle_;
        other.handle_ = kNone;
    }
    return *this;
}

FileLock FileLock::acquire(const std::string& path) {
    FileLock lock;
#if defined(_WIN32)
    // Sharing every access lets other processes open the file; LockFileEx
    // then refuses them the byte range, as flock refuses a second lock.
    HANDLE h = CreateFileW(widen(path).c_str(), GENERIC_READ | GENERIC_WRITE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return lock;
    OVERLAPPED at{};
    if (LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &at) == 0) {
        CloseHandle(h);
        return lock;
    }
    lock.handle_ = reinterpret_cast<intptr_t>(h);
#else
    const int fd = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return lock;
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        close(fd);
        return lock;
    }
    lock.handle_ = fd;
#endif
    return lock;
}

#if defined(_WIN32)
std::wstring widen(std::string_view utf8) {
    if (utf8.empty()) return {};
    const int length = static_cast<int>(utf8.size());
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), length, nullptr, 0);
    std::wstring out(static_cast<size_t>(std::max(n, 0)), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, utf8.data(), length, out.data(), n);
    return out;
}

std::string narrow(std::wstring_view utf16) {
    if (utf16.empty()) return {};
    const int length = static_cast<int>(utf16.size());
    const int n = WideCharToMultiByte(CP_UTF8, 0, utf16.data(), length, nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(std::max(n, 0)), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, utf16.data(), length, out.data(), n, nullptr, nullptr);
    return out;
}

std::string error_text(unsigned long code) {
    wchar_t* message = nullptr;
    const DWORD n = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                       FORMAT_MESSAGE_IGNORE_INSERTS,
                                   nullptr, static_cast<DWORD>(code), 0, reinterpret_cast<LPWSTR>(&message), 0, nullptr);
    std::string text = n > 0 && message != nullptr ? narrow(std::wstring_view(message, n)) : std::string();
    if (message != nullptr) LocalFree(message);
    // System messages end with ".\r\n"; strerror()'s have neither.
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ' || text.back() == '.')) {
        text.pop_back();
    }
    return (text.empty() ? std::string("error") : text) + " (Windows error " + std::to_string(code) + ")";
}
#endif

}  // namespace liyab::os
