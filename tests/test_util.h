// Liyab — minimal dependency-free test harness (no network fetch of a test
// framework is needed to build on CI or on a device).
#ifndef LIYAB_TESTS_TEST_UTIL_H
#define LIYAB_TESTS_TEST_UTIL_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <filesystem>
#include <system_error>
#endif

#include "core/os.h"

namespace liyab::test {

struct Case {
    const char* name;
    std::function<void()> fn;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

inline int& failures() {
    static int count = 0;
    return count;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

inline int run_all() {
    int failed_cases = 0;
    for (const Case& c : registry()) {
        const int before = failures();
        c.fn();
        const bool ok = failures() == before;
        failed_cases += ok ? 0 : 1;
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", c.name);
    }
    std::printf("%zu cases, %d failed\n", registry().size(), failed_cases);
    return failed_cases == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

// This process's id, which the tests put in their temporary file names.
inline int process_id() { return os::process_id(); }

// Directory for temporary files: $TMPDIR, else /tmp (/data/local/tmp on
// Android, the user's temporary directory on Windows: %TMP% / %TEMP%).
inline std::string temp_dir() {
    if (const char* dir = std::getenv("TMPDIR")) return dir;
#if defined(__ANDROID__)
    return "/data/local/tmp";
#elif defined(_WIN32)
    std::error_code ec;
    std::string dir = os::path_utf8(std::filesystem::temp_directory_path(ec));
    if (ec || dir.empty()) return ".";
    while (dir.size() > 3 && (dir.back() == '\\' || dir.back() == '/')) dir.pop_back();  // callers append "/name"
    return dir;
#else
    return "/tmp";
#endif
}

}  // namespace liyab::test

#define LIYAB_CONCAT_(a, b) a##b
#define LIYAB_CONCAT(a, b) LIYAB_CONCAT_(a, b)
#define TEST_CASE(name)                                                                                   \
    static void LIYAB_CONCAT(test_fn_, __LINE__)();                                                       \
    static ::liyab::test::Registrar LIYAB_CONCAT(test_reg_, __LINE__)(name, LIYAB_CONCAT(test_fn_, __LINE__)); \
    static void LIYAB_CONCAT(test_fn_, __LINE__)()

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::printf("  %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond);        \
            ++::liyab::test::failures();                                                  \
        }                                                                                 \
    } while (0)

// Like CHECK but aborts the current test case (for preconditions).
#define REQUIRE(cond)                                                                     \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::printf("  %s:%d: REQUIRE(%s) failed\n", __FILE__, __LINE__, #cond);      \
            ++::liyab::test::failures();                                                  \
            return;                                                                       \
        }                                                                                 \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                          \
    do {                                                                                               \
        const double va_ = (a), vb_ = (b);                                                             \
        if (!(std::fabs(va_ - vb_) <= (tol))) {                                                        \
            std::printf("  %s:%d: CHECK_NEAR(%s, %s) failed: %g vs %g (tol %g)\n", __FILE__, __LINE__, \
                        #a, #b, va_, vb_, static_cast<double>(tol));                                   \
            ++::liyab::test::failures();                                                               \
        }                                                                                              \
    } while (0)

#define TEST_MAIN() \
    int main() { return ::liyab::test::run_all(); }

#endif  // LIYAB_TESTS_TEST_UTIL_H
