// Liyab — <windows.h> for the Windows code paths (internal).
//
// Every Liyab source that calls Win32 includes this header instead of
// <windows.h>, so the macros that collide with C++ code are off everywhere:
// min / max (std::min, std::numeric_limits<T>::max), and wingdi.h's ERROR,
// OPAQUE, TRANSPARENT... Include it only from .cpp files, after the C++
// headers, never from a header other files include.
#ifndef LIYAB_CORE_WINDOWS_API_H
#define LIYAB_CORE_WINDOWS_API_H

#if !defined(_WIN32)
#error "core/windows_api.h is for Windows builds only"
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOGDI
#define NOGDI
#endif

#include <windows.h>

#endif  // LIYAB_CORE_WINDOWS_API_H
