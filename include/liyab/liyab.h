// Liyab — umbrella header for the C++ API.
//
// Liyab (Tagalog for "flame") is an on-device LLM inference engine for
// Android and iOS: GGUF weights are memory-mapped and never copied, compute
// is routed per device (NPU → GPU → CPU), and a power manager paces token
// output and reacts to thermal pressure. See README.md for an overview.
#ifndef LIYAB_LIYAB_H
#define LIYAB_LIYAB_H

#define LIYAB_VERSION_MAJOR 0
#define LIYAB_VERSION_MINOR 1
#define LIYAB_VERSION_PATCH 0
#define LIYAB_VERSION_STRING "0.1.0"

#include "liyab/backend.h"
#include "liyab/device_detect.h"
#include "liyab/engine.h"
#include "liyab/mmap_loader.h"
#include "liyab/power_manager.h"
#include "liyab/types.h"

#endif  // LIYAB_LIYAB_H
