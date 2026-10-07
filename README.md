# ⚡ Liyab AI: Mobile Silicon Inference Engine

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Android_%7C_iOS-brightgreen.svg)]()
[![Backend](https://img.shields.io/badge/Backend-Qualcomm_QNN_%7C_CoreML_%7C_Vulkan-orange.svg)]()

> **Liyab** (*Tagalog for "Flame"*): A bare-metal, hardware-accelerated AI execution runtime designed to run massive LLMs and generative models on mobile devices with zero Cloud dependency.

---

## 📸 Overview

Liyab bridges the gap between large-scale AI architectures and resource-constrained mobile hardware. By routing tensor graphs to specialized NPUs and utilizing NVMe/UFS 4.1 zero-copy streaming, Liyab enables smooth 30B+ parameter model inference directly on consumer smartphones (Apple Silicon, Snapdragon, Dimensity, Tensor).

---

## 🔥 Key Features

* **Zero-Copy Memory-Mapped Streaming (`mmap`):** Stream model weights directly from internal flash memory to GPU/NPU buffers, bypassing standard RAM allocation limits and OS memory throttles.
* **Heterogeneous Dynamic Routing:** Automatically detects device SoC and routes model execution to the optimal native accelerator:
  * **Qualcomm Snapdragon:** Hexagon NPU via Qualcomm QNN SDK.
  * **MediaTek Dimensity:** APU via NeuroPilot SDK.
  * **Apple iOS:** Apple Neural Engine (ANE) via CoreML & Metal Performance Shaders.
  * **Universal Fallback:** High-performance Vulkan Compute Shaders for Adreno and Mali GPUs.
* **Sub-Byte & Mixed Quantization:** Full support for AWQ, GPTQ, GGUF (IQ2_XXS, IQ3_M, INT4) optimized for integer-heavy NPU pipelines.
* **Thermal & Power Aware:** Dynamic context window scaling and batched execution to prevent thermal throttling during extended inference sessions.

---

## 🏛️ System Architecture
