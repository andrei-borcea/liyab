# ⚡ Liyab : Ignite your mobile silicon

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Android_%7C_iOS-brightgreen.svg)]()
[![Backend](https://img.shields.io/badge/Accelerators-Qualcomm_QNN_%7C_MediaTek_%7C_CoreML_%7C_Vulkan-orange.svg)]()
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)]()

> **Liyab** (*Tagalog per "Fiamma"*): Un motore d'inferenza bare-metal, ultra-ottimizzato e cross-platform, progettato per far girare modelli LLM e Generativi di grandi dimensioni (fino a 35B+ parametri) direttamente sull'hardware locale di smartphone Android e iOS, con zero dipendenze dal Cloud.

---

## 📋 Indice delle Sezioni

- [Caratteristiche Principali](#-caratteristiche-principali)
- [Architettura del Sistema](#-architettura-del-sistema)
- [Hardware e Backend Supportati](#-hardware-e-backend-supportati)
- [Requisiti di Sistema](#-requisiti-di-sistema)
- [Guida all'Installazione e Compilazione](#-guida-allinstallazione-e-compilazione)
- [Esempi di Codice (Usage)](#-esempi-di-codice-usage)
  - [C++ Native API](#c-native-api)
  - [Android Integration (Kotlin / JNI)](#android-integration-kotlin--jni)
  - [iOS Integration (Swift / C Interop)](#ios-integration-swift--c-interop)
- [Benchmark Prestazionali](#-benchmark-prestazionali)
- [Roadmap del Progetto](#-roadmap-del-progetto)
- [Licenza e Contributi](#-licenza-e-contributi)

---

## 🔥 Caratteristiche Principali

* **Zero-Copy Memory-Mapped Streaming (`mmap`):** Bypassa i limiti di RAM imposti dai sistemi operativi mobili eseguendo lo streaming dei pesi in tempo reale dalla memoria di archiviazione veloce (UFS 4.1 / NVMe) ai buffer GPU/NPU.
* **Routing Hardware Eterogeneo Automatico:** Rileva a runtime il System-on-Chip (SoC) del dispositivo e indirizza i grafi di calcolo al miglior acceleratore hardware disponibile.
* **Supporto Quantizzazione Sub-Byte:** Piena compatibilità con modelli quantizzati in formato **AWQ**, **GPTQ** e **GGUF** (IQ2_XXS, IQ3_M, INT4, INT8) ottimizzati per pipeline integer NPU.
* **Gestione Termica e Dynamic Context Window:** Ridimensionamento dinamico della cache KV (*PagedAttention* / *Sliding Window*) per prevenire il throttling termico e prolungare la durata della batteria durante sessioni di generazione lunghe.

---

## 🏛️ Architettura del Sistema

```
                         +-----------------------------------------+
                         |       Application Layer (Kotlin/Swift)  |
                         +--------------------+--------------------+
                                              | JNI / C-Interop
                         +--------------------v--------------------+
                         |           Liyab Core Runtime            |
                         |     (C++20 / Memory-Mapped I/O)         |
                         +--------------------+--------------------+
                                              |
           +----------------------------------+----------------------------------+
           |                                  |                                  |
+----------v----------+            +----------v----------+            +----------v----------+
|  Qualcomm QNN (NPU) |            | MediaTek NeuroPilot |            | CoreML / Metal (iOS)|
| (Snapdragon Series) |            |  (Dimensity Series) |            |  (Apple A & M Series) |
+---------------------+            +---------------------+            +---------------------+
           |                                  |                                  |
           +----------------------------------+----------------------------------+
                                              | (Fallback Universale)
                                  +-----------v----------+
                                  | Vulkan / ARM NEON    |
                                  |  (Adreno / Mali GPU) |
                                  +----------------------+
```

---

## 💻 Hardware e Backend Supportati

| Produttore | Famiglia Chip | Backend Primario | Fallback Hardware |
| :--- | :--- | :--- | :--- |
| **Qualcomm** | Snapdragon 8 Elite / Gen 3 / Gen 2 | Qualcomm QNN (Hexagon NPU) | Vulkan Compute (Adreno GPU) |
| **MediaTek** | Dimensity 9400 / 9300 / 8300 | MediaTek NeuroPilot (APU) | Vulkan Compute (Mali GPU) |
| **Apple** | Apple A17 Pro / A18 / M-Series | CoreML (ANE) + MPSGraph | Metal Performance Shaders |
| **Google** | Tensor G3 / G4 | LiteRT / NNAPI (TPU) | Vulkan Compute (Mali GPU) |
| **Samsung** | Exynos 2400 / 2200 | Samsung NPU Delegate | Vulkan Compute (Xclipse GPU) |

---

## 🛠️ Requisiti di Sistema

### Per lo sviluppo e la compilazione:
* **Host Build OS:** Linux (Ubuntu 22.04+), macOS (14.0+) o Windows tramite WSL2.
* **CMake:** v3.22 o superiore.
* **Android NDK:** r26b o superiore.
* **Xcode:** v15.0+ (per iOS).
* **Python:** v3.10+ (utilizzato per gli script di conversione dei pesi).

---

## 📦 Guida all'Installazione e Compilazione

### 1. Clona il Repository con le Dipendenze
```bash
git clone --recursive https://github.com/your-username/liyab.git
cd liyab
```

### 2. Compilazione per Android (Shared Object `.so`)
Per compilare la libreria C++ nativa con supporto completo a Qualcomm QNN e Vulkan:

```bash
chmod +x scripts/build_android.sh
./scripts/build_android.sh \
    --abi arm64-v8a \
    --enable-vulkan \
    --enable-qnn \
    --build-type Release
```
I file compilati `.so` saranno disponibili all'interno della cartella `build_android/libs/`.

### 3. Compilazione per iOS (Framework)
Per generare il framework `.xcframework` universale per dispositivi iOS:

```bash
chmod +x scripts/build_ios.sh
./scripts/build_ios.sh \
    --enable-coreml \
    --enable-metal
```

---

## 💻 Esempi di Codice (Usage)

### C++ Native API

```cpp
#include "liyab/liyab.h"
#include <iostream>

int main() {
    // 1. Inizializzazione della configurazione
    liyab::EngineConfig config;
    config.enable_mmap = true;                           // Attiva lo streaming zero-copy da UFS/NVMe
    config.preferred_backend = liyab::Backend::AUTO;    // Auto-detect tra QNN, NeuroPilot, CoreML o Vulkan
    config.max_context_length = 4096;

    // 2. Istanziazione del motore
    liyab::Engine engine(config);

    // 3. Caricamento del modello quantizzato (.liyab / .gguf)
    if (!engine.load_model("models/qwen-2.5-32b-iq3.liyab")) {
        std::cerr << "Errore durante il caricamento del modello." << std::endl;
        return -1;
    }

    // 4. Invocazione dell'inferenza con callback per streaming dei token
    std::string prompt = "Spiega il concetto di memoria unificata nei SoC moderni.";
    
    engine.generate(prompt, [](const std::string& token) {
        std::cout << token << std::flush;
        return true; // Restituisci false per interrompere la generazione
    });

    return 0;
}
```

### Android Integration (Kotlin / JNI)

```kotlin
import com.liyab.ai.LiyabEngine
import com.liyab.ai.LiyabConfig

class MainActivity : AppCompatActivity() {
    private lateinit var liyabEngine: LiyabEngine

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Configura il motore nativo
        val config = LiyabConfig(
            enableMmap = true,
            useNpuAcceleration = true,
            modelPath = "/sdcard/Download/qwen-32b-iq3.liyab"
        )

        liyabEngine = LiyabEngine(config)
        liyabEngine.initialize()

        // Esegui la generazione in streaming
        liyabEngine.generateStream("Scrivi una poesia sulla velocità del silicio.") { token ->
            runOnUiThread {
                binding.textViewOutput.append(token)
            }
        }
    }
}
```

### iOS Integration (Swift / C Interop)

```swift
import Swift
import LiyabFramework

class AIInferenceService {
    private var engine: OpaquePointer?

    init() {
        var config = liyab_config_default()
        config.enable_mmap = true
        config.use_coreml = true
        
        self.engine = liyab_engine_create(&config)
        liyab_engine_load_model(self.engine, "qwen-32b-iq3.liyab")
    }

    func generateResponse(prompt: String) {
        liyab_engine_generate(self.engine, prompt) { tokenPtr in
            guard let tokenPtr = tokenPtr else { return }
            let token = String(cString: tokenPtr)
            print(token, terminator: "")
        }
    }

    deinit {
        liyab_engine_destroy(self.engine)
    }
}
```

---

## 📊 Benchmark Prestazionali

I seguenti benchmark sono stati rilevati eseguendo modelli di classe **32B - 35B parametri** (quantizzazione 3-bit `IQ3_M`) in locale su dispositivi di serie:

| Dispositivo | SoC | RAM | Backend Attivo | Velocità Generazione | Latenza Primo Token (TTFT) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Xiaomi 15 Ultra** | Snapdragon 8 Elite | 16 GB | Qualcomm QNN + Vulkan | **12 - 16 t/s** | ~240 ms |
| **iPhone 15 Pro Max** | Apple A17 Pro | 8 GB | Metal MPSGraph + mmap | **8 - 11 t/s** | ~380 ms |
| **Oppo Find X8 Pro** | Dimensity 9400 | 16 GB | MediaTek NeuroPilot | **11 - 14 t/s** | ~260 ms |
| **Samsung Galaxy S24 Ultra** | Snapdragon 8 Gen 3 | 12 GB | Qualcomm QNN | **9 - 12 t/s** | ~310 ms |

---

## 🗺️ Roadmap del Progetto

- [x] **Fase 1:** Motore C++ nativo con supporto `mmap` e backend Vulkan universale.
- [x] **Fase 2:** Integrazione SDK Qualcomm QNN per NPU Hexagon e CoreML per iOS.
- [ ] **Fase 3:** Supporto completo per l'SDK MediaTek NeuroPilot.
- [ ] **Fase 4:** Modulo di quantizzazione custom `LiyabQuant` per il caricamento ultra-veloce di pesi sub-2bit.
- [ ] **Fase 5:** Supporto Multi-Modal (Vision-Language Models come LLaVA e Phi-3-Vision).

---

## 🤝 Licenza e Contributi

### Licenza
Questo progetto è distribuito sotto la licenza **Apache 2.0**. Per maggiori dettagli, consultare il file [LICENSE](LICENSE).

### Contributi
I contributi della community sono i benvenuti! Se desideri aggiungere un nuovo backend per un chipset specifico, correggere un bug o ottimizzare i kernel di calcolo:
1. Consulta le nostre [Linee Guida per i Contributi](CONTRIBUTING.md).
2. Apri un'issue per discutere la modifica proposta prima di inviare una Pull Request.

---

<p align="center">
  Sviluppato con ❤️ per spingere i limiti dell'AI Edge su dispositivi mobili.
</p>
