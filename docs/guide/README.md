# Documentation

[← Project README](../../README.md)

| Guide | Contents |
| :--- | :--- |
| [Project overview](overview.md) | Why Liyab, Design principles, What it does today |
| [Build and run](building.md) | Quick start |
| [Engine architecture](architecture.md) | Architecture |
| [Models, formats and reference tools](models.md) | Supported models |
| [Memory, streaming and serving](memory-and-serving.md) | Models larger than RAM, Context, memory and serving |
| [Power, speculation, pooling and adapters](inference-features.md) | Heat and power, Speculative decoding, Embeddings and reranking, LoRA adapters |
| [Engine API and CLI examples](api.md) | Using the engine |
| [App and local API](app.md) | The Liyab app |
| [Visual design](design.md) | Look and feel |
| [Measurements and correctness](benchmarks.md) | Measured results |
| [Testing, limitations and development](development.md) | Experimental modules, Testing, Roadmap, Known limitations, Project layout, Contributing, License |

| Model evidence | Contents |
| :--- | :--- |
| [Model descriptions](../model-descriptions.md) | Registry facts, source tracing, trained-checkpoint gates, tokenizer/export corrections |
| [Stage-1 coverage](../model-coverage-stage1.md) | Per-section results, failure controls, host benchmarks, deferred phone checks and stage 2 |
| [Reproduction archive](../validation/models-stage1/README.md) | Pinned configurations, commands, compatibility patches and numerical results |
| [Raw benchmark data](../benchmarks/results.json) | Measurements behind the speed charts |
| [Cold-expert study](../research/cold-experts.md) | Smaller representations for rarely used experts |
