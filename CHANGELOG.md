# Changelog (internal)

## [0.3.0-alpha] — Phase 3 complete
- codegen/model_extractor.py   — weight + tensor metadata extraction
- codegen/arena_planner.py     — TFLite Micro arena sizing + CCM detection
- codegen/codegen.py           — Jinja2 orchestrator
- templates/model_h.jinja2     — model.h with tensor defines + weight externs
- templates/model_c.jinja2     — model.c with hex weight arrays
- templates/memory_config_h.jinja2 — arena size + alignment + sanity checks
- templates/inference_runner_h.jinja2 — public C API
- templates/inference_runner_c.jinja2 — TFLite Micro wrapper
- templates/rtos_glue_c.jinja2 — FreeRTOS/Zephyr task-safe inference
- templates/CMakeLists.jinja2  — CMake build fragment
- templates/README.jinja2      — per-board integration guide
- edgeforge compile CLI command — fully implemented
- 26 new tests (52 total)

## [0.2.0-alpha] — Phase 2 complete
- optimizer/simplifier.py, quantizer.py, budget.py, optimizer.py
- 26 tests

## [0.1.0-alpha] — Phase 1 complete
- converter/detector.py, to_onnx.py, analyzer.py
- targets/loader.py + 3 MCU profiles
- 17 tests (+ 2 warnings — protobuf Python 3.14 deprecation, harmless)
