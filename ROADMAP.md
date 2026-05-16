# EdgeForge Roadmap (private)

## v0.1.0 ✅ Phase 1 — Model ingestion
- [x] Format detector (tflite, onnx, pt, pb, SavedModel)
- [x] ONNX conversion pipeline
- [x] Model analyser (ops, shapes, RAM/flash estimate)
- [x] Compatibility report per target
- [x] 17 tests passing

## v0.2.0 ✅ Phase 2 — Optimisation
- [x] Graph simplifier (constant folding, dead nodes, BN fusion)
- [x] INT8 dynamic quantisation
- [x] INT8 static quantisation (with calibration data)
- [x] MCU budget checker with actionable suggestions
- [x] Full pipeline orchestrator
- [x] 26 new tests (43 total)

## v0.3.0 🔲 Phase 3 — Code generation
- [ ] model.c / model.h generator (Jinja2 templates)
- [ ] inference_runner.c / .h
- [ ] memory_config.h
- [ ] RTOS glue: FreeRTOS, Zephyr
- [ ] edgeforge compile command

## v1.0.0 🔲 Phase 4 — Validation and launch
- [ ] Keyword spotting demo on STM32F407, PSoC6, nRF52840
- [ ] edgeforge benchmark command
- [ ] PyPI package
- [ ] docs.edgeforge.dev
