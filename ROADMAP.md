# EdgeForge Roadmap (private)

## v0.1.0 — Phase 1: Model ingestion (Week 3)
- [ ] Format detector (tflite, onnx, pt, pb)
- [ ] ONNX conversion pipeline
- [ ] Model analyser (ops, shapes, RAM/flash estimate)
- [ ] Compatibility report per target

## v0.2.0 — Phase 2: Optimisation (Week 6)
- [ ] INT8 post-training quantisation
- [ ] Operator fusion + constant folding
- [ ] MCU budget checker

## v0.3.0 — Phase 3: Code generation (Week 10)
- [ ] model.c / model.h generator
- [ ] inference_runner.c / .h
- [ ] memory_config.h
- [ ] RTOS glue (FreeRTOS, Zephyr)

## v1.0.0 — Public launch (Week 28)
- [ ] ESP32-S3 target
- [ ] edgeforge benchmark command
- [ ] PyPI package
- [ ] docs.edgeforge.dev

## v2.0.0 — NPU path
- [ ] PSoC AI Kit (Infineon ML accelerator)
- [ ] STM32N6 (Ethos-U55)
