# edgeforge-core

Internal development repository for EdgeForge.

## Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Model ingestion, format detection, ONNX IR, analyzer, target loader | Complete |
| Phase 2 | Graph simplification, INT8 quantisation, MCU budget checking | Complete |
| Phase 3 | C/C++ code generation, Jinja2 templates, arena planning, RTOS glue | Complete |
| Phase 4 | Validation on 3 boards, benchmark, launch | Next |

## Setup

```bash
pip install -e ".[dev]"
pip install tf2onnx    # for TFLite conversion
pytest tests/ -v       # 52 tests, 0 failures
```

## CLI

```bash
edgeforge analyze  model.tflite
edgeforge optimize model.tflite   --mcu=stm32f407
edgeforge compile  model_opt.onnx --mcu=stm32f407 --rtos=freertos
edgeforge targets
```
