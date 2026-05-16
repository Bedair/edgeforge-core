# edgeforge-core

Internal development repository for EdgeForge — a toolchain for deploying AI models to embedded MCUs.

> ⚠️ Private repository — do not share externally.

## Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Model ingestion, format detection, ONNX IR, analyzer, target loader | ✅ Complete |
| Phase 2 | Graph simplification, INT8 quantisation, MCU budget checking | ✅ Complete |
| Phase 3 | C/C++ code generation, RTOS glue, memory config | 🔲 Next |
| Phase 4 | Validation on 3 boards, benchmark, launch | 🔲 Planned |

## Repository structure

```
edgeforge-core/
├── edgeforge/
│   ├── cli.py                         # CLI: analyze / optimize / compile / targets
│   ├── converter/
│   │   ├── detector.py                # Magic byte + extension format detection
│   │   ├── to_onnx.py                 # Any format → ONNX IR
│   │   └── analyzer.py                # Graph ops, RAM/flash estimates
│   ├── targets/
│   │   └── loader.py                  # MCU YAML profile loader + compat checker
│   └── optimizer/
│       ├── simplifier.py              # Constant folding, dead node removal, BN fusion
│       ├── quantizer.py               # INT8 dynamic + static quantisation
│       ├── budget.py                  # MCU RAM/flash budget checker + suggestions
│       └── optimizer.py              # Orchestrator: full pipeline entry point
├── targets/                           # MCU target YAML profiles
│   ├── stm32f407.yaml
│   ├── psoc6.yaml
│   └── nrf52840.yaml
├── tests/                             # 43 tests — all passing
├── optimizer/                         # Phase 3 placeholder
├── codegen/                           # Phase 3 placeholder
├── templates/                         # Phase 3 Jinja2 templates (coming)
├── scripts/                           # Release automation (coming)
├── conftest.py                        # Pytest config — patches targets path
└── pyproject.toml
```

## Setup

```bash
git clone git@github.com:Bedair/edgeforge-core.git
cd edgeforge-core
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/ -v
```

Expected: **43 passed, 0 failed**

## CLI usage

```bash
# Analyze any model — format, graph, RAM/flash estimate, board compatibility
edgeforge analyze model.tflite
edgeforge analyze model.onnx --mcu=stm32f407

# Optimize — simplify + quantise + check MCU budget
edgeforge optimize model.tflite --mcu=stm32f407
edgeforge optimize model.onnx   --mcu=psoc6 --mode=static --calibration-dir ./cal_data/

# List supported MCU targets
edgeforge targets
edgeforge targets --mcu=stm32f407
```

## Commit this milestone

```bash
git add .
git commit -m "feat: Phase 2 complete — 43 tests passing

- edgeforge/optimizer/simplifier.py  — constant folding, dead node removal, BN fusion
- edgeforge/optimizer/quantizer.py   — INT8 dynamic + static quantisation (onnxruntime)
- edgeforge/optimizer/budget.py      — MCU RAM/flash budget checker + actionable suggestions
- edgeforge/optimizer/optimizer.py   — full pipeline orchestrator
- edgeforge/cli.py                   — working edgeforge optimize command with rich output
- tests/                             — 26 new tests (Phase 2), 17 carried from Phase 1
"
git push
```
