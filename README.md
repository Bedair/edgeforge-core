# edgeforge-core

Internal development repository for EdgeForge.

> ⚠️ This repository is the private development source.
> The public-facing repo (examples, docs, SDK headers) is at github.com/Bedair/edgeforge

---

## Project status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Model ingestion, format detection, ONNX IR, analyzer, target loader | ✅ Complete |
| Phase 2 | INT8 quantisation, operator fusion, MCU budget optimisation | 🔲 Next |
| Phase 3 | C/C++ code generation, RTOS glue, memory config | 🔲 Planned |
| Phase 4 | Validation on 3 boards, benchmark, public launch | 🔲 Planned |

---

## Repository structure

```
edgeforge-core/
├── edgeforge/
│   ├── cli.py                    # CLI entry point (analyze/optimize/compile/targets)
│   ├── converter/
│   │   ├── detector.py           # Magic byte + extension format detection
│   │   ├── to_onnx.py            # Any format → ONNX IR conversion
│   │   └── analyzer.py           # Graph ops, RAM/flash estimates
│   └── targets/
│       └── loader.py             # MCU YAML profile loader + compat checker
├── targets/                      # MCU target YAML profiles (source of truth)
│   ├── stm32f407.yaml
│   ├── psoc6.yaml
│   └── nrf52840.yaml
├── optimizer/                    # Phase 2: quantisation, pruning (coming)
├── codegen/                      # Phase 3: C/C++ generation engine (coming)
├── templates/                    # Phase 3: Jinja2 C templates (coming)
├── tests/
│   ├── test_detector.py          # 8 tests — format detection
│   └── test_loader.py            # 5 tests — target loading + compat
├── scripts/                      # Release automation (coming)
├── pyproject.toml
├── CHANGELOG.md
└── ROADMAP.md
```

---

## Setup

```bash
git clone git@github.com:Bedair/edgeforge-core.git
cd edgeforge-core
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the CLI

```bash
# Detect format, analyze graph, show board compatibility
edgeforge analyze model.tflite

# Filter to one target board
edgeforge analyze model.onnx --mcu=stm32f407

# List all supported MCU targets
edgeforge targets

# Detail on one target
edgeforge targets --mcu=psoc6
```

## Running tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_detector.py::test_tflite_by_magic        PASSED
tests/test_detector.py::test_tflite_by_extension    PASSED
tests/test_detector.py::test_onnx_by_extension      PASSED
tests/test_detector.py::test_torchscript_zip_magic  PASSED
tests/test_detector.py::test_savedmodel_directory   PASSED
tests/test_detector.py::test_unknown_format         PASSED
tests/test_detector.py::test_file_not_found         PASSED
tests/test_detector.py::test_describe_returns_dict  PASSED
tests/test_loader.py::test_all_targets_loads        PASSED
tests/test_loader.py::test_load_known_target        PASSED
tests/test_loader.py::test_load_unknown_target      PASSED
tests/test_loader.py::test_compatibility_fits       PASSED
tests/test_loader.py::test_compatibility_too_large  PASSED

13 passed in 0.03s
```

## Adding a new MCU target

Create a YAML file in `targets/` following the existing profiles.
Copy it to the public repo (`Bedair/edgeforge/targets/`) so the community can see it.

## Releasing

```bash
# Builds wheel, publishes to PyPI, creates GitHub release
python scripts/release.py --version 0.2.0
```
*(scripts/release.py coming in Phase 2)*
