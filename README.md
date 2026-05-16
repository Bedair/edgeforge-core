# edgeforge-core

Internal development repository for EdgeForge.

**Do not share this repository externally.**

## Structure

| Directory | Purpose |
|---|---|
| `edgeforge/` | CLI entry points and orchestration |
| `converter/` | Model format detection and ONNX IR conversion |
| `optimizer/` | Quantisation, pruning, operator fusion |
| `codegen/` | C/C++ code generation engine |
| `targets/` | MCU target profiles (source of truth) |
| `templates/` | Jinja2 templates for generated C/C++ |
| `tests/` | Full test suite |
| `scripts/` | Release and publishing automation |

## Setup

```bash
git clone git@github.com:edgeforge/edgeforge-core.git
cd edgeforge-core
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

## Releasing

```bash
# Build and publish to PyPI + update public repo
python scripts/release.py --version 0.1.0
```
