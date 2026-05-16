# Changelog (internal)

## [0.2.0-alpha] — Phase 2 complete

### Added
- optimizer/simplifier.py — constant folding, dead node removal, Conv+BN fusion
- optimizer/quantizer.py  — INT8 dynamic and static quantisation (onnxruntime)
- optimizer/budget.py     — MCU RAM/flash budget checker and suggestion engine
- optimizer/optimizer.py  — full pipeline orchestrator (simplify → quantise → check)
- edgeforge optimize CLI command — rich output with before/after stats and progress bars
- 26 new tests across test_simplifier, test_quantizer, test_budget, test_optimizer

## [0.1.0-alpha] — Phase 1 complete

### Added
- converter/detector.py — magic byte + extension format detection
- converter/to_onnx.py  — unified ONNX IR conversion (TFLite, PyTorch, TF)
- converter/analyzer.py — graph ops, RAM/flash estimates
- targets/loader.py     — MCU YAML profile loader + compatibility checker
- MCU profiles: stm32f407, psoc6, nrf52840
- edgeforge analyze CLI command
- edgeforge targets CLI command
- 17 tests
