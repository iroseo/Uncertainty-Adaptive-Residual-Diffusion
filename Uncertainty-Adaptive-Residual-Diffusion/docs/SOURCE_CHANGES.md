# Changes made for this source preview

This snapshot retains the research source structure and working model/training code, subject to the two explicit implementation redactions described in [RELEASE_SCOPE.md](../RELEASE_SCOPE.md). It is not a replacement implementation of the full method.

| Files | Release preparation change |
| --- | --- |
| `model/sr3_modules/uard.py` | Only the bodies of `UARDiffusion.q_sample` and `UARDiffusion.generate` are replaced by explicit `NotImplementedError`. Their signatures, the other methods, anchor/scale network, prediction heads and training losses are retained. |
| `release_status.py` | Added a standard-library release-availability check; no substitute model logic. |
| `main.py` | Uses an included S1 configuration by default. Parses arguments and checks availability before importing training dependencies. The S1/S2 training loop is retained. |
| `train_anchor.py` | Defers heavy imports until after argument parsing. S0 learning, losses, optimization and checkpoint behavior are retained. |
| `train_qxs.py` / `train_sar2opt.py` | Portable default data/GPU paths in the shared runner. The full S0-to-S2 pipeline checks availability before starting; `--prepare-only` remains usable. The SAR2Opt wrapper is retained. |
| `infer_uard.py` / `evaluate_uard.py` | Early release checks for unavailable diffusion generation. S0 anchor export remains possible through the S0 configuration in `infer_uard.py`. |
| `config/*.json` | Dataset/checkpoint paths are portable examples and default GPU is 0. Original architecture, losses and training update budgets are retained. |
| `core/metrics.py` | Removed the global HTTPS certificate-verification bypass. Numerical metric functions are retained. |
| `main.py`, `data/__init__.py`, `core/metrics.py` | Removed generated editor metadata where present. Copyright and source references are retained. |
| `main.py`, `model/`, `core/`, `data/` | Added a notice linking this provenance and modification record. |
| `tools/evaluate.py` | Retained the standalone evaluator's executable code. Its public module description identifies it as an existing-image evaluator. |
| Documentation, requirements and `.gitignore` | Prepared for this source-preview scope. |

Weights, datasets, experimental outputs, development backups, private review material and the original Git history are outside this snapshot. The complete research tree is not modified by this packaging process.
