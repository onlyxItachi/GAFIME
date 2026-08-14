# GAFIME v1 Golden Oracle

These fixtures preserve selected legacy-engine outputs captured before the v1
rewrite moved planning, scheduling, execution, and reduction behind the Rust
orchestration spine.

The original P0 phase used the old Core backend as its oracle. The files and
generator remain as historical compatibility evidence; they do not describe a
shipping legacy execution path.

The original capture command was:

```bash
python3 tests/golden/generate_golden.py --update
```

The historical check command was:

```bash
python3 tests/golden/generate_golden.py --check
```

Both generator modes depend on the removed `_analyze_legacy` boundary and are
not current v1 validation commands. Do not regenerate these fixtures during
normal v1 work. Use `tests/release_measure/v1_architecture_gate.py` and the
current release-contract suite for shipping-runtime validation.
