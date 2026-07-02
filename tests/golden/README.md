# GAFIME v1 Golden Oracle

These fixtures freeze selected legacy-engine outputs before the v1 rewrite moves
planning, scheduling, execution, and reduction behind the Rust orchestration spine.

P0 uses the old core backend as the oracle. Later phases must keep these outputs
within the phase tolerance before deleting or replacing legacy code paths.

Regenerate intentionally with:

```bash
python3 tests/golden/generate_golden.py --update
```

Check the current worktree against the fixtures with:

```bash
python3 tests/golden/generate_golden.py --check
```
