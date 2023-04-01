# Ring Attention Context Parallel

Technique from 2023, reimplemented in 2026.

Sequence-sharded ring attention using PyTorch distributed point-to-point communication. Implementation, distributed correctness tests, and benchmark commands are included. GPU performance results are pending.

## Reproduce

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest -q
PYTHONPATH=src python scripts/smoke.py
```

## References

- Liu et al., *Ring Attention with Blockwise Transformers for Near-Infinite Context*, 2023.
