# Disordered structure-reading fixtures

This directory contains 33 CIFs separated from the ordered structure-reading corpus:
32 representatives with explicit partial site occupancy, plus the former disordered
space-group-217 fixture. Some co-located different-species sites cannot yet be
represented losslessly by the ASU model; their committed golden records deliberately
pin the current repair behavior, including its warning and chemistry loss.

Regenerate the disorder golden after an intentional reading-semantics change:

```bash
python tools/generate_structreading_golden.py tests/fixtures/disorder \
  --output tests/data/disorder_structreading_golden.json.gz
```

Review the resulting diff before committing it. The disorder regression reads every
CIF in the normal test profile because this corpus is small.
