# Disordered structure-reading fixtures

This directory contains 33 CIFs separated from the ordered structure-reading corpus:
32 representatives with explicit partial site occupancy, plus the former disordered
space-group-217 fixture. The CIF reader combines co-located partial-occupancy rows into
one mixed `Species` and represents an unoccupied remainder explicitly as `vacancy`, so
the committed golden records pin the complete disorder chemistry.

The entries whose source was published in an IUCr journal were replaced by unmodified,
byte-for-byte COD entries of the same space group with explicit partial occupancy, from
non-IUCr journals and carrying only COD's public-domain header; they keep the space-group
file name, and `../SOURCES.md` lists their COD IDs (81, 100 and 177).
`168.cif` is instead a synthetic file written by the httk authors from the published
structure parameters of COD 2019502; it carries no journal text.

Regenerate the disorder golden after an intentional reading-semantics change:

```bash
python tools/generate_structreading_golden.py tests/fixtures/disorder \
  --output tests/data/disorder_structreading_golden.json.gz
```

Review the resulting diff before committing it. The disorder regression reads every
CIF in the normal test profile because this corpus is small.
