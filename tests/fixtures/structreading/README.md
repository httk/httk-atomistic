# Structure-reading fixtures

This directory vendors the complete 227-file httk v1 tutorial corpus: COD-derived,
FINDSYM-processed CIFs with Wyckoff labels. It has one CIF per space group except 93,
101, and 209, which had no CIF in the original collection. COD data is public domain.

Regenerate the full golden after an intentional reading-semantics change:

```bash
python tools/generate_structreading_golden.py
```

Review the resulting diff before committing it. The normal test profile uses a selected
25-file representative subset; the extended profile checks the full vendored corpus.
