# Structure-reading fixtures

This directory vendors the 227-file httk v1 tutorial collection (COD-derived,
FINDSYM-processed, public domain) completed in August 2026 with the three space groups
it lacked — 93 and 101 imported from the Materials Project (mp-2647159, mp-3211449;
CC-BY 4.0, attribution headers inside the files) and 209 from COD (1528204, public
domain) — giving one CIF per space group, 230 files total.

Regenerate the full golden after an intentional reading-semantics change:

```bash
python tools/generate_structreading_golden.py
```

Review the resulting diff before committing it. The normal test profile uses a selected
25-file representative subset; the extended profile checks the full vendored corpus.
