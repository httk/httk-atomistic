# Structure-reading fixtures

This directory contains a complete 230-file full-occupancy structure-reading corpus with
one ordered representative for every space group. It derives from the httk v1 tutorial collection
(COD-derived, FINDSYM-processed, public domain), with later ordered replacements and with
space groups 93 and 101 imported from the Materials Project (mp-2647159, mp-3211449;
CC-BY 4.0, attribution headers inside the files). Structures with partial occupancy or
other explicit site disorder have their own regression golden in the sibling directory.

Regenerate the full golden after an intentional reading-semantics change:

```bash
python tools/generate_structreading_golden.py
```

Review the resulting diff before committing it. The normal test profile uses a selected
25-file representative subset; the extended profile checks this full ordered corpus.
