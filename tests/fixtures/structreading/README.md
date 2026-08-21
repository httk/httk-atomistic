# Structure-reading fixtures

This directory contains the 198-file full-occupancy structure-reading corpus retained
after separating explicit partial-occupancy structures. It derives from the 227-file
httk v1 tutorial collection (COD-derived, FINDSYM-processed, public domain), completed
in August 2026 with space groups 93 and 101 imported from the Materials Project
(mp-2647159, mp-3211449; CC-BY 4.0, attribution headers inside the files) and space
group 209 from COD (1528204, public domain). Structures with partial occupancy or other
explicit site disorder live in the sibling `disorder` directory and have a separate
regression golden. Space group 217 has a full-occupancy replacement here while its
former disordered representative remains in that sibling corpus.

Regenerate the full golden after an intentional reading-semantics change:

```bash
python tools/generate_structreading_golden.py
```

Review the resulting diff before committing it. The normal test profile uses a selected
19-file representative subset; the extended profile checks this full ordered corpus.
