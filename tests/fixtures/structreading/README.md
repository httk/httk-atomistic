# Structure-reading fixtures

This directory contains a complete 230-file full-occupancy structure-reading corpus with
one ordered representative for every space group. It derives from the httk v1 tutorial collection
(COD-derived, FINDSYM-processed, public domain), with later ordered replacements and with
space groups 93 and 101 imported from the Materials Project (mp-2647159, mp-3211449;
CC-BY 4.0, attribution headers inside the files). Structures with partial occupancy or
other explicit site disorder have their own regression golden in the sibling directory.

The entries whose source was published in an IUCr journal (IUCr attribution terms, not public
domain), and space group 90 (an AMCSD attribution header), were replaced by unmodified,
byte-for-byte COD entries of the same space group, declared in its standard setting, fully
ordered, from non-IUCr journals and carrying only COD's public-domain header. They keep the
space-group file name; `../SOURCES.md` lists the COD ID of every file: 1, 14, 15, 32, 37, 41,
43, 59, 68, 80, 83, 84, 90, 96, 102, 103, 105, 106, 107, 113, 114, 123, 124, 125, 137, 144,
151, 172, 189, 202, 206, 207, 213, 225 and 226.

Regenerate the full golden after an intentional reading-semantics change:

```bash
python tools/generate_structreading_golden.py
```

Review the resulting diff before committing it. The normal test profile uses a selected
25-file representative subset; the extended profile checks this full ordered corpus.
