# Structure-reading fixtures

These CIFs are a representative subset of the httk v1 tutorial corpus: COD-derived,
FINDSYM-processed files with Wyckoff labels. COD data is public domain.

Regenerate the full golden after an intentional reading-semantics change:

```bash
python tools/generate_structreading_golden.py \
  /home/rar/Documents/containers/devel/agents/httk2/old/httk/Tutorial/tutorial_data/all_spacegroups/cifs
```

Review the resulting diff before committing it. The fixtures intentionally cover every
crystal system, the named boundary groups, and disordered examples 16 and 168.
