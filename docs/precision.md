# Data precision

A data file states its numbers to a definite number of digits, and that is a
claim. `0.3333` says a coordinate is known to about `1e-4` of a cell edge;
`0.33` says only `1e-2`. httk records that claim and uses it, so a matching
tolerance or an spglib `symprec` follows the data instead of being a constant
somebody guessed.

```python
from httk.atomistic import structure_tolerance
from httk.core import load

asu = load("measured.cif")
asu.coordinate_precision      # Fraction(1, 10000) — from the file's own digits
structure_tolerance(asu)      # a matching tolerance derived from the data
```

Automatic tolerance is always capped below half the nearest distinct-site
separation, including when the fallback is used. The nearest-image search
respects skew cells and only folds periodic directions. Coincident sites raise
`ValueError`: no positive automatic tolerance can distinguish them. Resolve
duplicate sites or supply an explicit tolerance to the operation when merging
them is intentional.

The full guide, {doc}`details/precision`, covers exactly what is recorded,
where precision comes from (digits, stated esds, format defaults), what it is
used for, and how it is served over OPTIMADE.
