# Data precision: letting the file choose the tolerance

Every tolerance in a structure pipeline is a guess about how good the data is.
Match two coordinates within `1e-3`? Hand spglib a `symprec` of `1e-5`? Those
numbers usually come from habit rather than from the data, and when they are
wrong they are wrong silently.

But a file already says how good it is. A coordinate written `0.3333` claims a
precision of about `1e-4` of a cell edge; written `0.33` it claims only `1e-2`.
A cell edge written `5.6402(3)` states its uncertainty outright. httk records
those claims and derives its tolerances from them.

This example shows where the numbers come from, how they are converted, and one
case where deriving the tolerance changes the answer from wrong to right.

## The two values

A structure records two precisions, each on the object that owns the numbers it
describes:

- **coordinate precision** on `Sites`, in *fractional* units — reduced
  coordinates are dimensionless, and a `Sites` has no cell to convert with;
- **basis precision** on `Cell`, in the cell's *length* units.

Both are exact rationals, so nothing picks up binary noise on the way to
becoming a tolerance. `None` means unknown, which is a real answer and not the
same as claiming exactness.

`UnitcellStructure.cartesian_precision()` combines them into the number a tolerance
actually wants: a distance. That conversion is not cosmetic — the same `1e-4`
coordinates mean `5e-4` in a 5 A cell and `3e-3` in a 30 A one.

## The rules

Three, and each exists because the naive alternative is wrong:

- **The coarsest value wins.** A structure is only as precisely stated as its
  least precisely stated number.
- **A stated uncertainty widens the claim.** `5.6402(3)` is good to `3e-4`, not
  to the `1e-4` its four decimals alone suggest.
- **An exact rational makes no claim at all.** `1/3` states a value rather than a
  measurement, and must not drag a table down to the width of its last digit.

Run this file to see every step printed.

```{literalinclude} ../../examples/precision.py
:language: python
:lines: 46-
```
