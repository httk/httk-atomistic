# Asymmetric units: reading a CIF, expanding it, and going back again

A crystal is mostly repetition. A structure with 192 atoms in its cell may have
only two that are genuinely independent — the rest are copies placed by
symmetry. The *asymmetric unit* is that irreducible part, and
`httk.atomistic.ASUStructure` is httk's representation of it: a space group, one
site per orbit, and the values of each site's free parameters.

This example reads a CIF into an ASU, expands it to the full cell, recovers the
ASU from the expansion, and checks the crystal survived the trip.

## Why not just store the atoms?

Three reasons, and the third is the interesting one.

- It is smaller, and it says what the structure *means*: "sodium on Wyckoff `a`"
  rather than four coordinates that happen to be related.
- Symmetry is preserved rather than inferred. Expanding and re-detecting a
  structure is lossy in a way that storing the ASU is not.
- **It is exact.** Reduced coordinates, symmetry operations, and Wyckoff
  parameters are all rationals, and httk keeps them that way — as
  `Fraction`s, never as floats. Expanding an ASU is affine arithmetic over the
  rationals followed by an exact equality test, so it needs no coordinate grid,
  no snapping, and no tolerance at all.

## Any setting, including ones in no table

A space group can be written in many *settings* — different axis choices,
different origins. They are genuinely different coordinate systems: Wyckoff
letter `e` of space group 15 is `0,y,1/4` in the standard setting `15:b1` and
`1/4,0,z` in `15:c1`.

httk always records the Wyckoff data against the International Tables
**standard** setting, and carries a `SettingTransform` saying how to get from
there to the setting the structure is actually in. Because that transform is
*stored* rather than looked up, a setting that appears in no table is
representable just as well as a tabulated one — which is the point.

The transform is never *derived*. Searching for one that maps a group onto
itself is massively underdetermined: for space group 15 alone there are 192
valid candidates among just the small integer matrices, and they describe
*different crystals*. So httk stores the transform it was given and refuses to
guess one it was not.

## Where tolerance lives

In exactly one place: recognizing a structure that does not already carry its
symmetry.

A file writes `0.3333`. httk embeds that as the rational 3333/10000 — what the
file says — not as `float("0.3333")`, whose exact value is
6004199023210345/18014398509481984 and claims a precision nobody stated. The
recognition step then decides which Wyckoff position each site is *meant* to be
on, and the position supplies exact values for its fixed components. After that
everything is exact again.

So the two directions are not symmetric, and the example prints both:

- **expansion is lossless**, and
- **recognition is lossy at the tolerance level** — it snaps a measured structure onto an
  idealised one.

Run this file to see every step printed.

```{literalinclude} ../../examples/asu_from_cif.py
:language: python
:lines: 65-
```
