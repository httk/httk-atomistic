# Disorder, vacancies, and attached species: what a site can actually hold

Most crystal-structure libraries assume a site holds exactly one atom, and
encode a structure as a list of atomic numbers. Real materials modelling breaks
that assumption constantly — a substitutional alloy has a site that is 50% Fe
and 50% Ni, an off-stoichiometric compound has a site that is 90% occupied and
10% empty, and an adsorbed methyl group is a carbon with three hydrogens hanging
off it that nobody wants to place explicitly. *httk-atomistic* follows OPTIMADE
here: the object at a site is a **`Species`**, not an element, and a `Species`
is rich enough to say all three things.

A `Species` has a `name` (its label within the structure — it need *not* be a
chemical symbol), a tuple of `chemical_symbols`, and a matching tuple of
`concentration` values:

`Species(name="M", chemical_symbols=("Fe", "Ni"), concentration=(0.5, 0.5))`
: **Substitutional disorder.** The site is occupied, but by Fe half the time
  and Ni the other half. This is a statistical statement about an ensemble,
  not a statement about one particular atom.

`Species(name="Ti", chemical_symbols=("Ti", "vacancy"), concentration=(0.9, 0.1))`
: **A partial vacancy.** `"vacancy"` is one of the two pseudo-symbols OPTIMADE
  allows (the other is `"X"`, for *unknown*), and it names the *absence* of an
  atom. This site is empty 10% of the time. Note what this is not: it is not a
  Ti atom with a strange charge, and it is not a defect at a known location —
  it is an occupancy.

`Species(name="CH3", chemical_symbols=("C",), concentration=(1.0,), attached=("H",), nattached=(3,))`
: **Attached particles.** The site holds one carbon, fully occupied, with
  three hydrogens attached to it whose positions are deliberately not given.
  `attached` and `nattached` must always be supplied together and share a
  length. This is how you record a methyl group, a hydroxyl, or bound water
  without inventing coordinates you do not have.

**Which representations can carry this?** The Unitcell representation — a `UnitcellStructure`, and the
`UnitcellStructureView` over any backend — carries all of it, because it stores
the `Species` objects themselves. The OPTIMADE species dict (via
`PlainSpeciesView`) carries all of it too, because that is where the model
came from. But the **primitive** representation is the spglib-style
`(lattice, positions, numbers)` triple, and `numbers` is a list of bare atomic
numbers: there is no integer that means "90% Ti", and no integer that means
"carbon plus three hydrogens". So `PlainStructureView` does not silently
round, drop, or approximate — it raises `TypeError`, naming the species it
could not express.

`Species.is_single_element` is the predicate behind that check, and it is worth
knowing about directly: it is true only for a species made of exactly one *real*
element symbol (so not `"X"` and not `"vacancy"`) with nothing attached. Test it
before you reach for a symmetry library, a force field, or anything else that
speaks in atomic numbers.

Keep disordered structures in the Unitcell representation, and convert to
primitive only at the boundary of a tool that cannot represent disorder; a
`TypeError` reports unsupported species at that boundary.

```{literalinclude} ../../examples/disorder_and_vacancies.py
:language: python
:lines: 56-
```
