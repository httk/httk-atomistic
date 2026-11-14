# Composition and formula families

The chemical-formula family gives one interface to elemental amounts and their
canonical renderings. `Composition` is the immutable elemental value,
`ChemicalFormula` is a canonical reduced formula, and `AnonymousFormula` is the
canonical OPTIMADE anonymous formula of a composition.
`CompositionView`, `ChemicalFormulaView`, and `AnonymousFormulaView` present
compatible backends as those three values;
the composition view is lazy, while the two formula views are eager. The
information ordering is `composition ⊃ reduced formula ⊃ anonymous formula`:
each step to the right deliberately drops information, so conversion back to a
more informative form can raise `ValueError` rather than inventing element
identities or silently accepting an incomplete composition.

`Formulatype` (with the eager `FormulatypeView`) uses the same anonymous text
grammar but counts *sites*, not elements: how many sites each species class of a
structure, structuretype, prototype or protostructure occupies. Disorder and
partial occupancy do not affect it, so it is the prefix of every httk prototype
and protostructure label. See [Formulatype and site counts](#formulatype-and-site-counts).

## Three levels of information

`Composition` stores positive, exact elemental amounts (and, when supplied,
uncertainties, completeness, normalization, and diagnostics). Its amounts are
ordered alphabetically by real element symbol. A complete composition can
render `chemical_formula_reduced`, with integer coefficients divided by their
greatest common divisor, and `chemical_formula_anonymous`.

`ChemicalFormula` is a strict canonical reduced formula: element symbols are
alphabetical and coefficients are GCD-reduced. `AnonymousFormula` and
`Formulatype` use consecutive labels `A`, `B`, ... and non-increasing coefficients. Both are
subclasses of `str`. Their corresponding views retain a backend, so
`unwrap()` can recover it.

The directionality rules follow the information ordering:

- `ChemicalFormulaView` can present a complete real-element composition, but
  raises for anonymous labels, incomplete compositions (including an unknown
  `"X"` species), or an empty composition.
- `AnonymousFormulaView` can anonymize a complete real-element composition and
  can preserve an already-anonymous formula. It raises for incomplete or empty
  compositions.
- `FormulatypeView` accepts the same inputs only when every amount is an
  integer, read as one species per site at full occupancy; a fractional amount
  raises `ValueError`.
- `CompositionView` can project real-element data, but raises when its source
  is already anonymous: it cannot invent the missing elements.

## Structures stay connected

`Structure.composition` is a lazy `CompositionView`. The projection is not
performed until composition data is read. `CompositionView(structure)` is the
explicit equivalent. In contrast, `structure.formula` is eager and is a genuine
`str` subclass (`ChemicalFormulaView`), so it raises `ValueError` when the
composition is incomplete rather than returning `None`. Use
`structure.chemical_formula_reduced` when the `str | None` escape hatch is
needed. The formula and composition views retain the source:

```python
from httk.atomistic import (
    ChemicalFormulaView,
    CompositionView,
    Species,
    UnitcellStructure,
)

structure = UnitcellStructure(
    [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
    [[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]],
    [Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))],
    ["Na", "Cl"],
)

composition = structure.composition
assert isinstance(composition, CompositionView)
assert composition.amounts == (("Cl", 1), ("Na", 1))
assert structure.formula == "ClNa"
assert isinstance(structure.formula, str)
assert ChemicalFormulaView(composition).unwrap() is structure
assert CompositionView(structure).unwrap() is structure
```

For an incomplete composition, the optional property and the eager property
intentionally differ:

```python
incomplete = UnitcellStructure(
    [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
    [[0, 0, 0]],
    [Species("unknown", ("X",), (1,))],
    ["unknown"],
)
assert incomplete.chemical_formula_reduced is None
try:
    incomplete.formula
except ValueError as error:
    assert "incomplete" in str(error)
else:
    raise AssertionError("an incomplete composition must not produce formula")
```

## Accepted constructions

The family accepts a mapping of element amounts, a canonical reduced formula
string, a canonical anonymous formula string, a normalized composition record,
and structure/prototype objects. The view constructors make the desired
presentation explicit:

```python
from fractions import Fraction

from httk.atomistic import (
    AnonymousFormulaView,
    ChemicalFormulaView,
    CompositionView,
    FormulatypeView,
)

assert CompositionView({"Al": 2, "O": 3}).chemical_formula_reduced == "Al2O3"
assert ChemicalFormulaView("Al2O3") == "Al2O3"
assert AnonymousFormulaView({"Fe": Fraction(1, 2), "O": 1}) == "A2B"
assert FormulatypeView("A3B2") == "A3B2"
assert FormulatypeView({"Al": 2, "O": 3}) == "A3B2"
try:
    FormulatypeView({"Fe": Fraction(1, 2), "O": 1})
except ValueError:
    pass
else:
    raise AssertionError("a fractional amount says nothing about site counts")
```

A `str` in `ChemicalFormulaLike` is always a formula, never a filename. Load a
file into a structure first; formula views do not perform file I/O. The reverse
guard is equally deliberate: structure views reject formula objects, because a
formula does not contain a cell or sites.

## Anonymous labels and rendering

Anonymous labels are presentation labels. A dummy species carries its identity
in `labels`, not in `chemical_symbols`:

```python
from httk.atomistic import Species

dummy = Species("A", ("X",), (1,), labels=("A",))
assert dummy.labels == ("A",)
assert dummy.chemical_symbols == ("X",)
```

When real elements are anonymized, amounts are sorted by descending amount,
then by alphabetical element symbol; the resulting positions receive
`anonymous_symbol(0)`, `anonymous_symbol(1)`, and so on. Rendering is GCD-
reduced at the same time. Thus equal amounts use alphabetical order, while a
composition with ratios 4:12:4 becomes reduced `FeO3Sm` and anonymous `A3BC`.

## Formulatype and site counts

`FormulatypeView` of a structural object counts, for each species class, the
sites that class occupies, reduces the counts by their GCD and labels them `A`,
`B`, ... largest first. A structure's classes are its species names: a mixed
(Fe,Ni) species on one position is one class, and a half-occupied site counts
fully. The classes of a prototype or protostructure are its label classes,
counted by conventional Wyckoff multiplicity. The text is the GCD-reduced
formula, while `FormulatypeView(...).amounts` keeps the per-cell class counts.
Every level's `anonymous_formula` is this value, and it is the prefix of the
level's label:

```python
from fractions import Fraction

from httk.atomistic import AnonymousFormulaView, BareProtostructure, FormulatypeView, Species

mixed = Species("FeNi", ("Fe", "Ni"), (Fraction(1, 2), Fraction(1, 2)))
value = BareProtostructure(221, [("a", mixed), ("c", "O")])
assert str(value.label).startswith("A3B_")
assert value.anonymous_formula == FormulatypeView(value) == "A3B"
assert AnonymousFormulaView(value) == "A6BC"
```

The OPTIMADE `chemical_formula_anonymous` of a structure stays composition-based
(`AnonymousFormulaView` gives the same value). It agrees with the `Formulatype`
only for an ordered, fully occupied structure whose species names are distinct
elements.

Each view keeps its own meaning when handed the other: `FormulatypeView` of an
`AnonymousFormulaView` of a structural object recounts that object's sites, and
`AnonymousFormulaView` of such a `FormulatypeView` re-projects its composition.
A `kind=` hint other than `"sites"` on a structural source raises `ValueError`
rather than switching `FormulatypeView` to composition.

See {doc}`prototypes` and the full guide, {doc}`details/structural_classes`,
for where `ChemicalFormula` and `Formulatype` sit within the full
material-information taxonomy.
