# Composition and formula families

The chemical-formula family gives one interface to elemental amounts and their
canonical renderings. `Composition` is the immutable elemental value,
`ChemicalFormula` is a canonical reduced formula, and `Formulatemplate` is a
canonical OPTIMADE anonymous formula (`AnonymousFormula` remains an alias).
`CompositionView`, `ChemicalFormulaView`, and `FormulatemplateView` present
compatible backends as those three values;
the composition view is lazy, while the two formula views are eager. The
information ordering is `composition ⊃ reduced formula ⊃ anonymous formula`:
each step to the right deliberately drops information, so conversion back to a
more informative form can raise `ValueError` rather than inventing element
identities or silently accepting an incomplete composition.

## Three levels of information

`Composition` stores positive, exact elemental amounts (and, when supplied,
uncertainties, completeness, normalization, and diagnostics). Its amounts are
ordered alphabetically by real element symbol. A complete composition can
render `chemical_formula_reduced`, with integer coefficients divided by their
greatest common divisor, and `chemical_formula_anonymous`.

`ChemicalFormula` is a strict canonical reduced formula: element symbols are
alphabetical and coefficients are GCD-reduced. `Formulatemplate` uses
consecutive labels `A`, `B`, ... and non-increasing coefficients. Both are
subclasses of `str`. Their corresponding views retain a backend, so
`unwrap()` can recover it.

The directionality rules follow the information ordering:

- `ChemicalFormulaView` can present a complete real-element composition, but
  raises for anonymous labels, incomplete compositions (including an unknown
  `"X"` species), or an empty composition.
- `FormulatemplateView` can anonymize a complete real-element composition and
  can preserve an already-anonymous formula. It raises for incomplete or empty
  compositions.
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
from httk.atomistic import (
    FormulatemplateView,
    ChemicalFormulaView,
    CompositionView,
)

assert CompositionView({"Al": 2, "O": 3}).chemical_formula_reduced == "Al2O3"
assert ChemicalFormulaView("Al2O3") == "Al2O3"
assert FormulatemplateView("A3B2") == "A3B2"
assert FormulatemplateView({"Al": 2, "O": 3}) == "A3B2"
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

See {doc}`prototypes` and the full guide, {doc}`details/structural_classes`,
for where `ChemicalFormula` and `Formulatemplate` sit within the full
material-information taxonomy.
