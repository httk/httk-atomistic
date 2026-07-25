"""Disorder, vacancies, and attached species: what a site can actually hold

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

**Which representations can carry this?** This is the practical point, and the
answer is sharp. The Simple representation — a `Structure`, and the
`StructureSimpleView` over any backend — carries all of it, because it stores
the `Species` objects themselves. The OPTIMADE species dict (via
`SpeciesPrimitiveView`) carries all of it too, because that is where the model
came from. But the **primitive** representation is the spglib-style
`(lattice, positions, numbers)` triple, and `numbers` is a list of bare atomic
numbers: there is no integer that means "90% Ti", and no integer that means
"carbon plus three hydrogens". So `StructurePrimitiveView` does not silently
round, drop, or approximate — it raises `TypeError`, naming the species it
could not express.

`Species.is_single_element` is the predicate behind that check, and it is worth
knowing about directly: it is true only for a species made of exactly one *real*
element symbol (so not `"X"` and not `"vacancy"`) with nothing attached. Test it
before you reach for a symmetry library, a force field, or anything else that
speaks in atomic numbers.

The rule of thumb: **keep disordered structures in the Simple representation,
and convert to primitive only at the boundary of a tool that genuinely cannot
represent disorder** — where the `TypeError` is exactly the error you want.
"""

from httk.core import unwrap

from httk.atomistic import (
    Species,
    SpeciesPrimitiveView,
    Structure,
    StructurePrimitiveView,
    StructureSimpleView,
)

CUBIC = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

#: A 50/50 substitutional Fe/Ni alloy site.
FE_NI = Species(name="M", chemical_symbols=("Fe", "Ni"), concentration=(0.5, 0.5))
#: A titanium site that is empty one time in ten.
TI_WITH_VACANCY = Species(name="Ti", chemical_symbols=("Ti", "vacancy"), concentration=(0.9, 0.1))
#: A carbon carrying three hydrogens whose positions are not modelled.
METHYL = Species(name="CH3", chemical_symbols=("C",), concentration=(1.0,), attached=("H",), nattached=(3,))
#: An ordinary, fully ordered species -- the only kind a primitive triple can hold.
OXYGEN = Species(name="O", chemical_symbols=("O",), concentration=(1.0,))


def explain(species: Species) -> None:
    """Print what a species says, and whether it survives the trip to an atomic number."""
    occupancy = ", ".join(
        f"{100 * fraction:g}% {symbol}" for symbol, fraction in zip(species.chemical_symbols, species.concentration)
    )
    attachments = ""
    if species.attached is not None and species.nattached is not None:
        attachments = " + attached " + ", ".join(
            f"{count}x{symbol}" for symbol, count in zip(species.attached, species.nattached)
        )
    print(f"  {species.name:<4} {occupancy}{attachments}")
    print(f"       is_single_element = {species.is_single_element}")


def species_are_richer_than_elements() -> None:
    print("What each species says:")
    for species in (FE_NI, TI_WITH_VACANCY, METHYL, OXYGEN):
        explain(species)


def disorder_survives_the_simple_representation() -> None:
    """The Structure, and any StructureSimpleView, keep every field intact."""
    structure = Structure(
        cell=CUBIC,
        sites=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        species=[TI_WITH_VACANCY, METHYL],
        species_at_sites=["Ti", "CH3"],
    )

    by_name = {species.name: species for species in structure.species}
    print("Simple representation (a Structure):")
    print("  Ti  chemical_symbols ", by_name["Ti"].chemical_symbols)
    print("  Ti  concentration    ", by_name["Ti"].concentration)
    print("  CH3 attached         ", by_name["CH3"].attached)
    print("  CH3 nattached        ", by_name["CH3"].nattached)

    # A view over the same structure is the same data seen again, not a lossy copy:
    # unwrap() hands back the original object.
    view = StructureSimpleView(structure)
    print("  round-trips through StructureSimpleView:", unwrap(view) is structure)

    # ...and so does the OPTIMADE species dict, which is where this model comes from.
    print("  as OPTIMADE dicts:")
    for species in structure.species:
        print("   ", dict(SpeciesPrimitiveView(species)))


def the_primitive_representation_refuses() -> None:
    """A bare atomic number cannot mean '90% Ti', so the conversion fails loudly."""
    print("Primitive representation (an spglib-style triple):")
    for label, species in (("partial vacancy", TI_WITH_VACANCY), ("attached particles", METHYL), ("alloy", FE_NI)):
        structure = Structure(
            cell=CUBIC,
            sites=[[0.0, 0.0, 0.0]],
            species=[species],
            species_at_sites=[species.name],
        )
        try:
            StructurePrimitiveView(structure)
        except TypeError as error:
            print(f"  {label:<19} -> TypeError: {error}")
        else:  # pragma: no cover - unreachable while the check above holds
            print(f"  {label:<19} -> unexpectedly accepted")

    # A fully ordered structure converts without complaint, of course:
    ordered = Structure(cell=CUBIC, sites=[[0.0, 0.0, 0.0]], species=[OXYGEN], species_at_sites=["O"])
    _lattice, _positions, numbers = StructurePrimitiveView(ordered)
    print("  fully ordered       -> atomic numbers", numbers)


def guard_before_converting() -> None:
    """The idiomatic pre-flight check before handing a structure to an element-only tool."""
    mixed = Structure(
        cell=CUBIC,
        sites=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        species=[FE_NI, OXYGEN],
        species_at_sites=["M", "O"],
    )
    offenders = [species.name for species in mixed.species if not species.is_single_element]
    print("Pre-flight check:")
    print("  species that cannot become an atomic number:", offenders)
    print("  safe to convert to a primitive triple:", not offenders)


def main() -> None:
    species_are_richer_than_elements()
    print()
    disorder_survives_the_simple_representation()
    print()
    the_primitive_representation_refuses()
    print()
    guard_before_converting()


if __name__ == "__main__":
    main()
