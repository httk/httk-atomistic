"""A tiny example: build a Structure and present it as a primitive triple."""

from httk.atomistic import Structure, StructurePrimitiveView


def main() -> None:
    structure = Structure(
        cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        sites=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        species=[
            {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
            {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1.0]},
        ],
        species_at_sites=["Na", "Cl"],
    )
    lattice, positions, numbers = StructurePrimitiveView(structure)
    print(lattice, positions, numbers)


if __name__ == "__main__":
    main()
