"""Smoke tests: the package and its handler registration package import cleanly."""


def test_import_io_package():
    import httk.atomistic.io  # noqa: F401
    from httk.atomistic.io import cif

    assert hasattr(cif, "read_cif")


def test_import_cif_subpackage():
    from httk.atomistic.io import cif

    for name in cif.__all__:
        assert hasattr(cif, name), name


def test_import_handlers_package():
    # Importing the handler package must register the cif loader as a side effect.
    import httk.registry.io.atomistic  # noqa: F401
