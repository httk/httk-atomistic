"""The cif loader is discoverable through httk-core's registry."""

import importlib.util

import httk.core
import pytest


def test_cif_extension_registered():
    # Discovery runs on `import httk.core` and imports httk.registry.io.atomistic,
    # which registers the ".cif" loader.
    assert ".cif" in httk.core.register.known_extensions()


def test_cif_loader_points_at_the_atomistic_reader():
    """``load`` yields interpreted asymmetric units, not the raw token tree.

    The low-level ``read_cif`` tokenizer is still exported for callers who want the tags
    verbatim; it is just not what the registry dispatches to, so that a ``.cif`` behaves
    like a ``POSCAR`` and can be handed straight to the structure builders.
    """
    spec = httk.core.register.readers.require(".cif")
    assert spec.name == "cif"
    assert spec.handler == "httk.atomistic.cif_structures:_read_cif_for_atomistic"


def test_cif_loader_resolves_to_callable():
    from httk.core._plugins import resolve_callable

    spec = httk.core.register.readers.require(".cif")
    fn = resolve_callable(spec.handler)
    from httk.atomistic.cif_structures import _read_cif_for_atomistic

    assert fn is _read_cif_for_atomistic


def test_mcif_extension_registered_with_neutral_loader():
    spec = httk.core.register.readers.require(".mcif")
    assert spec.name == "mcif"
    assert spec.handler == "httk.atomistic.io.cif:read_mcif_asus"


def test_legacy_httk_io_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from httk.registry.io.atomistic import _reject_legacy_httk_io

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "httk.registry.io.io" else None)
    with pytest.raises(RuntimeError, match="pip uninstall httk-io"):
        _reject_legacy_httk_io()
