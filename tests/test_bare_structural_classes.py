"""Distinct bare families retain source identity through explicit view projection."""

import pickle

import pytest

from httk.atomistic import (
    BareProtostructure,
    BareProtostructureLabel,
    BareProtostructureView,
    BarePrototype,
    BarePrototypeLabel,
    BarePrototypeView,
    Protostructure,
    ProtostructureView,
    Prototype,
    PrototypeView,
    Species,
)


@pytest.mark.parametrize(
    'bare_cls,refined_cls,bare_view,refined_view,label_cls,occupations',
    [
        (BarePrototype, Prototype, BarePrototypeView, PrototypeView, BarePrototypeLabel, [('a', 'A'), ('b', 'B')]),
        (
            BareProtostructure,
            Protostructure,
            BareProtostructureView,
            ProtostructureView,
            BareProtostructureLabel,
            [('a', Species('Na', ('Na',), (1,))), ('b', Species('Cl', ('Cl',), (1,)))],
        ),
    ],
)
def test_bare_projection_and_refinement_contract(
    bare_cls, refined_cls, bare_view, refined_view, label_cls, occupations
):
    """Projection preserves a source; materialization deliberately sheds refinement."""
    bare = bare_cls(225, occupations)
    with pytest.raises(ValueError, match='requires a representative or discriminator'):
        refined_cls(225, occupations)
    for kwargs in ({'representative': None}, {'discriminator': '001'}):
        with pytest.raises(TypeError):
            bare_cls(225, occupations, **kwargs)
    refined = refined_cls(225, occupations, discriminator='001')
    other = refined_cls(225, occupations, discriminator='002')
    assert refined != other and refined != bare
    projected = bare_view(refined)
    assert projected.unwrap() is refined
    assert refined_view(projected).unview() is refined
    assert projected.unview() == bare_view(other).unview() == bare
    assert not hasattr(projected, 'representative')
    assert not hasattr(projected, 'discriminator')
    assert not hasattr(projected, 'similar')
    assert refined_view(label_cls(projected)).unview() is refined
    restored = pickle.loads(pickle.dumps(projected))
    assert restored.unview() == bare
    assert refined_view(restored).unview() == refined
    with pytest.raises(TypeError):
        refined_view(projected.unview())
    with pytest.raises(TypeError):
        refined_view(str(projected.label))
    assert bare_view(str(projected.label)).unview() == bare


def test_bare_assigned_erasure_and_explicit_refinement():
    """Anonymous bare projection preserves the assigned source and canonical labels."""
    assigned = BareProtostructure(225, [('a', 'Na'), ('b', 'Cl')])
    anonymous = BarePrototypeView(assigned)
    assert anonymous.unwrap() is assigned
    assert anonymous.unview() == BarePrototype(225, [('a', 'A'), ('b', 'B')])
    refined = Prototype(prototype=anonymous.unview(), discriminator='001')
    assert BarePrototypeView(refined).unview() == anonymous.unview()
    assert assigned.formula == 'ClNa'
    assert anonymous.anonymous_formula == 'AB'


def test_assigned_source_recovery_through_anonymous_views():
    """Anonymous views retain their original assigned backend at both levels."""
    bare = BareProtostructure(225, [('a', 'Na'), ('b', 'Cl')])
    refined = Protostructure(225, bare.occupations, discriminator='001')
    assert BareProtostructureView(BarePrototypeView(bare)).unview() is bare
    assert ProtostructureView(BarePrototypeView(refined)).unview() is refined
    assert ProtostructureView(PrototypeView(refined)).unview() is refined
    assert BareProtostructureView(BarePrototypeView(refined)).unview() == bare


def test_structural_input_aliases_resolve_at_runtime():
    """Lazy cross-family unions expose canonical runtime types without import cycles."""
    from typing import get_args

    from httk.atomistic import BareProtostructureLike, BarePrototypeLike, ProtostructureLike, PrototypeLike

    for alias, native in (
        (BarePrototypeLike, BarePrototype),
        (BareProtostructureLike, BareProtostructure),
        (PrototypeLike, Prototype),
        (ProtostructureLike, Protostructure),
    ):
        assert native in get_args(alias.__value__)
