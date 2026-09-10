"""Bounded, identity-based preparation caches for approximate comparisons."""

from collections import OrderedDict
from collections.abc import Callable
from typing import cast

__all__ = ["StructureComparisonCache"]


class StructureComparisonCache:
    """Cache reusable approximate-comparison preparation for one caller scope.

    Entries are keyed by object identity and retain strong references to their source
    objects.  This avoids relying on the unhashable exact structure models and prevents
    object-id reuse while an entry is live.  The cache stores derived structures,
    normalizer-image tuples and prepared NumPy orbit arrays. Cell-pair metrics remain
    per-comparison data. Normalizer images have a separate capacity so preparing them
    does not evict cached canonical structures.

    :param max_structures: Maximum number of structure/template conversion and
        canonicalization entries retained together, and independently the maximum
        number of normalizer-image tuples retained.
    :param max_geometries: Maximum number of NumPy Cartesian-orbit entries retained.
    :param max_geometry_bytes: Maximum combined array payload in the geometry cache,
        excluding Python objects and exact source structures. An oversized geometry
        is used without caching or evicting other entries.
    :raises ValueError: If any limit is not a positive integer.
    """

    def __init__(
        self, *, max_structures: int = 256, max_geometries: int = 1024, max_geometry_bytes: int = 16 * 1024 * 1024
    ) -> None:
        _validate_limit(max_structures, "max_structures")
        _validate_limit(max_geometries, "max_geometries")
        _validate_limit(max_geometry_bytes, "max_geometry_bytes")
        self._max_structures = max_structures
        self._max_geometries = max_geometries
        self._max_geometry_bytes = max_geometry_bytes
        self._geometry_bytes = 0
        self._structures: OrderedDict[tuple[str, int, float | None], tuple[object, float | None, object]] = (
            OrderedDict()
        )
        self._normalizers: OrderedDict[int, tuple[object, object]] = OrderedDict()
        self._geometries: OrderedDict[int, tuple[object, object, int]] = OrderedDict()

    def clear(self) -> None:
        """Discard all cached derived structures and geometry arrays."""
        self._structures.clear()
        self._normalizers.clear()
        self._geometries.clear()
        self._geometry_bytes = 0

    def _structure[Result](
        self,
        source: object,
        factory: Callable[[], Result],
        *,
        tolerance: float | None = None,
        kind: str = "structure",
    ) -> Result:
        """Return a cached structure conversion or canonicalization result.

        ``source`` is matched with ``is`` and ``tolerance`` is part of the key, so
        canonical results produced under different tolerances cannot be confused.
        ``factory`` is called only on a cache miss; a failed factory call is never
        inserted.

        :param source: The exact source object whose derived result is being prepared.
        :param factory: Zero-argument function producing the derived result.
        :param tolerance: Tolerance used by the preparation, if any.
        :param kind: Independent preparation namespace, such as ``"canonical"`` or
            ``"anonymous"``.
        :return: The cached or newly prepared result.
        """
        key = (kind, id(source), tolerance)
        cached = self._structures.get(key)
        if cached is not None and cached[0] is source and cached[1] == tolerance:
            self._structures.move_to_end(key)
            return cast(Result, cached[2])
        result = factory()
        self._structures[key] = (source, tolerance, result)
        self._structures.move_to_end(key)
        while len(self._structures) > self._max_structures:
            self._structures.popitem(last=False)
        return result

    def _normalizer_images[Result](self, source: object, factory: Callable[[], Result]) -> Result:
        """Reuse exact normalizer images of one already represented, immutable ASU."""
        key = id(source)
        cached = self._normalizers.get(key)
        if cached is not None and cached[0] is source:
            self._normalizers.move_to_end(key)
            return cast(Result, cached[1])
        result = factory()
        self._normalizers[key] = (source, result)
        self._normalizers.move_to_end(key)
        while len(self._normalizers) > self._max_structures:
            self._normalizers.popitem(last=False)
        return result

    def _geometry[Result](
        self, source: object, factory: Callable[[], Result], *, nbytes: Callable[[Result], int] | None = None
    ) -> Result:
        """Return cached NumPy Cartesian orbit geometry for one exact structure.

        The factory is called only on a cache miss.  Failed preparation is never
        cached.  Geometry entries use object identity and retain their exact source
        strongly until evicted or cleared.

        :param source: The exact structure from which the geometry is rendered.
        :param factory: Zero-argument function producing the prepared orbit arrays.
        :param nbytes: Array-payload size of a newly prepared result. Non-array
            preparations can omit it and remain bounded by the entry count.
        :return: The cached or newly prepared orbit arrays.
        """
        key = id(source)
        cached = self._geometries.get(key)
        if cached is not None and cached[0] is source:
            self._geometries.move_to_end(key)
            return cast(Result, cached[1])
        result = factory()
        size = 0 if nbytes is None else nbytes(result)
        if size > self._max_geometry_bytes:
            return result
        self._geometries[key] = (source, result, size)
        self._geometry_bytes += size
        self._geometries.move_to_end(key)
        while len(self._geometries) > self._max_geometries or self._geometry_bytes > self._max_geometry_bytes:
            _, (_, _, removed_size) = self._geometries.popitem(last=False)
            self._geometry_bytes -= removed_size
        return result


def _validate_limit(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
