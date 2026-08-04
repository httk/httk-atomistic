"""Lazy structure loading from files, URLs, and open text streams."""

import io
import os
import urllib.request
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

import httk.core
from httk.core.datastream import BytestreamBackend, BytestreamURLView, BytestreamView, TextstreamBackend, TextstreamView
from httk.core.datastream.network_policy import require_network_consent
from httk.core.optimade import is_optimade_entry_url, redact_optimade_url

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.backend import StructureBackend


class DatastreamStructure(StructureBackend):
    """A structure source parsed only when its data is first accessed.

    Requests are intentionally loader-only: an OPTIMADE-shaped Request is declined so
    its headers are never lost by replacing it with ``fetch(url)``. Open streams are
    one-shot sources; a failed parse is not cached, but consumed data cannot be replayed.
    """

    kind: ClassVar[str] = "datastream"

    @staticmethod
    def _is_url_string(value: str) -> bool:
        return urlsplit(value).scheme in {"http", "https", "ftp", "file"} and "://" in value

    @staticmethod
    def _is_optimade_url(value: str) -> bool:
        return urlsplit(value).scheme in {"http", "https", "ftp", "file"} and is_optimade_entry_url(value)

    @staticmethod
    def _is_network_optimade_url(value: str) -> bool:
        return urlsplit(value).scheme in {"http", "https", "ftp"} and is_optimade_entry_url(value)

    @staticmethod
    def _is_stream_source(obj: Any) -> bool:
        return isinstance(
            obj,
            (
                TextstreamBackend,
                TextstreamView,
                BytestreamBackend,
                BytestreamView,
                io.IOBase,
            ),
        )

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints.get("kind", cls.kind) != cls.kind:
            return None
        name = cls._source_name(obj, hints)
        if name is None:
            return None
        url = cls._url(obj)
        optimade = url is not None and cls._is_optimade_url(url)
        if isinstance(obj, urllib.request.Request) and optimade:
            return super().__new__(cls) if httk.core.has_reader_for(name) else None
        if httk.core.has_reader_for(name):
            return super().__new__(cls)
        if (
            isinstance(obj, str) and not cls._is_stream_source(obj) or isinstance(obj, httk.core.DatastreamURL)
        ) and optimade:
            return super().__new__(cls)
        return None

    def __init__(self, obj: Any, **hints: Any) -> None:
        self._source = obj
        self._hints = hints
        self._name = self._source_name(obj, hints)
        self._parsed: Any = None

        request_url = self._url(obj)
        if (
            isinstance(obj, urllib.request.Request)
            and request_url is not None
            and self._is_network_optimade_url(request_url)
            and self._name is not None
            and httk.core.has_reader_for(self._name)
            and "name" not in self._hints
        ):
            raise ValueError(
                "Request URL is both OPTIMADE-shaped and loader-named; use httk.core.fetch(url, kind=...) "
                "or pass a name= hint to select the loader."
            )

        if isinstance(obj, os.PathLike):
            source = os.fsdecode(os.fspath(obj))
            if not Path(source).exists():
                raise FileNotFoundError(f"Datastream structure source does not exist: {source!r}")
        elif isinstance(obj, str) and not self._is_stream_source(obj):
            source = obj
            if self._is_url_string(source) and urlsplit(source).scheme in {"http", "https", "ftp"}:
                try:
                    require_network_consent(source)
                except PermissionError as error:
                    raise PermissionError(
                        f"{error} For a lazy structure source, wrap the URL: "
                        f"DatastreamURL({redact_optimade_url(source)!r})."
                    ) from None
            elif not self._is_url_string(source) and not Path(source).exists():
                raise FileNotFoundError(f"Datastream structure source does not exist: {source!r}")

    @staticmethod
    def _url(obj: Any) -> str | None:
        if isinstance(obj, str):
            return obj if DatastreamStructure._is_url_string(obj) else None
        if isinstance(obj, httk.core.DatastreamURL):
            return obj.url
        if isinstance(obj, urllib.request.Request):
            return obj.full_url
        if DatastreamStructure._is_stream_source(obj):
            return getattr(obj, "url", None)
        return getattr(obj, "url", None)

    @classmethod
    def _source_name(cls, obj: Any, hints: dict[str, Any]) -> str | None:
        if "name" in hints:
            name = hints["name"]
            return None if name is None else os.fsdecode(os.fspath(name))
        if cls._is_stream_source(obj):
            url = getattr(obj, "url", None)
            if url is None and isinstance(obj, (httk.core.TextstreamURLView, BytestreamURLView)):
                url = str(obj)
            if url is not None:
                return urlsplit(url).path
            name = getattr(obj, "name", None)
            return name if isinstance(name, str) else None
        if isinstance(obj, (str, os.PathLike)):
            value = os.fsdecode(os.fspath(obj))
            return urlsplit(value).path if isinstance(obj, str) and cls._is_url_string(value) else value
        if isinstance(obj, httk.core.DatastreamURL):
            return urlsplit(obj.url).path
        if isinstance(obj, urllib.request.Request):
            return urlsplit(obj.full_url).path
        return None

    def _native(self) -> Any:
        if self._parsed is not None:
            return self._parsed

        source = self._source
        if isinstance(source, httk.core.DatastreamURL):
            parsed = httk.core.fetch(source.url, timeout=source.timeout)
        elif self._is_stream_source(source):
            assert self._name is not None
            stream = (
                source if isinstance(source, httk.core.TextstreamFileView) else httk.core.TextstreamFileView(source)
            )
            parsed = httk.core.load_source(stream, self._name)
        elif isinstance(source, (str, os.PathLike)):
            source_text = os.fsdecode(os.fspath(source))
            parsed = (
                httk.core.load(source_text)
                if not isinstance(source, str) or not self._is_url_string(source_text)
                else httk.core.fetch(source_text)
            )
        else:
            assert self._name is not None
            if isinstance(source, urllib.request.Request):
                stream = httk.core.TextstreamFileView(source)
            elif isinstance(source, httk.core.TextstreamFileView):
                stream = source
            else:
                stream = httk.core.TextstreamFileView(source)
            parsed = httk.core.load_source(stream, self._name)
        self._parsed = parsed
        return parsed

    def resolve(self) -> StructureBackend:
        """Return the memoized native structure, resolving this source on demand."""
        return self._native()

    @property
    def cell(self) -> Cell:
        return self.resolve().cell

    @property
    def sites(self) -> Sites:
        return self.resolve().sites

    @property
    def species(self) -> tuple[Species, ...]:
        return self.resolve().species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        return self.resolve().species_at_sites

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.resolve(), name)

    def unwrap(self) -> Any:
        return self._source
