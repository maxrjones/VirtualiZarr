"""
Experimental: per-chunk codec selection with zarr-python's ``switch`` codec.

A sketch of d-v-b's proposal in zarr-developers/zarr-specs#399. Every chunk becomes
``header || source bytes``: the header is the chunk's case key and its parameters, and
the ``switch`` codec (``zarr.codecs.switch``) reads it to pick the chunk's codec chain
and fill in that chain's open values. VirtualiZarr only keeps the header's fields and
serves the bytes; all decoding happens in zarr-python's codecs.

In memory the header stays typed: [ChunkManifest][virtualizarr.manifests.ChunkManifest]
holds a ``uint64`` case-ID array and a parameters column (an object array of ``bytes``),
next to the paths, offsets and lengths. They are packed into header bytes only at the
store boundary, by [ManifestStore][virtualizarr.manifests.ManifestStore].

A case ID is the case key (the first 8 bytes of the SHA-256 of the case's canonical
JSON) read as a big-endian ``uint64``, so its hex is the key in ``zarr.json``. Keys name
cases by content, so concatenation merges case tables by union and never rewrites a
chunk's header. ID 0 marks a chunk that is not stored.

Two entry points:

- ``np.concatenate`` (and so ``xr.concat``) of [ManifestArrays][virtualizarr.manifests.ManifestArray]
  whose codecs differ, or that already use ``switch``, calls [concatenate][virtualizarr.manifests.switch.concatenate]:
  the result has one ``switch`` codec whose cases are the union of the inputs'. Where
  chunk shapes differ, or a partial chunk would land inside the array, the result uses
  zarr's experimental rectilinear chunk grid and edge chunks stored whole get ``crop``.
- [fold_cf_encoding][virtualizarr.manifests.switch.fold_cf_encoding], called on each
  virtual dataset before combining, turns per-file CF ``scale_factor`` / ``add_offset``
  and a change of dtype into codecs, with the numbers as per-chunk parameters.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable, cast

import numpy as np
import zarr
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.dtype import parse_data_type

from virtualizarr.manifests.manifest import MISSING_CHUNK_PATH, ChunkManifest
from virtualizarr.manifests.utils import RegularChunkGridMetadata

if TYPE_CHECKING:
    import xarray as xr

    from virtualizarr.manifests.array import ManifestArray

MISSING_CASE_ID = 0
_ID_BYTES = 8


def _switch_codec_class() -> type:
    try:
        from zarr.codecs.switch import SwitchCodec
    except ImportError:
        raise NotImplementedError(
            "Combining arrays whose codecs differ needs the experimental `switch` codec, "
            "which this zarr-python does not provide. See "
            "https://github.com/zarr-developers/zarr-specs/issues/399"
        ) from None
    return SwitchCodec


def case_id(case: dict[str, Any]) -> int:
    """The case's ID: its ``switch`` key read as a big-endian ``uint64``."""
    _switch_codec_class()
    from zarr.codecs.switch import case_key

    return int.from_bytes(case_key(case), "big")


def header(case_id: int, parameters: bytes) -> bytes:
    """A chunk's ``switch`` header: its case key, then its packed parameters."""
    return case_id.to_bytes(_ID_BYTES, "big") + parameters


def is_switch(metadata: ArrayV3Metadata) -> bool:
    return (
        len(metadata.codecs) == 1 and type(metadata.codecs[0]).__name__ == "SwitchCodec"
    )


def cases(marr: "ManifestArray") -> dict[str, dict[str, Any]]:
    """The array's case table; an array without ``switch`` is one case, its codecs."""
    if is_switch(marr.metadata):
        doc = cast(dict[str, Any], marr.metadata.codecs[0].to_dict())
        return doc["configuration"]["cases"]
    case = {"codecs": [c.to_dict() for c in marr.metadata.codecs], "parameters": []}
    return {f"{case_id(case):016x}": case}


def _shift_parameters(node: Any, by: int) -> Any:
    """``node`` with every ``{"parameter": k}`` renumbered to ``k + by``."""
    if isinstance(node, dict):
        if set(node) == {"parameter"}:
            return {"parameter": node["parameter"] + by}
        return {k: _shift_parameters(v, by) for k, v in node.items()}
    if isinstance(node, list):
        return [_shift_parameters(v, by) for v in node]
    return node


def _pack(dtypes: Iterable[str], values: Iterable[float | int]) -> bytes:
    return b"".join(
        np.array([v], dtype=np.dtype(d).newbyteorder("<")).tobytes()
        for d, v in zip(dtypes, values, strict=True)
    )


def with_cases(
    marr: "ManifestArray",
    *,
    prefix: list[dict[str, Any]] | None = None,
    parameters: list[str] | None = None,
    values: list[float | int] | None = None,
    data_type: np.dtype | None = None,
    fill_value: Any = None,
) -> "ManifestArray":
    """
    ``marr`` as a ``switch`` array, with ``prefix`` in front of every case's chain.

    ``prefix`` is a list of array -> array codecs (as JSON) whose open values
    ``{"parameter": k}`` count from 0 over ``parameters``; every chunk gets ``values``
    appended to its parameters. ``data_type`` and ``fill_value`` replace the array's,
    for a prefix that changes the decoded dtype. Every case gets a new key, and every
    chunk's case ID is remapped; no chunk is read.
    """
    from virtualizarr.manifests.array import ManifestArray

    prefix, parameters, values = prefix or [], parameters or [], values or []
    manifest = marr.manifest
    old_cases = cases(marr)
    if manifest.has_case_ids:
        assert manifest._case_ids is not None and manifest._parameters is not None
        old_ids, old_params = manifest._case_ids, manifest._parameters
    else:
        (only,) = old_cases
        stored = manifest._paths != MISSING_CHUNK_PATH
        old_ids = np.where(stored, np.uint64(int(only, 16)), np.uint64(MISSING_CASE_ID))
        old_params = np.full(manifest.shape_chunk_grid, b"", dtype=object)

    new_cases: dict[str, dict[str, Any]] = {}
    new_ids = np.full_like(old_ids, MISSING_CASE_ID)
    for key, case in old_cases.items():
        n = len(case["parameters"])
        new = {
            "codecs": [_shift_parameters(c, n) for c in prefix] + list(case["codecs"]),
            "parameters": list(case["parameters"]) + list(parameters),
        }
        new_id = case_id(new)
        new_cases[f"{new_id:016x}"] = new
        new_ids[old_ids == np.uint64(int(key, 16))] = new_id
    packed = _pack(parameters, values)
    # appended to every chunk's parameters (missing chunks too, never served)
    new_params = (
        np.frompyfunc(lambda p: p + packed, 1, 1)(old_params).astype(object)
        if packed
        else old_params
    )

    doc: dict[str, Any] = dict(marr.metadata.to_dict())
    doc["codecs"] = [
        _switch_codec_class()(cases=new_cases).to_dict()  # validates every key
    ]
    if data_type is not None:
        zdtype = parse_data_type(data_type, zarr_format=3)
        doc["data_type"] = zdtype.to_json(zarr_format=3)
        doc["fill_value"] = zdtype.to_json_scalar(
            marr.metadata.fill_value if fill_value is None else fill_value,
            zarr_format=3,
        )
    return ManifestArray(
        metadata=ArrayV3Metadata.from_dict(doc),
        chunkmanifest=ChunkManifest.from_arrays(
            paths=manifest._paths,
            offsets=manifest._offsets,
            lengths=manifest._lengths,
            validate_paths=False,
            inlined=dict(manifest._inlined),
            case_ids=new_ids,
            parameters=new_params,
        ),
    )


def _axis_edges(metadata: ArrayV3Metadata, axis: int) -> int | tuple[int, ...]:
    """The chunk grid along ``axis``: a regular step (an int), or every chunk's length."""
    grid = metadata.chunk_grid
    if isinstance(grid, RegularChunkGridMetadata):
        return grid.chunk_shape[axis]
    return grid.chunk_shapes[axis]  # type: ignore[union-attr]


def _exact_edges(metadata: ArrayV3Metadata, axis: int) -> tuple[int, ...]:
    """Each chunk's length along ``axis``, the last one cut at the array's edge."""
    n, edges = metadata.shape[axis], _axis_edges(metadata, axis)
    if isinstance(edges, int):
        return (edges,) * (n // edges) + ((n % edges,) if n % edges else ())
    out, left = [], n
    for e in edges:
        if left <= 0:
            break
        out.append(min(e, left))
        left -= e
    return tuple(out)


def _rectilinear_enabled() -> bool:
    return bool(zarr.config.get("array.rectilinear_chunks"))


def concatenate(
    arrays: list["ManifestArray"] | tuple["ManifestArray", ...], axis: int
) -> "ManifestArray":
    """
    Concatenate arrays whose codecs or chunk shapes differ.

    - Codecs differ, or any input already uses ``switch``: every input becomes a
      ``switch`` array and the result's case table is the union of theirs. Two inputs
      that use the same chain share its key, so it appears once.
    - Chunk shapes differ along ``axis``, or an input other than the last ends in a
      partial chunk: the result uses the rectilinear chunk grid, which zarr-python only
      reads with ``zarr.config.set({"array.rectilinear_chunks": True})``. An input that
      ends in a partial chunk gets ``crop`` (its edge chunk is stored whole) and so
      ``switch``.

    Along the other axes the inputs must be chunked alike.
    """
    from virtualizarr.manifests.array import ManifestArray
    from virtualizarr.manifests.array_api import _concat_manifests
    from virtualizarr.manifests.utils import check_same_dtypes

    check_same_dtypes([arr.dtype for arr in arrays])
    metas = [arr.metadata for arr in arrays]
    ndim = metas[0].ndim
    for d in range(ndim):
        if d != axis and len({_axis_edges(m, d) for m in metas}) > 1:
            raise ValueError(
                f"Cannot concatenate arrays chunked differently along axis {d}: "
                f"{[_axis_edges(m, d) for m in metas]}"
            )

    steps = [_axis_edges(m, axis) for m in metas]
    regular = (
        all(isinstance(m.chunk_grid, RegularChunkGridMetadata) for m in metas)
        and len(set(steps)) == 1
        and all(m.shape[axis] % s == 0 for m, s in zip(metas[:-1], steps[:-1]))  # type: ignore[operator]
    )
    if not regular and not _rectilinear_enabled():
        raise ValueError(
            "Cannot concatenate arrays with partial chunks or inconsistent chunk shapes "
            f"along axis {axis} ({steps}) without a rectilinear chunk grid. zarr-python's "
            "is experimental; enable it with "
            "zarr.config.set({'array.rectilinear_chunks': True})."
        )

    # an input that ends in a partial chunk stored whole needs `crop` in a rectilinear grid
    crop = [
        not regular
        and isinstance(m.chunk_grid, RegularChunkGridMetadata)
        and m.shape[axis] % m.chunk_grid.chunk_shape[axis] != 0
        for m in metas
    ]
    use_switch = (
        any(crop)
        or any(is_switch(m) or a.manifest.has_case_ids for m, a in zip(metas, arrays))
        or any(m.codecs != metas[0].codecs for m in metas[1:])
    )
    if use_switch:
        arrays = [
            with_cases(
                a,
                prefix=[
                    {
                        "name": "crop",
                        "configuration": {"shape": list(m.chunk_grid.chunk_shape)},  # type: ignore[union-attr]
                    }
                ]
                if c
                else None,
            )
            for a, m, c in zip(arrays, metas, crop)
        ]

    shape = list(metas[0].shape)
    shape[axis] = sum(m.shape[axis] for m in metas)
    doc: dict[str, Any] = dict(arrays[0].metadata.to_dict())
    doc["shape"] = shape
    if use_switch:
        merged: dict[str, dict[str, Any]] = {}
        for a in arrays:
            for key, case in cases(a).items():
                if merged.setdefault(key, case) != case:
                    raise ValueError(f"switch case {key} names two different chains")
        doc["codecs"] = [_switch_codec_class()(cases=merged).to_dict()]
    if not regular:
        chunk_shapes: list[Any] = [_axis_edges(metas[0], d) for d in range(ndim)]
        chunk_shapes[axis] = [e for m in metas for e in _exact_edges(m, axis)]
        doc["chunk_grid"] = {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": chunk_shapes},
        }
    return ManifestArray(
        metadata=ArrayV3Metadata.from_dict(doc),
        chunkmanifest=_concat_manifests([a.manifest for a in arrays], axis=axis),
    )


def fold_cf_encoding(vds: "xr.Dataset", dtype: Any = np.float32) -> "xr.Dataset":
    """
    Move each virtual variable's CF packing into its codecs, before combining.

    For every virtual variable with ``scale_factor`` / ``add_offset``, or whose dtype is
    not ``dtype``, the result decodes to ``dtype`` in zarr-python's codecs:

    - ``scale_factor`` / ``add_offset`` become a ``scale_offset`` codec whose ``offset``
      and ``scale`` are per-chunk parameters (so files packed differently share a case);
    - a different stored dtype becomes ``cast_value``; from an integer type to a float
      type, the CF ``_FillValue`` maps to NaN, the array's new fill value.

    Those attributes are removed from the variable. Other variables are unchanged.
    Concatenating the results then merges their cases (see ``concatenate``).

    shortcut: ``dtype`` is one target for every variable, not CF's per-variable choice
    from the attributes' types; ``missing_value`` and ``valid_range`` are not handled.
    """
    import xarray as xr

    from virtualizarr.manifests.array import ManifestArray

    target = np.dtype(dtype)
    out = vds.copy()
    for name, var in vds.variables.items():
        marr = var.data
        if not isinstance(marr, ManifestArray):
            continue
        attrs = dict(var.attrs)
        scale, offset = attrs.pop("scale_factor", None), attrs.pop("add_offset", None)
        stored = marr.dtype
        if scale is None and offset is None and stored == target:
            continue
        prefix: list[dict[str, Any]] = []
        params: list[str] = []
        values: list[float | int] = []
        if scale is not None or offset is not None:
            # CF: physical = packed * scale_factor + add_offset; scale_offset decodes
            # x / scale + offset
            prefix.append(
                {
                    "name": "scale_offset",
                    "configuration": {
                        "offset": {"parameter": 0},
                        "scale": {"parameter": 1},
                    },
                }
            )
            params += ["float64", "float64"]
            values += [
                float(offset if offset is not None else 0),
                1 / float(scale if scale is not None else 1),
            ]
        fill_value = None
        if stored != target:
            config: dict[str, Any] = {"data_type": stored.name}
            if stored.kind in "iu" and target.kind == "f":
                # the new fill value, NaN, must round-trip through the integer type
                fill = attrs.pop("_FillValue", None)
                if fill is None:
                    raise ValueError(
                        f"Variable {name!r} is {stored} without a _FillValue, so no "
                        f"integer can stand for NaN when it is cast to {target}"
                    )
                config["scalar_map"] = {
                    "encode": [["NaN", int(fill)]],
                    "decode": [[int(fill), "NaN"]],
                }
                fill_value = math.nan
            prefix.append({"name": "cast_value", "configuration": config})
        folded = with_cases(
            marr,
            prefix=prefix,
            parameters=params,
            values=values,
            data_type=target,
            fill_value=fill_value,
        )
        out[name] = xr.Variable(var.dims, folded, attrs, var.encoding)
    return out


def refuse_case_ids(virtual_datasets: Iterable["xr.Dataset"], writer: str) -> None:
    """Raise before writing anything if a virtual variable's chunks carry case IDs."""
    from virtualizarr.manifests.array import ManifestArray

    for vds in virtual_datasets:
        for name, var in vds.variables.items():
            if isinstance(var.data, ManifestArray) and var.data.manifest.has_case_ids:
                raise NotImplementedError(
                    f"Cannot write variable {name!r} to {writer}: its chunks carry "
                    "switch case IDs (an experimental per-chunk codec header, see "
                    "zarr-developers/zarr-specs#399) that the "
                    f"{writer} format cannot store yet. Read it through ManifestStore."
                )
