"""
The many-file cases of zarr-developers/zarr-specs#399, combined with the experimental
`switch` codec (see `virtualizarr.manifests.switch`).

Ported from the demo in d-v-b/refkv (branch demo/zarr-specs-399,
design/experiments/inhomogeneous/demo.py). Each case writes a few small netCDF4 files
holding one smooth field `v(time, x)` that continues across them, virtualizes each
file, combines them with `xr.concat`, and reads the result back with xarray and
zarr-python through `ManifestStore`. The reference is each file read on its own by
xarray and h5netcdf, and every case must match it bit for bit (NaN equal to NaN).
"""

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.core.metadata.v3 import ArrayV3Metadata

import virtualizarr as vz
from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.switch import case_id, cases, fold_cf_encoding, header
from virtualizarr.manifests.utils import copy_and_replace_metadata
from virtualizarr.parsers import HDFParser
from virtualizarr.tests import requires_hdf5plugin, requires_imagecodecs

pytest.importorskip("zarr.codecs.switch")
h5py = pytest.importorskip("h5py")
pytest.importorskip("h5netcdf")

NX = 6

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    requires_hdf5plugin,
    requires_imagecodecs,
]


@pytest.fixture(autouse=True)
def rectilinear_chunks():
    with zarr.config.set({"array.rectilinear_chunks": True}):
        yield


def smooth(t: np.ndarray, base: float = 285.0, amp: float = 4.0) -> np.ndarray:
    x = np.arange(NX)
    return (
        base
        + amp * np.sin(2 * np.pi * t[:, None] / 37.0)
        + 0.6 * np.cos(2 * np.pi * x[None, :] / NX)
    )


def write(
    path: Path,
    data: np.ndarray,
    *,
    t0: int,
    chunks,
    compression=None,
    shuffle=False,
    attrs=None,
) -> Path:
    """One netCDF4 file: `v(time, x)`, its `time` coordinate starting at t0."""
    with h5py.File(path, "w") as f:
        t = f.create_dataset(
            "time", data=np.arange(t0, t0 + data.shape[0], dtype="int64")
        )
        t.make_scale("time")
        t.attrs["units"] = "days since 2000-01-01"
        x = f.create_dataset("x", data=np.arange(NX, dtype="int64"))
        x.make_scale("x")
        v = f.create_dataset(
            "v", data=data, chunks=chunks, compression=compression, shuffle=shuffle
        )
        v.dims[0].attach_scale(t)
        v.dims[1].attach_scale(x)
        for k, val in (attrs or {}).items():
            v.attrs[k] = val
    return path


def era_files(
    root: Path, eras=(("gzip", False), ("gzip", False), ("gzip", True), (None, False))
) -> list[Path]:
    """1. codec eras (GOES-16): gzip, gzip, shuffle + gzip, uncompressed."""
    return [
        write(
            root / f"era_{i}.nc",
            smooth(np.arange(8 * i, 8 * i + 8)).astype("float32"),
            t0=8 * i,
            chunks=(4, NX),
            compression=comp,
            shuffle=shuf,
        )
        for i, (comp, shuf) in enumerate(eras)
    ]


def monthly_files(root: Path) -> list[Path]:
    """2. monthly files of 31, 28 and 31 days, one chunk each."""
    paths, t0 = [], 0
    for i, nt in enumerate([31, 28, 31]):
        paths.append(
            write(
                root / f"month_{i}.nc",
                smooth(np.arange(t0, t0 + nt)).astype("float32"),
                t0=t0,
                chunks=(nt, NX),
                compression="gzip",
            )
        )
        t0 += nt
    return paths


def ragged_files(root: Path) -> list[Path]:
    """3. ragged inner chunks: 10 steps per file chunked by 4 (4, 4, 2)."""
    return [
        write(
            root / f"ragged_{i}.nc",
            smooth(np.arange(10 * i, 10 * i + 10)).astype("float32"),
            t0=10 * i,
            chunks=(4, NX),
            compression="gzip",
        )
        for i in range(3)
    ]


def scaled_files(root: Path) -> list[Path]:
    """4. per-file CF scale_factor / add_offset on int16 (BOM BARPA)."""
    paths = []
    for i, (scale, offset) in enumerate([(0.01, 280.0), (0.02, 270.0), (0.005, 290.0)]):
        packed = np.round(
            (smooth(np.arange(10 * i, 10 * i + 10)) - offset) / scale
        ).astype("int16")
        paths.append(
            write(
                root / f"scaled_{i}.nc",
                packed,
                t0=10 * i,
                chunks=(5, NX),
                compression="gzip",
                attrs={
                    "scale_factor": np.float32(scale),
                    "add_offset": np.float32(offset),
                    "_FillValue": np.int16(-32767),
                },
            )
        )
    return paths


def dtype_files(root: Path) -> list[Path]:
    """5. varying dtype: int16, int16, float32."""
    paths = []
    for i, dtype in enumerate(["int16", "int16", "float32"]):
        values = smooth(np.arange(10 * i, 10 * i + 10), base=50, amp=40)
        paths.append(
            write(
                root / f"dtype_{i}.nc",
                (np.round(values) if dtype == "int16" else values).astype(dtype),
                t0=10 * i,
                chunks=(5, NX),
                compression="gzip",
                attrs={"_FillValue": np.int16(-32767)} if dtype == "int16" else None,
            )
        )
    return paths


def virtualize(paths: list[Path], root: Path) -> list[xr.Dataset]:
    registry = ObjectStoreRegistry({f"file://{root}": LocalStore(prefix=root)})
    return [
        vz.open_virtual_dataset(
            f"file://{p}",
            registry=registry,
            parser=HDFParser(),
            loadable_variables=["time", "x"],
        )
        for p in paths
    ]


def combine(vdss: list[xr.Dataset]) -> xr.Dataset:
    return xr.concat(
        vdss, dim="time", coords="minimal", compat="override", combine_attrs="override"
    )


def truth(paths: list[Path]) -> np.ndarray:
    """Every file decoded on its own by xarray and h5netcdf, then stacked."""
    out = []
    for p in paths:
        with xr.open_dataset(p, engine="h5netcdf") as ds:
            out.append(ds["v"].values)
    return np.concatenate(out)


def store_of(vds: xr.Dataset, root: Path, name: str = "v") -> ManifestStore:
    """A ManifestStore serving `name`, with its dimension names and attributes."""
    var = vds[name]
    marr = var.data
    metadata = copy_and_replace_metadata(
        marr.metadata,
        new_dimension_names=[str(d) for d in var.dims],
        new_attributes=dict(var.attrs),
    )
    return ManifestStore(
        group=ManifestGroup(
            arrays={name: ManifestArray(metadata=metadata, chunkmanifest=marr.manifest)}
        ),
        registry=ObjectStoreRegistry({f"file://{root}": LocalStore(prefix=root)}),
    )


def read_back(vds: xr.Dataset, root: Path, name: str = "v") -> np.ndarray:
    with xr.open_zarr(
        store_of(vds, root, name), consolidated=False, zarr_format=3
    ) as ds:
        return ds[name].values


def assert_bit_for_bit(got: np.ndarray, want: np.ndarray) -> None:
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want, strict=True)


def n_cases(marr: ManifestArray) -> int:
    return len(cases(marr))


@pytest.mark.parametrize(
    "make, fold, expect_cases, rectilinear",
    [
        pytest.param(era_files, False, 3, False, id="1-codec-eras"),
        pytest.param(monthly_files, False, 0, True, id="2-monthly-files"),
        pytest.param(ragged_files, False, 1, True, id="3-ragged-inner-chunks"),
        pytest.param(scaled_files, True, 1, False, id="4-per-file-cf-scale-offset"),
        pytest.param(dtype_files, True, 2, False, id="5-varying-dtype"),
    ],
)
def test_case_round_trips_bit_for_bit(tmp_path, make, fold, expect_cases, rectilinear):
    paths = make(tmp_path)
    vdss = virtualize(paths, tmp_path)
    if fold:  # CF attributes stay where the parser puts them; this is combine time
        vdss = [fold_cf_encoding(vds) for vds in vdss]
    combined = combine(vdss)
    marr = combined["v"].data
    assert isinstance(marr, ManifestArray)

    if expect_cases:
        assert marr.manifest.has_case_ids
        assert n_cases(marr) == expect_cases
    else:  # the same codecs and no edge chunks inside: no switch needed
        assert not marr.manifest.has_case_ids
        assert [c.to_dict()["name"] for c in marr.metadata.codecs] != ["switch"]
    assert (marr.metadata.chunk_grid.to_dict()["name"] == "rectilinear") is rectilinear
    assert not any(k in combined["v"].attrs for k in ("scale_factor", "add_offset"))

    assert_bit_for_bit(read_back(combined, tmp_path), truth(paths))


def test_case_6_concatenating_two_combined_arrays(tmp_path):
    """Two arrays combined separately, then concatenated: the case tables merge by
    union of their content keys, and every chunk keeps its case ID."""
    paths = era_files(
        tmp_path, eras=(("gzip", False), ("gzip", True), ("gzip", False), (None, False))
    )
    vdss = virtualize(paths, tmp_path)
    a, b = combine(vdss[:2]), combine(vdss[2:])
    both = combine([a, b])
    ma, mb, mab = (ds["v"].data for ds in (a, b, both))
    assert (n_cases(ma), n_cases(mb), n_cases(mab)) == (2, 2, 3)  # the gzip case once
    assert cases(mab) == {**cases(ma), **cases(mb)}
    assert cases(mab) == cases(combine(vdss)["v"].data)  # as combining all four at once
    np.testing.assert_array_equal(
        mab.manifest._case_ids,
        np.concatenate([ma.manifest._case_ids, mb.manifest._case_ids]),
    )
    assert_bit_for_bit(read_back(both, tmp_path), truth(paths))


def test_scaling_10k_chunks_each_with_its_own_scale_offset(tmp_path):
    """Every chunk packed with its own CF scale and offset, as GRIB packs each message:
    one case with two parameters, so zarr.json stays small however many chunks."""
    n, chunk, fill = 10_000, (4, 6), -32767

    def packing(i):  # a different pair for every chunk
        return 0.01 * (1 + (i % 97) / 97), 280.0 + i * 1e-5

    def physical(i):
        t = np.arange(chunk[0] * i, chunk[0] * (i + 1))[:, None]
        return (
            285
            + 4 * np.sin(2 * np.pi * t / 37.0)
            + 0.6 * np.cos(2 * np.pi * np.arange(chunk[1])[None, :] / chunk[1])
        )

    def packed(i):
        scale, offset = packing(i)
        return np.round((physical(i) - offset) / scale).astype("<i2")

    source = tmp_path / "source.bin"
    offsets, lengths = [], []
    with open(source, "wb") as f:
        for i in range(n):
            data = zlib.compress(packed(i).tobytes(), 4)
            offsets.append(f.tell())
            lengths.append(len(data))
            f.write(data)

    case = {
        "codecs": [
            {
                "name": "scale_offset",
                "configuration": {
                    "offset": {"parameter": 0},
                    "scale": {"parameter": 1},
                },
            },
            {
                "name": "cast_value",
                "configuration": {
                    "data_type": "int16",
                    "scalar_map": {
                        "encode": [["NaN", fill]],
                        "decode": [[fill, "NaN"]],
                    },
                },
            },
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "numcodecs.zlib", "configuration": {"level": 4}},
        ],
        "parameters": ["float64", "float64"],
    }
    cid = case_id(case)
    params = np.empty((n, 1), dtype=object)
    for i in range(n):
        scale, offset = packing(i)
        params[i, 0] = np.array([offset, 1 / scale], dtype="<f8").tobytes()
    manifest = ChunkManifest.from_arrays(
        paths=np.full((n, 1), source.as_uri(), dtype=np.dtypes.StringDType()),
        offsets=np.array(offsets, dtype="uint64").reshape(n, 1),
        lengths=np.array(lengths, dtype="uint64").reshape(n, 1),
        case_ids=np.full((n, 1), cid, dtype="uint64"),
        parameters=params,
    )
    metadata = ArrayV3Metadata.from_dict(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [chunk[0] * n, chunk[1]],
            "data_type": "float32",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": list(chunk)},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": "NaN",
            "codecs": [
                {"name": "switch", "configuration": {"cases": {f"{cid:016x}": case}}}
            ],
            "attributes": {},
        }
    )
    store = ManifestStore(
        group=ManifestGroup(
            arrays={"v": ManifestArray(metadata=metadata, chunkmanifest=manifest)}
        ),
        registry=ObjectStoreRegistry({"file://": LocalStore()}),
    )

    zarr_json = zarr.core.sync.sync(
        store.get("v/zarr.json", default_buffer_prototype())
    )
    t0 = time.perf_counter()
    arr = zarr.open_array(store, path="v", mode="r")
    t_open = time.perf_counter() - t0
    for i in np.linspace(0, n - 1, 200).astype(int):
        scale, offset = packing(i)
        want = packed(i).astype("float32") / np.float32(1 / scale) + np.float32(offset)
        np.testing.assert_array_equal(arr[chunk[0] * i : chunk[0] * (i + 1)], want)
    print(
        f"\nscaling: {n} chunks, zarr.json {len(zarr_json.to_bytes())} bytes, "
        f"manifest {manifest.nbytes:,} bytes in memory, open {t_open * 1e3:.1f} ms",
        file=sys.stderr,
    )
    assert len(zarr_json.to_bytes()) < 2_000


class TestHeaderAtTheStoreBoundary:
    """`ManifestStore.get` serves `header || source bytes`, for any byte range."""

    @pytest.fixture
    def store(self, tmp_path):
        body = bytes(range(100, 140))  # 40 bytes in the middle of a file
        (tmp_path / "f.bin").write_bytes(b"\x00" * 10 + body + b"\xff" * 10)
        case = {
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "parameters": ["float64"],
        }
        cid = case_id(case)
        params = np.empty((2,), dtype=object)
        params[:] = [np.array([1.5], "<f8").tobytes(), b""]
        manifest = ChunkManifest.from_arrays(
            paths=np.array(
                [(tmp_path / "f.bin").as_uri(), ""], dtype=np.dtypes.StringDType()
            ),
            offsets=np.array([10, 0], dtype="uint64"),
            lengths=np.array([40, 0], dtype="uint64"),
            case_ids=np.array([cid, 0], dtype="uint64"),
            parameters=params,
        )
        metadata = ArrayV3Metadata.from_dict(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [8],
                "data_type": "uint8",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [4]},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "fill_value": 0,
                "codecs": [
                    {
                        "name": "switch",
                        "configuration": {"cases": {f"{cid:016x}": case}},
                    }
                ],
                "attributes": {},
            }
        )
        store = ManifestStore(
            group=ManifestGroup(
                arrays={"v": ManifestArray(metadata=metadata, chunkmanifest=manifest)}
            ),
            registry=ObjectStoreRegistry({"file://": LocalStore()}),
        )
        return store, header(cid, np.array([1.5], "<f8").tobytes()) + body

    @pytest.mark.parametrize(
        "byte_range, sl",
        [
            (None, slice(None)),
            (RangeByteRequest(0, 8), slice(0, 8)),  # the case key
            (RangeByteRequest(3, 12), slice(3, 12)),  # inside the header
            (RangeByteRequest(10, 30), slice(10, 30)),  # across the header's end
            (RangeByteRequest(16, 30), slice(16, 30)),  # the body only
            (OffsetByteRequest(5), slice(5, None)),  # across, to the end
            (OffsetByteRequest(16), slice(16, None)),
            (SuffixByteRequest(4), slice(-4, None)),
            (SuffixByteRequest(50), slice(-50, None)),  # back into the header
        ],
    )
    def test_byte_ranges(self, store, byte_range, sl):
        store, served = store
        got = zarr.core.sync.sync(
            store.get("v/c/0", default_buffer_prototype(), byte_range)
        )
        assert got.to_bytes() == served[sl]

    def test_missing_chunk_has_no_header(self, store):
        store, _ = store
        assert (
            zarr.core.sync.sync(store.get("v/c/1", default_buffer_prototype())) is None
        )


def test_case_ids_survive_indexing_and_compare(tmp_path):
    vdss = virtualize(era_files(tmp_path), tmp_path)
    marr = combine(vdss)["v"].data
    head = marr[0:4, :]
    np.testing.assert_array_equal(head.manifest._case_ids, marr.manifest._case_ids[0:1])
    assert marr.manifest == marr.manifest
    assert not (marr.manifest == combine(vdss[::-1])["v"].data.manifest)
    assert (
        marr.manifest.nbytes
        > ChunkManifest.from_arrays(
            paths=marr.manifest._paths,
            offsets=marr.manifest._offsets,
            lengths=marr.manifest._lengths,
            validate_paths=False,
        ).nbytes
    )


def test_partial_chunks_need_the_rectilinear_flag(tmp_path):
    vdss = virtualize(ragged_files(tmp_path), tmp_path)
    with zarr.config.set({"array.rectilinear_chunks": False}):
        with pytest.raises(ValueError, match="array.rectilinear_chunks"):
            combine(vdss)


def test_differing_dtypes_still_need_folding(tmp_path):
    vdss = virtualize(dtype_files(tmp_path), tmp_path)
    with pytest.raises(ValueError, match="inconsistent dtypes"):
        combine(vdss)


def test_writers_refuse_case_ids(tmp_path):
    combined = combine(virtualize(era_files(tmp_path), tmp_path))
    with pytest.raises(
        NotImplementedError, match="to kerchunk: its chunks carry switch case IDs"
    ):
        combined.vz.to_kerchunk(format="dict")
    icechunk = pytest.importorskip("icechunk")
    storage = icechunk.local_filesystem_storage(str(tmp_path / "repo"))
    session = icechunk.Repository.create(storage).writable_session("main")
    with pytest.raises(
        NotImplementedError, match="to Icechunk: its chunks carry switch case IDs"
    ):
        combined.vz.to_icechunk(session.store)
    with pytest.raises(NotImplementedError, match="to Icechunk"):
        xr.DataTree(combined).vz.to_icechunk(session.store)


def test_zarr_json_lists_each_case_once(tmp_path):
    combined = combine(virtualize(era_files(tmp_path), tmp_path))
    doc = json.loads(
        zarr.core.sync.sync(
            store_of(combined, tmp_path).get("v/zarr.json", default_buffer_prototype())
        ).to_bytes()
    )
    (codec,) = doc["codecs"]
    assert codec["name"] == "switch"
    assert len(codec["configuration"]["cases"]) == 3
