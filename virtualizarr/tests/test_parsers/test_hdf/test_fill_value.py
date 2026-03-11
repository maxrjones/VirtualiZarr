"""Tests for fill value handling that previously failed due to xarray FillValueCoder coupling."""

import base64
import struct

import h5py
import numpy as np
import pytest

from virtualizarr.parsers.utils import encode_cf_fill_value
from virtualizarr.tests import requires_hdf5plugin, requires_imagecodecs
from virtualizarr.tests.utils import manifest_store_from_hdf_url


def _decode_b64_float(encoded: str) -> float:
    """Decode a base64-encoded little-endian double (matches FillValueCoder.decode)."""
    return struct.unpack("<d", base64.standard_b64decode(encoded))[0]


class TestEncodeCfFillValue:
    """Unit tests for encode_cf_fill_value."""

    def test_int(self):
        assert encode_cf_fill_value(np.int32(-9999), np.dtype("i4")) == -9999
        assert isinstance(encode_cf_fill_value(np.int32(-9999), np.dtype("i4")), int)

    def test_float(self):
        result = encode_cf_fill_value(np.float64(1.5), np.dtype("f8"))
        # Floats are base64-encoded for xarray FillValueCoder compatibility
        assert isinstance(result, str)
        assert _decode_b64_float(result) == 1.5

    def test_nan(self):
        result = encode_cf_fill_value(np.float64("nan"), np.dtype("f8"))
        assert isinstance(result, str)
        assert np.isnan(_decode_b64_float(result))

    def test_bool(self):
        result = encode_cf_fill_value(np.bool_(True), np.dtype("?"))
        assert result is True

    def test_string(self):
        result = encode_cf_fill_value("missing", np.dtype("U10"))
        assert result == "missing"
        assert isinstance(result, str)

    def test_bytes(self):
        result = encode_cf_fill_value(b"-", np.dtype("S1"))
        assert isinstance(result, str)  # base64 encoded
        assert base64.standard_b64decode(result) == b"-"

    def test_complex(self):
        result = encode_cf_fill_value(complex(1.0, 2.0), np.dtype("c16"))
        assert result == [1.0, 2.0]

    def test_numpy_complex(self):
        result = encode_cf_fill_value(np.complex128(1.0 + 2.0j), np.dtype("c16"))
        assert result == [1.0, 2.0]

    def test_numpy_array_scalar(self):
        result = encode_cf_fill_value(np.array(42), np.dtype("i4"))
        assert result == 42

    def test_numpy_array_1element(self):
        result = encode_cf_fill_value(np.array([3.14]), np.dtype("f8"))
        assert isinstance(result, str)
        assert _decode_b64_float(result) == pytest.approx(3.14)

    def test_numpy_array_multi_element_raises(self):
        with pytest.raises(ValueError, match="Expected a scalar"):
            encode_cf_fill_value(np.array([1, 2, 3]), np.dtype("i4"))

    def test_none(self):
        result = encode_cf_fill_value(None, np.dtype("f8"))
        assert result is None

    def test_int8_negative(self):
        """Regression: issue #628 - int8 _FillValue of -1 crashed FillValueCoder."""
        result = encode_cf_fill_value(np.int8(-1), np.dtype("i1"))
        assert result == -1
        assert isinstance(result, int)

    def test_uint32_sentinel(self):
        """Regression: issue #874 - large integer sentinel fill value."""
        result = encode_cf_fill_value(np.int32(2147483647), np.dtype("i4"))
        assert result == 2147483647


@requires_hdf5plugin
@requires_imagecodecs
class TestHDFFillValueParsing:
    """Integration tests: create HDF5 files with various fill values, parse them."""

    def test_int_fill_value(self, tmp_path):
        filepath = str(tmp_path / "int_fill.h5")
        with h5py.File(filepath, "w") as f:
            dset = f.create_dataset("data", data=np.arange(10, dtype="i4"), chunks=True)
            dset.attrs["_FillValue"] = np.int32(-9999)

        ms = manifest_store_from_hdf_url(f"file://{filepath}")
        metadata = ms._group.arrays["data"].metadata
        assert metadata.attributes["_FillValue"] == -9999
        assert isinstance(metadata.attributes["_FillValue"], int)

    def test_float_fill_value(self, tmp_path):
        filepath = str(tmp_path / "float_fill.h5")
        with h5py.File(filepath, "w") as f:
            dset = f.create_dataset("data", data=np.arange(10, dtype="f4"), chunks=True)
            dset.attrs["_FillValue"] = np.float32(-9999.0)

        ms = manifest_store_from_hdf_url(f"file://{filepath}")
        metadata = ms._group.arrays["data"].metadata
        # Float _FillValue is base64-encoded for xarray compatibility
        assert isinstance(metadata.attributes["_FillValue"], str)
        assert _decode_b64_float(metadata.attributes["_FillValue"]) == -9999.0

    def test_nan_fill_value(self, tmp_path):
        filepath = str(tmp_path / "nan_fill.h5")
        with h5py.File(filepath, "w") as f:
            dset = f.create_dataset("data", data=np.arange(10, dtype="f8"), chunks=True)
            dset.attrs["_FillValue"] = np.nan

        ms = manifest_store_from_hdf_url(f"file://{filepath}")
        metadata = ms._group.arrays["data"].metadata
        assert isinstance(metadata.attributes["_FillValue"], str)
        assert np.isnan(_decode_b64_float(metadata.attributes["_FillValue"]))

    def test_int8_negative_fill_value(self, tmp_path):
        """Regression: issue #628 - int8 _FillValue of -1."""
        filepath = str(tmp_path / "int8_fill.h5")
        with h5py.File(filepath, "w") as f:
            dset = f.create_dataset("data", data=np.zeros(5, dtype="i1"), chunks=True)
            dset.attrs["_FillValue"] = np.int8(-1)

        ms = manifest_store_from_hdf_url(f"file://{filepath}")
        metadata = ms._group.arrays["data"].metadata
        assert metadata.attributes["_FillValue"] == -1

    def test_large_int_sentinel_fill_value(self, tmp_path):
        """Regression: issue #874 - large integer sentinel fill value."""
        filepath = str(tmp_path / "sentinel_fill.h5")
        with h5py.File(filepath, "w") as f:
            dset = f.create_dataset("data", data=np.zeros(5, dtype="i4"), chunks=True)
            dset.attrs["_FillValue"] = np.int32(2147483647)

        ms = manifest_store_from_hdf_url(f"file://{filepath}")
        metadata = ms._group.arrays["data"].metadata
        assert metadata.attributes["_FillValue"] == 2147483647

    def test_bytes_fill_value(self, tmp_path):
        """Regression: issue #785 - bytes fill value b'-'."""
        filepath = str(tmp_path / "bytes_fill.h5")
        with h5py.File(filepath, "w") as f:
            dset = f.create_dataset(
                "data", data=np.array([b"abc", b"def"], dtype="S3"), chunks=True
            )
            dset.attrs["_FillValue"] = b"-"

        ms = manifest_store_from_hdf_url(f"file://{filepath}")
        metadata = ms._group.arrays["data"].metadata
        assert "_FillValue" in metadata.attributes
        # h5py returns bytes attrs as str, so it goes through the "U" path
        assert metadata.attributes["_FillValue"] == "-"

    def test_string_fill_value_attr(self, tmp_path):
        """Regression: issue #878 - string dtype fill values."""
        filepath = str(tmp_path / "str_fill.h5")
        with h5py.File(filepath, "w") as f:
            dset = f.create_dataset(
                "data",
                data=np.array(["hello", "world"], dtype="S10"),
                chunks=True,
            )
            dset.attrs["_FillValue"] = "missing"

        ms = manifest_store_from_hdf_url(f"file://{filepath}")
        metadata = ms._group.arrays["data"].metadata
        assert metadata.attributes["_FillValue"] == "missing"

    def test_bool_fill_value(self, tmp_path):
        filepath = str(tmp_path / "bool_fill.h5")
        with h5py.File(filepath, "w") as f:
            dset = f.create_dataset(
                "data",
                data=np.array([True, False, True]),
                chunks=True,
            )
            dset.attrs["_FillValue"] = False

        ms = manifest_store_from_hdf_url(f"file://{filepath}")
        metadata = ms._group.arrays["data"].metadata
        assert metadata.attributes["_FillValue"] is False

    def test_no_fill_value_attr(self, tmp_path):
        """Default HDF5 fill value (no _FillValue attr) should work for all dtypes."""
        filepath = str(tmp_path / "no_fv.h5")
        with h5py.File(filepath, "w") as f:
            f.create_dataset("data", data=np.zeros(5, dtype="f4"), chunks=True)

        ms = manifest_store_from_hdf_url(f"file://{filepath}")
        metadata = ms._group.arrays["data"].metadata
        assert metadata.fill_value is not None
        assert "_FillValue" not in metadata.attributes

    def test_string_dtype_default_fillvalue(self, tmp_path):
        """Regression: issue #878 - dataset.fillvalue.item() crashes for bytes."""
        filepath = str(tmp_path / "str_default.h5")
        with h5py.File(filepath, "w") as f:
            f.create_dataset(
                "data",
                data=np.array([b"a", b"b"], dtype="S1"),
                chunks=True,
            )

        ms = manifest_store_from_hdf_url(f"file://{filepath}")
        metadata = ms._group.arrays["data"].metadata
        assert metadata.fill_value is not None
