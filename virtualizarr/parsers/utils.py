import base64
import struct
from typing import Any

import numpy as np

#: JSON-serializable scalar types produced by :func:`encode_cf_fill_value`.
JSONScalar = int | float | bool | str | list[float] | tuple | None


def encode_cf_fill_value(
    data: object,
    dtype: np.dtype[Any],
) -> JSONScalar:
    """
    Serialize a scalar value to a JSON-compatible form for a zarr ``_FillValue`` attribute.

    The encoding is dtype-dispatched and must match what xarray's
    ``FillValueCoder.decode()`` expects:

    - floats: base64-encoded little-endian double (str)
    - bytes (dtype kind "S"): base64-encoded (str)
    - ints: Python int
    - bools: Python bool
    - strings (dtype kind "U"): Python str

    For dtypes that ``FillValueCoder`` does not support (complex, structured),
    this function handles them directly instead of raising.

    See `xarray's FillValueCoder
    <https://github.com/pydata/xarray/blob/c9ac506d723485adc7e75934739b3093227cce08/xarray/backends/zarr.py#L118-L141>`_
    for the corresponding decode implementation.

    Parameters
    ----------
    data : object
        A scalar fill value. Numpy scalars and 0-d/1-element arrays are
        converted to Python natives before encoding.
    dtype : np.dtype
        The dtype of the array this fill value belongs to.

    Returns
    -------
    JSONScalar
        A JSON-serializable representation of the scalar.
    """
    # Extract Python native from numpy containers
    if isinstance(data, np.ndarray):
        if data.size > 1:
            raise ValueError("Expected a scalar")
        data = data.item()
    elif isinstance(data, np.generic):
        data = data.item()

    if data is None:
        return data

    if dtype.kind in "S":
        if isinstance(data, bytes):
            return base64.standard_b64encode(data).decode()
        # h5py may return byte attrs as str
        return str(data)
    if dtype.kind in "b":
        return bool(data)
    if dtype.kind in "iu":
        return int(data)
    if dtype.kind in "f":
        return base64.standard_b64encode(struct.pack("<d", float(data))).decode()
    if dtype.kind in "U":
        return str(data)
    if dtype.kind in "c":
        # Complex: xarray doesn't handle this yet, store as [real, imag]
        c = complex(data)
        return [c.real, c.imag]
    if dtype.names is not None:
        # Structured/compound dtype — pass through as tuple
        if isinstance(data, tuple):
            return data
        return data

    # Fallback for unknown dtypes
    return str(data)
