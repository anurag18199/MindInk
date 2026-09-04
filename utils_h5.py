"""
utils_h5.py

Helpers for reading MATLAB v7.3 (.mat) files using h5py.

Important:
    MATLAB v7.3 files are HDF5 containers.
    MATLAB structs are commonly represented using HDF5 object references.

This module deliberately does NOT recursively walk the #refs# group.
The #refs# group can contain a very large internal object graph and is not
useful for understanding the logical MATLAB structure.

The inspection workflow is:

    top-level dataset
        -> HDF5 reference
            -> referenced MATLAB struct/group
                -> fields
                    -> more references
                        -> actual arrays

MUST contain:
    - HDF5 reference detection
    - reference dereferencing
    - MATLAB char-array decoding
    - EEG orientation handling
    - safe logical-structure inspection

MUST NOT contain:
    - assumptions about ZuCo field names
    - recursive traversal of #refs#
"""

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

def open_mat(path):
    """Open a MATLAB v7.3 .mat file read-only."""
    return h5py.File(str(path), "r")


# ---------------------------------------------------------------------------
# HDF5 reference handling
# ---------------------------------------------------------------------------

def is_ref(obj):
    """
    Return True if obj is an HDF5 object reference.
    """
    return isinstance(obj, h5py.h5r.Reference)


def is_ref_dtype(dtype):
    """
    Return True if an HDF5 dataset contains object references.
    """
    return h5py.check_dtype(ref=dtype) is not None


def deref(f, ref):
    """
    Dereference one MATLAB/HDF5 object reference.

    Returns the referenced HDF5 object.
    """
    if not is_ref(ref):
        raise TypeError(f"Expected HDF5 reference, got {type(ref)}")

    if not ref:
        raise ValueError("Invalid/null HDF5 reference")

    return f[ref]


def deref_value(f, ref):
    """
    Dereference one reference and return its stored value.
    """
    obj = deref(f, ref)
    return obj[()]


def reference_target_name(f, ref):
    """
    Return the HDF5 path pointed to by a reference.

    Useful for debugging the MATLAB reference structure.
    """
    if not is_ref(ref) or not ref:
        return None

    try:
        return f[ref].name
    except Exception:
        return None


# ---------------------------------------------------------------------------
# MATLAB character/string handling
# ---------------------------------------------------------------------------

def mat_to_str(f, obj_or_ref):
    """
    Decode a MATLAB character array.

    MATLAB strings/chars may appear as:
        - an HDF5 reference to a uint16 dataset
        - a direct uint16 array
        - a scalar/array of character codes

    Returns:
        Python str
    """
    try:
        if is_ref(obj_or_ref):
            arr = deref_value(f, obj_or_ref)
        elif isinstance(obj_or_ref, h5py.Dataset):
            arr = obj_or_ref[()]
        else:
            arr = np.asarray(obj_or_ref)
    except Exception:
        return ""

    arr = np.asarray(arr).flatten()

    if arr.size == 0:
        return ""

    chars = []

    for value in arr:
        try:
            code = int(value)

            # MATLAB strings normally contain Unicode code points.
            if code != 0:
                chars.append(chr(code))

        except Exception:
            return ""

    return "".join(chars)


# ---------------------------------------------------------------------------
# Array handling
# ---------------------------------------------------------------------------

def as_channels_by_time(arr, n_channels_hint=256):
    """
    Convert a 2-D EEG array to:

        (channels, time)

    MATLAB/HDF5 orientation may be reversed relative to what Python expects.

    The shorter dimension is assumed to be the channel dimension because
    EEG recordings normally have:

        channels << number_of_time_samples

    Returns:
        np.ndarray or None
    """
    a = np.asarray(arr)

    if a.ndim != 2 or a.size == 0:
        return None

    r, c = a.shape

    # Typical EEG:
    #   (channels, time)
    # or
    #   (time, channels)

    if r <= n_channels_hint and r <= c:
        return a

    if c <= n_channels_hint and c < r:
        return a.T

    # Ambiguous case.
    # Return the original array and let the caller inspect it.
    return a


# ---------------------------------------------------------------------------
# Reference-array helpers
# ---------------------------------------------------------------------------

def get_dataset_refs(dataset):
    """
    Return all references contained in an HDF5 reference dataset.

    Example:
        sentenceData -> (N, 1) reference dataset

    Returns:
        list of HDF5 references
    """
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError("Expected an HDF5 Dataset")

    if not is_ref_dtype(dataset.dtype):
        raise TypeError(
            f"Dataset {dataset.name} does not contain HDF5 references. "
            f"dtype={dataset.dtype}"
        )

    return list(np.asarray(dataset[()]).flatten())


def first_valid_ref(dataset):
    """
    Return the first non-null reference from a reference dataset.
    """
    refs = get_dataset_refs(dataset)

    for ref in refs:
        if is_ref(ref) and bool(ref):
            return ref

    return None


# ---------------------------------------------------------------------------
# Logical MATLAB structure inspection
# ---------------------------------------------------------------------------

def describe_dataset(dataset):
    """
    Return basic information about an HDF5 dataset.
    """
    return {
        "path": dataset.name,
        "shape": dataset.shape,
        "dtype": str(dataset.dtype),
        "is_reference_array": is_ref_dtype(dataset.dtype),
    }


def describe_group(group):
    """
    Describe the immediate children of an HDF5 group.

    This intentionally does NOT recursively enter #refs#.
    """
    result = []

    for name, obj in group.items():

        if isinstance(obj, h5py.Dataset):
            result.append({
                "name": name,
                "kind": "dataset",
                "shape": obj.shape,
                "dtype": str(obj.dtype),
                "is_reference_array": is_ref_dtype(obj.dtype),
            })

        elif isinstance(obj, h5py.Group):
            result.append({
                "name": name,
                "kind": "group",
                "shape": None,
                "dtype": None,
                "is_reference_array": False,
            })

    return result


# ---------------------------------------------------------------------------
# Safe structure walker
# ---------------------------------------------------------------------------

def walk(f, root="/", max_depth=4, _depth=0):
    """
    Walk the LOGICAL HDF5 structure while skipping MATLAB's internal #refs#.

    This replaces the old recursive walker that became trapped in:

        /#refs#/
            ...
            ...
            ...

    Yields:
        (path, kind, shape, dtype)
    """

    node = f[root]

    if isinstance(node, h5py.Dataset):

        dt = node.dtype

        if is_ref_dtype(dt):
            kind = "ref-array"
        else:
            kind = "dataset"

        yield (
            root,
            kind,
            node.shape,
            str(dt),
        )

        return

    # Group
    yield (
        root,
        "group",
        None,
        None,
    )

    if _depth >= max_depth:
        return

    for key in node.keys():

        # ---------------------------------------------------------------
        # CRITICAL:
        # Do not recursively walk MATLAB's internal reference database.
        # ---------------------------------------------------------------
        if key == "#refs#":
            continue

        child = f"{root.rstrip('/')}/{key}"

        yield from walk(
            f,
            child,
            max_depth=max_depth,
            _depth=_depth + 1,
        )