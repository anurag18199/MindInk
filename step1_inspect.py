"""
step1_inspect.py

MindInk / ZuCo Step 1
---------------------

Purpose:
    Inspect the actual MATLAB v7.3 / HDF5 structure of one ZuCo file.

This version is designed for the structure discovered in the current
ZuCo 2.0 NR file:

    /sentenceData
        /allFixations
        /content
        /rawData
        /word
        /wordbounds
        /mean_...
        ...

Each of these fields may contain HDF5 object references.

Step 1 must establish:

    1. sentenceData exists
    2. rawData exists
    3. rawData can be dereferenced
    4. rawData contains time-domain EEG
    5. EEG channel dimension is identifiable
    6. fixation/timing information exists
    7. word/text information exists

Do NOT start preprocessing or ICA until these are confirmed.
"""

import sys
import numpy as np
import h5py

import utils_h5 as H


# ============================================================================
# Generic helpers
# ============================================================================

def print_separator(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def describe_object(name, obj, indent="  "):
    """
    Print basic information about one HDF5 object.
    """

    if isinstance(obj, h5py.Group):

        print(
            f"{indent}{name:<30} "
            f"[group]"
        )

    elif isinstance(obj, h5py.Dataset):

        is_ref = H.is_ref_dtype(obj.dtype)

        print(
            f"{indent}{name:<30} "
            f"[dataset] "
            f"shape={obj.shape} "
            f"dtype={obj.dtype} "
            f"refs={is_ref}"
        )


# ============================================================================
# Reference inspection
# ============================================================================

def inspect_reference_field(f, parent_group, field_name, max_refs=3):
    """
    Inspect one reference-array field under sentenceData.

    Example:

        sentenceData/rawData
            |
            +-- reference 0
            +-- reference 1
            ...

    We inspect only a few references so that the script remains fast.
    """

    print(f"\n--- {field_name} ---")

    if field_name not in parent_group:

        print("Field does not exist.")
        return []

    dataset = parent_group[field_name]

    if not isinstance(dataset, h5py.Dataset):

        print(
            f"Unexpected object type: "
            f"{type(dataset).__name__}"
        )

        return []

    print(f"Path       : {dataset.name}")
    print(f"Shape      : {dataset.shape}")
    print(f"Dtype      : {dataset.dtype}")
    print(
        f"References : "
        f"{H.is_ref_dtype(dataset.dtype)}"
    )

    if not H.is_ref_dtype(dataset.dtype):

        print("This is not a reference-array dataset.")

        try:
            value = dataset[()]

            arr = np.asarray(value)

            print(
                f"Direct value shape: "
                f"{arr.shape}"
            )

        except Exception as exc:

            print(
                f"Could not read direct value: "
                f"{exc}"
            )

        return []

    refs = H.get_dataset_refs(dataset)

    print(f"Number of references: {len(refs)}")

    valid_refs = [
        ref
        for ref in refs
        if isinstance(ref, h5py.h5r.Reference)
        and bool(ref)
    ]

    print(
        f"Valid references: "
        f"{len(valid_refs)}"
    )

    targets = []

    for i, ref in enumerate(valid_refs[:max_refs]):

        try:

            target = f[ref]

            print(
                f"\n  Reference {i}"
            )

            print(
                f"      target path : "
                f"{target.name}"
            )

            print(
                f"      target type : "
                f"{type(target).__name__}"
            )

            if isinstance(target, h5py.Dataset):

                print(
                    f"      target shape: "
                    f"{target.shape}"
                )

                print(
                    f"      target dtype: "
                    f"{target.dtype}"
                )

                # --------------------------------------------------------
                # Show a tiny preview for non-reference arrays.
                # --------------------------------------------------------

                if not H.is_ref_dtype(target.dtype):

                    try:

                        arr = np.asarray(target[()])

                        print(
                            f"      numpy shape : "
                            f"{arr.shape}"
                        )

                        if arr.size <= 20:

                            print(
                                f"      values      : "
                                f"{arr.flatten()}"
                            )

                    except Exception as exc:

                        print(
                            f"      preview error: "
                            f"{exc}"
                        )

            elif isinstance(target, h5py.Group):

                print(
                    "      group fields:"
                )

                for child_name, child in target.items():

                    describe_object(
                        child_name,
                        child,
                        indent="        "
                    )

            targets.append(target)

        except Exception as exc:

            print(
                f"  Reference {i} could not be "
                f"dereferenced: {exc}"
            )

    return targets


# ============================================================================
# Raw EEG candidate inspection
# ============================================================================

def inspect_rawdata_targets(f, targets):
    """
    Inspect the objects referenced by sentenceData/rawData.

    We need to determine whether these are:
        - raw/time-domain EEG arrays
        - sentence-level containers
        - word-level containers
        - something else
    """

    print_separator(
        "RAW EEG CANDIDATE INSPECTION"
    )

    if not targets:

        print(
            "No rawData references could be inspected."
        )

        return

    for i, target in enumerate(targets):

        print(
            f"\nRAW DATA TARGET {i}"
        )

        print(
            f"Path : {target.name}"
        )

        print(
            f"Type : {type(target).__name__}"
        )

        if isinstance(target, h5py.Dataset):

            arr = np.asarray(target[()])

            print(
                f"Shape: {arr.shape}"
            )

            print(
                f"Dtype: {arr.dtype}"
            )

            # ------------------------------------------------------------
            # Determine whether the array looks EEG-like.
            # ------------------------------------------------------------

            if arr.ndim == 2:

                r, c = arr.shape

                print(
                    f"2-D dimensions: "
                    f"{r} × {c}"
                )

                if r <= 256 and c > r:

                    print(
                        "  Interpretation: "
                        "likely channels × time"
                    )

                elif c <= 256 and r > c:

                    print(
                        "  Interpretation: "
                        "likely time × channels"
                    )

                else:

                    print(
                        "  Interpretation: "
                        "orientation ambiguous"
                    )

            elif arr.ndim == 1:

                print(
                    "  Interpretation: "
                    "one-dimensional signal/vector"
                )

            else:

                print(
                    "  Interpretation: "
                    f"{arr.ndim}-D array; "
                    "requires further inspection"
                )

        elif isinstance(target, h5py.Group):

            print(
                "Referenced rawData object is a GROUP."
            )

            print(
                "Immediate fields:"
            )

            for name, obj in target.items():

                describe_object(
                    name,
                    obj,
                    indent="  "
                )


# ============================================================================
# Text and timing inspection
# ============================================================================

def inspect_text_and_timing(f, sentence_group):

    print_separator(
        "WORD / TEXT / TIMING FIELDS"
    )

    fields = [
        "content",
        "word",
        "wordbounds",
        "allFixations",
    ]

    for field in fields:

        if field not in sentence_group:

            print(
                f"{field:<20} : NOT FOUND"
            )

            continue

        dataset = sentence_group[field]

        print(
            f"{field:<20} : "
            f"shape={dataset.shape}, "
            f"dtype={dataset.dtype}, "
            f"refs={H.is_ref_dtype(dataset.dtype)}"
        )

        if H.is_ref_dtype(dataset.dtype):

            try:

                refs = H.get_dataset_refs(
                    dataset
                )

                first = None

                for ref in refs:

                    if bool(ref):

                        first = ref
                        break

                if first is None:

                    print(
                        "    No valid reference."
                    )

                    continue

                target = f[first]

                print(
                    f"    first target: "
                    f"{target.name}"
                )

                print(
                    f"    type        : "
                    f"{type(target).__name__}"
                )

                if isinstance(
                    target,
                    h5py.Dataset
                ):

                    print(
                        f"    shape       : "
                        f"{target.shape}"
                    )

                    print(
                        f"    dtype       : "
                        f"{target.dtype}"
                    )

                    try:

                        arr = np.asarray(
                            target[()]
                        )

                        print(
                            f"    value shape : "
                            f"{arr.shape}"
                        )

                        # ------------------------------------------------
                        # For small numeric arrays show values.
                        # ------------------------------------------------

                        if arr.size <= 50:

                            print(
                                f"    values      : "
                                f"{arr.flatten()}"
                            )

                    except Exception as exc:

                        print(
                            f"    value read error: "
                            f"{exc}"
                        )

                elif isinstance(
                    target,
                    h5py.Group
                ):

                    print(
                        "    group fields:"
                    )

                    for name, obj in target.items():

                        describe_object(
                            name,
                            obj,
                            indent="      "
                        )

            except Exception as exc:

                print(
                    f"    dereference error: "
                    f"{exc}"
                )


# ============================================================================
# Main inspection
# ============================================================================

def inspect(path):

    print_separator(
        "MindInk / ZuCo Step 1 — "
        "HDF5 Structure Inspection"
    )

    print(f"\nFile:\n  {path}")

    with H.open_mat(path) as f:

        # =================================================================
        # 1. TOP LEVEL
        # =================================================================

        print_separator(
            "1. TOP-LEVEL KEYS"
        )

        for name, obj in f.items():

            describe_object(
                name,
                obj
            )

        # =================================================================
        # 2. LOGICAL STRUCTURE
        # =================================================================

        print_separator(
            "2. LOGICAL STRUCTURE "
            "(#refs# excluded)"
        )

        for (
            node_path,
            kind,
            shape,
            dtype
        ) in H.walk(
            f,
            "/",
            max_depth=3
        ):

            if node_path == "/":

                print("/")

                continue

            depth = node_path.strip(
                "/"
            ).count("/")

            indent = "  " * depth

            print(
                f"{indent}{node_path} "
                f"[{kind}] "
                f"shape={shape} "
                f"dtype={dtype}"
            )

        # =================================================================
        # 3. SENTENCE DATA GROUP
        # =================================================================

        print_separator(
            "3. sentenceData"
        )

        if "sentenceData" not in f:

            print(
                "ERROR: sentenceData not found."
            )

            return

        sentence_data = f[
            "sentenceData"
        ]

        print(
            f"Path : {sentence_data.name}"
        )

        print(
            f"Type : "
            f"{type(sentence_data).__name__}"
        )

        if not isinstance(
            sentence_data,
            h5py.Group
        ):

            print(
                "ERROR: sentenceData is not "
                "a group as expected."
            )

            return

        print(
            "\nFields inside sentenceData:"
        )

        for name, obj in sentence_data.items():

            describe_object(
                name,
                obj,
                indent="  "
            )

        # =================================================================
        # 4. RAW DATA
        # =================================================================

        print_separator(
            "4. rawData"
        )

        raw_targets = inspect_reference_field(
            f,
            sentence_data,
            "rawData",
            max_refs=3
        )

        inspect_rawdata_targets(
            f,
            raw_targets
        )

        # =================================================================
        # 5. TEXT + TIMING
        # =================================================================

        inspect_text_and_timing(
            f,
            sentence_data
        )

        # =================================================================
        # 6. FINAL VERDICT
        # =================================================================

        print_separator(
            "6. STEP-1 VERDICT"
        )

        print(
            """
We are looking for:

    sentenceData
        |
        +-- rawData
        |     |
        |     +-- actual time-domain EEG
        |
        +-- allFixations
        |
        +-- word
        |
        +-- wordbounds
        |
        +-- content

A 105-element vector alone is NOT enough to prove raw EEG.

Step 1 is complete only after rawData has been dereferenced and its
actual numerical contents/dimensions have been inspected.

Do NOT start preprocessing or ICA until this verification is complete.
"""
        )


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "  python step1_inspect.py "
            "dataset-NR/resultsYAG_NR.mat"
        )

        sys.exit(1)

    inspect(
        sys.argv[1]
    )