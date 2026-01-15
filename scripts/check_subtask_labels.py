import argparse
import json
import os
from pathlib import Path

import h5py


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _sorted_demo_keys(data_group):
    demo_keys = list(data_group.keys())

    def demo_index(key):
        try:
            return int(key.split("_")[-1])
        except (ValueError, AttributeError):
            return key

    return sorted(demo_keys, key=demo_index)


def check_demo_attrs(demo_key, demo_group):
    """Return a tuple (has_attr, has_nl, count) and optionally parsed entries."""
    if "subtask_history" not in demo_group.attrs:
        return False, False, 0, None

    raw = _decode_attr(demo_group.attrs["subtask_history"])
    try:
        parsed = json.loads(raw)
    except Exception:
        return True, False, 0, None

    history = parsed.get("subtask_history", [])
    has_nl = any(
        isinstance(entry, dict) and entry.get("natural_language")
        for entry in history
    )
    return True, has_nl, len(history), parsed


def check_dataset(path, max_demos=None, verbose=False):
    path = Path(path)
    if not path.exists():
        print(f"[skip] Dataset not found: {path}")
        return

    with h5py.File(path, "r") as f:
        data_group = f["data"]
        demo_keys = _sorted_demo_keys(data_group)
        if max_demos is not None and max_demos >= 0:
            demo_keys = demo_keys[:max_demos]

        found = missing = no_nl = 0
        for demo_key in demo_keys:
            has_attr, has_nl, count, parsed = check_demo_attrs(
                demo_key, data_group[demo_key]
            )
            if not has_attr:
                missing += 1
                if verbose:
                    print(f"  {demo_key}: no subtask_history attr")
                continue
            found += 1
            if not has_nl:
                no_nl += 1
                if verbose:
                    print(f"  {demo_key}: subtasks={count}, missing natural_language")
            elif verbose:
                print(f"  {demo_key}: subtasks={count}, has natural_language")

        total = len(demo_keys)
        print(
            f"{path.name}: demos checked {total}, with labels {found}, without labels {missing}, without natural_language {no_nl}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Check whether subtask labels (with natural language) are stored in LIBERO HDF5 demos."
    )
    parser.add_argument("--dataset", help="Path to a single HDF5 dataset.")
    parser.add_argument(
        "--dataset-dir",
        help="Directory containing HDF5 datasets (all *.hdf5 / *.h5 will be checked).",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Limit number of demos per dataset (use -1 for all).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-demo status details."
    )
    args = parser.parse_args()

    max_demos = None if (args.max_demos is not None and args.max_demos < 0) else args.max_demos

    dataset_paths = []
    if args.dataset_dir:
        for root, _, files in os.walk(args.dataset_dir):
            for fname in files:
                if fname.endswith(".hdf5") or fname.endswith(".h5"):
                    dataset_paths.append(Path(root) / fname)
        dataset_paths.sort()
    if args.dataset:
        dataset_paths.append(Path(args.dataset))

    if not dataset_paths:
        print("No datasets provided. Use --dataset or --dataset-dir.")
        return

    for path in dataset_paths:
        check_dataset(path, max_demos=max_demos, verbose=args.verbose)


if __name__ == "__main__":
    main()
