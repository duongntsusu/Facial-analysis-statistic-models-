#!/usr/bin/env python3
"""Calculate Fleiss's kappa from the image summary CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {
    "comparison_set",
    "n_raters",
    "style1_preference_count",
    "style2_preference_count",
    "skipped_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate Fleiss's kappa for inter-rater agreement."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="/Users/leoli/Downloads/FINAL_Image_Summary.csv",
        help="Path to FINAL_Image_Summary.csv.",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Treat skipped judgments as a third rating category.",
    )
    return parser.parse_args()


def require_columns(reader: csv.DictReader, csv_path: Path) -> None:
    fieldnames = set(reader.fieldnames or [])
    missing = sorted(REQUIRED_COLUMNS - fieldnames)
    if missing:
        raise SystemExit(f"{csv_path} is missing required columns: {', '.join(missing)}")


def parse_count(row: dict[str, str], column: str, row_number: int) -> int:
    raw_value = (row.get(column) or "").strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise SystemExit(
            f"Row {row_number}: {column} must be an integer, got {raw_value!r}"
        ) from exc
    if value < 0:
        raise SystemExit(f"Row {row_number}: {column} cannot be negative")
    return value


def item_agreement(category_counts: list[int], n_raters: int) -> float:
    """Return P_i for one item."""
    if n_raters < 2:
        raise SystemExit("Fleiss's kappa requires at least two raters per item")
    numerator = sum(count * (count - 1) for count in category_counts)
    return numerator / (n_raters * (n_raters - 1))


def fleiss_kappa(rows: list[list[int]], n_raters: int) -> tuple[float, float, float, list[float]]:
    """Return kappa, observed agreement, expected agreement, and category proportions."""
    if not rows:
        raise SystemExit("No rows found in CSV")

    n_items = len(rows)
    category_totals = [0] * len(rows[0])
    observed_agreements = []

    for counts in rows:
        observed_agreements.append(item_agreement(counts, n_raters))
        for index, count in enumerate(counts):
            category_totals[index] += count

    observed = sum(observed_agreements) / n_items
    category_proportions = [
        total / (n_items * n_raters) for total in category_totals
    ]
    expected = sum(proportion * proportion for proportion in category_proportions)

    if expected == 1:
        raise SystemExit("Expected agreement is 1.0, so Fleiss's kappa is undefined")

    kappa = (observed - expected) / (1 - expected)
    return kappa, observed, expected, category_proportions


def load_counts(csv_path: Path, include_skipped: bool) -> tuple[list[list[int]], int, dict[str, list[list[int]]]]:
    counts_by_item: list[list[int]] = []
    counts_by_set: dict[str, list[list[int]]] = defaultdict(list)
    expected_n_raters: int | None = None

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        require_columns(reader, csv_path)

        for row_number, row in enumerate(reader, start=1):
            n_raters = parse_count(row, "n_raters", row_number)
            style1 = parse_count(row, "style1_preference_count", row_number)
            style2 = parse_count(row, "style2_preference_count", row_number)
            skipped = parse_count(row, "skipped_count", row_number)

            if expected_n_raters is None:
                expected_n_raters = n_raters
            elif n_raters != expected_n_raters:
                raise SystemExit(
                    "All rows must have the same n_raters for Fleiss's kappa. "
                    f"Expected {expected_n_raters}, found {n_raters} on row {row_number}."
                )

            category_counts = [style1, style2, skipped] if include_skipped else [style1, style2]
            counted_votes = sum(category_counts)
            if counted_votes != n_raters:
                mode = "including skipped" if include_skipped else "excluding skipped"
                raise SystemExit(
                    f"Row {row_number}: category counts add to {counted_votes}, "
                    f"but n_raters is {n_raters} ({mode})."
                )

            counts_by_item.append(category_counts)
            counts_by_set[row.get("comparison_set", "unknown")].append(category_counts)

    if expected_n_raters is None:
        raise SystemExit("No rows found in CSV")

    return counts_by_item, expected_n_raters, counts_by_set


def print_result(label: str, counts: list[list[int]], n_raters: int, categories: list[str]) -> None:
    kappa, observed, expected, proportions = fleiss_kappa(counts, n_raters)
    print(label)
    print(f"  items: {len(counts)}")
    print(f"  raters per item: {n_raters}")
    print(f"  observed agreement: {observed:.6f}")
    print(f"  expected agreement: {expected:.6f}")
    print(f"  Fleiss's kappa: {kappa:.6f}")
    print("  category proportions:")
    for category, proportion in zip(categories, proportions):
        print(f"    {category}: {proportion:.6f}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    categories = ["style1", "style2"]
    if args.include_skipped:
        categories.append("skipped")

    counts_by_item, n_raters, counts_by_set = load_counts(csv_path, args.include_skipped)
    print_result("Overall", counts_by_item, n_raters, categories)

    if len(counts_by_set) > 1:
        print()
        for comparison_set in sorted(counts_by_set):
            print_result(comparison_set, counts_by_set[comparison_set], n_raters, categories)
            print()


if __name__ == "__main__":
    main()
