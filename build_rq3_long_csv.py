#!/usr/bin/env python3
"""Create the RQ3 long-format AU CSV from photo_comparison_aggregated.xlsx."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


DEFAULT_WORKBOOK = Path(
    "outputs/019f6265-9162-7c13-b594-47e0099df6c8/" "photo_comparison_aggregated.xlsx"
)
DEFAULT_OUTPUT = Path("data/rq3_au_long.csv")
SHEET_NAME = "AU Presence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the aggregated workbook's AU Presence sheet to RQ3 long form."
    )
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalize_base_id(values: pd.Series) -> pd.Series:
    """Preserve the four-digit project base-image identifier."""
    normalized = (
        values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    )
    return normalized.str.zfill(4)


def build_long_data(workbook_path: Path) -> pd.DataFrame:
    wide = pd.read_excel(workbook_path, sheet_name=SHEET_NAME, header=1)
    wide.columns = [str(column).strip() for column in wide.columns]
    required = {
        "dataset",
        "comparison_set",
        "base_id",
        "image_variant",
        "winning_votes",
        "filename",
        "image_path",
        "actor_id",
    }
    missing = sorted(required - set(wide.columns))
    if missing:
        raise ValueError(f"{SHEET_NAME!r} is missing required column(s): {missing}")

    au_columns = [column for column in wide.columns if re.fullmatch(r"AU\d+", column)]
    if not au_columns:
        raise ValueError(
            "No AU columns matching names such as AU01 or AU12 were found."
        )

    wide["dataset"] = wide["dataset"].astype("string").str.strip().str.casefold()
    wide["base_id"] = normalize_base_id(wide["base_id"])
    wide["actor_id"] = wide["actor_id"].astype("string").str.strip()
    wide["filename"] = wide["filename"].astype("string").str.strip()
    for column in au_columns:
        wide[column] = pd.to_numeric(wide[column], errors="coerce")

    invalid_binary = ~wide[au_columns].isin([0, 1])
    if invalid_binary.any().any():
        bad_columns = invalid_binary.any()[invalid_binary.any()].index.tolist()
        raise ValueError(
            "AU Presence contains missing or non-binary values in: "
            + ", ".join(bad_columns)
        )
    wide[au_columns] = wide[au_columns].astype(int)

    original = wide.loc[wide["dataset"] == "original"].copy()
    stylized = wide.loc[wide["dataset"].isin(["disney", "ghibli"])].copy()
    unknown = sorted(set(wide["dataset"].dropna()) - {"original", "disney", "ghibli"})
    if unknown:
        print(f"WARNING: ignoring unrecognized dataset rows: {unknown}")
    if original.empty or stylized.empty:
        raise ValueError("Both original and Disney/Ghibli rows are required.")
    duplicate_original = original["base_id"].duplicated(keep=False)
    if duplicate_original.any():
        duplicate_ids = (
            original.loc[duplicate_original, "base_id"].unique().tolist()[:10]
        )
        raise ValueError(f"Original AU rows are not unique by base_id: {duplicate_ids}")
    if stylized["filename"].duplicated().any():
        raise ValueError("Generated-image filenames are not unique.")

    candidates_per_cell = stylized.groupby(["base_id", "dataset"], observed=True)[
        "filename"
    ].nunique()
    if candidates_per_cell.ne(1).any():
        print(
            "WARNING: some base-image/style cells contain other than one generated image. "
            "All available generated rows will be retained."
        )

    original_lookup = original.set_index("base_id")
    unmatched = sorted(set(stylized["base_id"]) - set(original_lookup.index))
    if unmatched:
        raise ValueError(
            f"Stylized rows lack a matching processed original: {unmatched[:10]}"
        )

    original_long = original.melt(
        id_vars=["base_id", "filename", "image_path", "actor_id"],
        value_vars=au_columns,
        var_name="au",
        value_name="original_au_presence",
    ).rename(
        columns={
            "filename": "original_filename",
            "image_path": "original_image_path",
            "actor_id": "original_actor_id",
        }
    )
    stylized_long = stylized.melt(
        id_vars=[
            "dataset",
            "base_id",
            "image_variant",
            "winning_votes",
            "filename",
            "image_path",
            "actor_id",
        ],
        value_vars=au_columns,
        var_name="au",
        value_name="stylized_au_presence",
    ).rename(
        columns={
            "dataset": "style",
            "image_variant": "option_id",
            "filename": "stylized_filename",
            "image_path": "stylized_image_path",
            "actor_id": "stylized_actor_id",
        }
    )
    long_data = stylized_long.merge(
        original_long,
        on=["base_id", "au"],
        how="left",
        validate="many_to_one",
    )
    if long_data["original_au_presence"].isna().any():
        raise ValueError(
            "At least one stylized AU row did not match an original AU estimate."
        )
    actor_mismatch = long_data["stylized_actor_id"] != long_data["original_actor_id"]
    if actor_mismatch.any():
        examples = long_data.loc[
            actor_mismatch, ["base_id", "stylized_actor_id", "original_actor_id"]
        ].drop_duplicates()
        raise ValueError(
            f"Actor mapping mismatch between original and stylized rows:\n{examples.head()}"
        )

    long_data["style"] = long_data["style"].str.title()
    long_data["person_id"] = long_data["stylized_actor_id"]
    long_data["base_image_id"] = long_data["base_id"]
    long_data["generated_image_id"] = long_data["stylized_filename"]
    # Every row on AU Presence represents an image for which Py-Feat produced AU output.
    long_data["face_detected"] = 1
    # The aggregated workbook contains the human-selected winner for each base/style cell.
    long_data["human_selected"] = 1
    long_data["source_workbook"] = workbook_path.name

    columns = [
        "person_id",
        "base_image_id",
        "generated_image_id",
        "style",
        "option_id",
        "au",
        "original_au_presence",
        "stylized_au_presence",
        "face_detected",
        "human_selected",
        "winning_votes",
        "original_filename",
        "stylized_filename",
        "original_image_path",
        "stylized_image_path",
        "source_workbook",
    ]
    long_data = long_data[columns].copy()
    for column in ["original_au_presence", "stylized_au_presence", "face_detected"]:
        long_data[column] = long_data[column].astype(int)
    long_data = long_data.sort_values(
        ["style", "base_image_id", "generated_image_id", "au"], kind="stable"
    ).reset_index(drop=True)
    return long_data


def validate_and_report(data: pd.DataFrame) -> None:
    group_counts = {
        "rows": len(data),
        "people": data["person_id"].nunique(),
        "base_images": data["base_image_id"].nunique(),
        "generated_images": data["generated_image_id"].nunique(),
        "aus": data["au"].nunique(),
    }
    expected_rows = group_counts["generated_images"] * group_counts["aus"]
    if len(data) != expected_rows:
        raise ValueError(
            f"Expected generated_images × AUs = {expected_rows:,} rows, found {len(data):,}."
        )
    rows_per_generated = data.groupby("generated_image_id", observed=True).size()
    if rows_per_generated.nunique() != 1:
        raise ValueError(
            "Generated images do not all contribute the same number of AU rows."
        )
    print("Long-format RQ3 data validation")
    for label, value in group_counts.items():
        print(f"  {label}: {value:,}")
    print("  rows by style:")
    for style, count in data["style"].value_counts().sort_index().items():
        print(f"    {style}: {count:,}")
    print(
        "  face_detected is 1 because AU Presence contains successfully processed images."
    )
    print("  human_selected is 1 because the workbook contains winner images only.")


def main() -> None:
    args = parse_args()
    workbook_path = args.workbook.resolve()
    output_path = args.output.resolve()
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")
    data = build_long_data(workbook_path)
    validate_and_report(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
