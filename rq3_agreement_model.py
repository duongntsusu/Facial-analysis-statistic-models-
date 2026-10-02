#!/usr/bin/env python3
"""RQ3: original-to-stylized binary AU agreement mixed-effects analysis."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

if __package__:
    from .rq3_helpers import (
        au_specific_clustered_models,
        fixed_effect_table,
        fit_bayesian_glmm,
        model_diagnostics,
        plain_language_interpretation,
        plot_predicted_probabilities,
        plot_transition_distribution,
        posterior_predictions_and_contrasts,
        prepare_analysis_data,
        random_effect_table,
        transition_outputs,
        warn_user,
        write_text,
    )
else:
    from rq3_helpers import (
        au_specific_clustered_models,
        fixed_effect_table,
        fit_bayesian_glmm,
        model_diagnostics,
        plain_language_interpretation,
        plot_predicted_probabilities,
        plot_transition_distribution,
        posterior_predictions_and_contrasts,
        prepare_analysis_data,
        random_effect_table,
        transition_outputs,
        warn_user,
        write_text,
    )


# -----------------------------------------------------------------------------
# Configuration: edit these defaults to match the long-format AU CSV.
# Command-line arguments can override paths and the human-selection column.
# -----------------------------------------------------------------------------
INPUT_CSV = Path("data/rq3_au_long.csv")
OUTPUT_DIRECTORY = Path("results")
OUTPUT_PREFIX = "rq3"

COLUMN_NAMES = {
    "person_id": "person_id",
    "base_image_id": "base_image_id",
    "generated_image_id": "generated_image_id",
    "style": "style",
    "option_id": "option_id",
    "au": "au",
    "original_au_presence": "original_au_presence",
    "stylized_au_presence": "stylized_au_presence",
    "face_detected": "face_detected",
    # Optional boolean/0-1 column for the human-selected-candidate sensitivity run.
    "selected": "human_selected",
}

RANDOM_INTERCEPTS = ["person_id", "base_image_id", "generated_image_id", "au"]
VCP_PRIOR_SD = 0.5
FIXED_EFFECT_PRIOR_SD = 2.0
MAX_OPTIMIZER_ITERATIONS = 2000
NEAR_ZERO_RANDOM_SD = 0.05
AUTO_SIMPLIFY_AFTER_FAILED_FIT = True
POSTERIOR_DRAWS = 30000
RANDOM_SEED = 20260714


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the RQ3 binomial mixed-effects model to a long-format Py-Feat AU CSV."
        )
    )
    parser.add_argument("--input", type=Path, default=INPUT_CSV, help="Input CSV path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help="Directory for tables and figures",
    )
    parser.add_argument(
        "--output-prefix", default=OUTPUT_PREFIX, help="Filename prefix (default: rq3)"
    )
    parser.add_argument(
        "--candidate-mode",
        choices=["all", "selected"],
        default="all",
        help="Use all candidates or rows marked in the configured selected column",
    )
    parser.add_argument(
        "--selected-column",
        default=None,
        help="Source CSV column marking a human-selected candidate (overrides configuration)",
    )
    parser.add_argument(
        "--model-form",
        choices=["full", "reduced", "present-only", "absent-only"],
        default="full",
        help="Primary full interaction model or an optional sensitivity model",
    )
    parser.add_argument(
        "--au-specific",
        action="store_true",
        help="Also fit AU-specific clustered-logistic sensitivity models with BH correction",
    )
    return parser.parse_args()


def safe_prefix(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not cleaned:
        raise ValueError("output-prefix must contain at least one letter or number.")
    return cleaned


def model_specification(
    model_form: str, data: pd.DataFrame
) -> tuple[pd.DataFrame, str, list[int]]:
    if model_form == "full":
        return data, "agreement ~ style_ghibli * original_presence", [0, 1]
    if model_form == "reduced":
        return data, "agreement ~ style_ghibli + original_presence", [0, 1]
    original = 1 if model_form == "present-only" else 0
    subset = data.loc[data["original_presence"] == original].copy()
    if subset.empty:
        raise ValueError(f"No rows remain for model form {model_form!r}.")
    return subset, "agreement ~ style_ghibli", [original]


def prefit_checks(data: pd.DataFrame, model_form: str) -> None:
    if data["agreement"].nunique() < 2:
        raise ValueError(
            "Agreement has only one observed value; logistic regression is not estimable."
        )
    if data["style_ghibli"].nunique() < 2:
        raise ValueError(
            "Only one style remains; the Disney-Ghibli comparison is not estimable."
        )
    if model_form in {"full", "reduced"} and data["original_presence"].nunique() < 2:
        raise ValueError(
            "Original AU presence has only one value; the presence effect/interaction is not estimable."
        )
    if model_form == "full":
        cell_sizes = data.groupby(
            ["style_ghibli", "original_presence"], observed=True
        ).size()
        if len(cell_sizes) < 4:
            raise ValueError(
                "At least one style-by-original-presence cell is empty; the interaction is not estimable."
            )
        if int(cell_sizes.min()) < 5:
            warn_user(
                f"The smallest style-by-original-presence cell has only {int(cell_sizes.min())} rows; "
                "interaction estimates may be unstable."
            )


def main() -> None:
    args = parse_args()
    prefix = safe_prefix(args.output_prefix)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.selected_column:
        COLUMN_NAMES["selected"] = args.selected_column

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}. Use --input or edit INPUT_CSV near the top."
        )
    print(f"Reading {input_path}")
    raw = pd.read_csv(input_path, low_memory=False)
    data, cleaning_messages = prepare_analysis_data(
        raw, COLUMN_NAMES, candidate_mode=args.candidate_mode
    )
    analysis_data, formula, original_values = model_specification(args.model_form, data)
    prefit_checks(analysis_data, args.model_form)

    transition_counts, agreement_summary = transition_outputs(analysis_data)
    transition_counts.to_csv(
        output_dir / f"{prefix}_transition_counts.csv", index=False
    )
    agreement_summary.to_csv(
        output_dir / f"{prefix}_agreement_summary.csv", index=False
    )

    print("Fitting statsmodels BinomialBayesMixedGLM with variational Bayes...")
    fit = fit_bayesian_glmm(
        analysis_data,
        formula,
        RANDOM_INTERCEPTS,
        vcp_prior_sd=VCP_PRIOR_SD,
        fixed_prior_sd=FIXED_EFFECT_PRIOR_SD,
        maxiter=MAX_OPTIMIZER_ITERATIONS,
        near_zero_sd=NEAR_ZERO_RANDOM_SD,
        auto_simplify=AUTO_SIMPLIFY_AFTER_FAILED_FIT,
    )
    coefficients = fixed_effect_table(fit)
    random_effects = random_effect_table(fit, analysis_data)
    probabilities, contrasts = posterior_predictions_and_contrasts(
        fit,
        original_values=original_values,
        draws=POSTERIOR_DRAWS,
        random_seed=RANDOM_SEED,
    )
    diagnostics = model_diagnostics(fit, analysis_data, probabilities)
    interpretation = plain_language_interpretation(
        coefficients, probabilities, retained_random_effects=fit.random_effects
    )

    coefficients.to_csv(output_dir / f"{prefix}_model_coefficients.csv", index=False)
    random_effects.to_csv(output_dir / f"{prefix}_random_effects.csv", index=False)
    probabilities.to_csv(
        output_dir / f"{prefix}_estimated_probabilities.csv", index=False
    )
    contrasts.to_csv(output_dir / f"{prefix}_planned_contrasts.csv", index=False)

    plot_predicted_probabilities(
        probabilities, output_dir / f"{prefix}_predicted_probabilities"
    )
    plot_transition_distribution(
        transition_counts, output_dir / f"{prefix}_transition_categories"
    )

    if args.au_specific:
        cluster_column = "person_id"
        if cluster_column not in analysis_data.columns:
            for fallback in ["base_image_id", "generated_image_id"]:
                if fallback in analysis_data.columns:
                    cluster_column = fallback
                    break
        au_results = au_specific_clustered_models(
            analysis_data, cluster_column=cluster_column
        )
        au_results.to_csv(
            output_dir / f"{prefix}_au_specific_sensitivity.csv", index=False
        )

    method_text = (
        "METHOD\n"
        "statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM fitted with the "
        "mean-field variational-Bayes optimizer. This is a Bayesian logistic GLMM, not a "
        "frequentist lme4 fit. Fixed-effect intervals are approximate 95% posterior credible "
        "intervals. Predictions are conditional estimates with every random intercept set to "
        "zero. The interaction is evaluated by whether its 95% credible interval excludes zero; "
        "a likelihood-ratio test and frequentist p-value are not available from this fit."
    )
    model_text = (
        "MODEL SPECIFICATION\n"
        f"Formula: {formula}\n"
        f"Random intercepts retained: {', '.join(fit.random_effects)}\n"
        f"Random intercepts omitted: {', '.join(fit.removed_random_effects) or 'none'}\n"
        f"Fixed-effect prior SD: {FIXED_EFFECT_PRIOR_SD}\n"
        f"Log random-effect-SD prior SD: {VCP_PRIOR_SD}\n"
        f"Candidate mode: {args.candidate_mode}\n"
        f"Model form: {args.model_form}"
    )
    write_text(
        output_dir / f"{prefix}_model_summary.txt",
        [
            "RQ3 ORIGINAL-TO-STYLIZED PY-FEAT AU AGREEMENT",
            method_text,
            model_text,
            "STATSMODELS SUMMARY\n" + str(fit.result.summary()),
            "DATA-CLEANING MESSAGES\n" + "\n".join(cleaning_messages),
            "DIAGNOSTICS\n" + "\n".join(diagnostics),
            "PLAIN-LANGUAGE INTERPRETATION\n" + interpretation,
        ],
    )

    print(interpretation)
    print(f"Saved RQ3 outputs to {output_dir}")


if __name__ == "__main__":
    main()
