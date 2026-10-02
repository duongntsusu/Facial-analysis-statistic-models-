"""Reusable data, model, and reporting helpers for the RQ3 agreement analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import cohen_kappa_score
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.multitest import multipletests


TRANSITIONS = [
    (1, 1, "1 -> 1", "Preserved AU presence"),
    (1, 0, "1 -> 0", "AU lost"),
    (0, 1, "0 -> 1", "AU introduced"),
    (0, 0, "0 -> 0", "Preserved AU absence"),
]

FIXED_TERM_LABELS = {
    "Intercept": "Intercept: Disney, originally absent",
    "style_ghibli": "Ghibli vs Disney, originally absent",
    "original_presence": "Originally present vs absent, within Disney",
    "style_ghibli:original_presence": (
        "Interaction: change in present-vs-absent effect for Ghibli vs Disney"
    ),
}


def warn_user(message: str) -> None:
    """Print a visible, non-fatal analysis warning."""
    print(f"WARNING: {message}")


def _binary_value(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if float(value) in (0.0, 1.0) else np.nan
    normalized = str(value).strip().casefold()
    lookup = {
        "1": 1.0,
        "1.0": 1.0,
        "true": 1.0,
        "t": 1.0,
        "yes": 1.0,
        "y": 1.0,
        "present": 1.0,
        "detected": 1.0,
        "0": 0.0,
        "0.0": 0.0,
        "false": 0.0,
        "f": 0.0,
        "no": 0.0,
        "n": 0.0,
        "absent": 0.0,
        "not detected": 0.0,
    }
    return lookup.get(normalized, np.nan)


def coerce_binary(series: pd.Series, column_name: str) -> pd.Series:
    """Convert common binary representations to nullable 0/1 values."""
    converted = series.map(_binary_value).astype("Float64")
    bad = series.notna() & converted.isna()
    if bad.any():
        examples = ", ".join(map(repr, series.loc[bad].astype(str).unique()[:5]))
        warn_user(
            f"{int(bad.sum())} value(s) in {column_name!r} were not recognized as "
            f"binary and will be excluded. Examples: {examples}."
        )
    return converted


def _normalize_style(value: object) -> object:
    if pd.isna(value):
        return np.nan
    normalized = str(value).strip().casefold()
    if "disney" in normalized:
        return "Disney"
    if "ghibli" in normalized:
        return "Ghibli"
    return np.nan


def prepare_analysis_data(
    data: pd.DataFrame,
    column_map: Mapping[str, str],
    candidate_mode: str = "all",
) -> tuple[pd.DataFrame, list[str]]:
    """Rename, lightly clean, filter, and derive RQ3 variables."""
    messages: list[str] = []

    def note(message: str) -> None:
        messages.append(message)
        warn_user(message)

    rename = {
        source: canonical
        for canonical, source in column_map.items()
        if source and source in data.columns and canonical != source
    }
    clean = data.rename(columns=rename).copy()
    required = ["style", "original_au_presence", "stylized_au_presence"]
    missing = [column for column in required if column not in clean.columns]
    if missing:
        configured = {name: column_map.get(name) for name in missing}
        raise ValueError(
            "Missing essential column(s): "
            + ", ".join(missing)
            + f". Current source-column settings: {configured}"
        )

    starting_rows = len(clean)
    if "face_detected" in clean.columns:
        detected = coerce_binary(clean["face_detected"], "face_detected")
        keep = detected.eq(1).fillna(False)
        removed = int((~keep).sum())
        clean = clean.loc[keep].copy()
        note(
            f"Filtered to successfully detected stylized faces: retained {len(clean):,} "
            f"of {starting_rows:,} rows ({removed:,} removed)."
        )
    else:
        note(
            "No face_detected column was found; continuing without a detection filter."
        )

    if candidate_mode == "selected":
        if "selected" not in clean.columns:
            raise ValueError(
                "candidate_mode='selected' requires the configured human-selection column. "
                "Set COLUMN_NAMES['selected'] or --selected-column."
            )
        selected = coerce_binary(clean["selected"], "selected")
        keep = selected.eq(1).fillna(False)
        note(
            f"Human-selected-candidate sensitivity filter retained {int(keep.sum()):,} "
            f"of {len(clean):,} detected rows."
        )
        clean = clean.loc[keep].copy()
        selection_keys = [
            column for column in ["base_image_id", "style"] if column in clean.columns
        ]
        if len(selection_keys) == 2 and "generated_image_id" in clean.columns:
            selected_per_cell = clean.groupby(selection_keys, dropna=False)[
                "generated_image_id"
            ].nunique()
            unexpected = selected_per_cell.ne(1)
            if unexpected.any():
                note(
                    f"{int(unexpected.sum())} base-image/style cell(s) contain other than one "
                    "selected generated image. The script will continue, but this is not a "
                    "one-candidate-per-cell sensitivity dataset."
                )

    clean["original_presence"] = coerce_binary(
        clean["original_au_presence"], "original_au_presence"
    )
    clean["stylized_presence"] = coerce_binary(
        clean["stylized_au_presence"], "stylized_au_presence"
    )
    before_binary_drop = len(clean)
    clean = clean.dropna(subset=["original_presence", "stylized_presence"]).copy()
    if len(clean) < before_binary_drop:
        note(
            f"Dropped {before_binary_drop - len(clean):,} row(s) with missing or invalid "
            "original/stylized AU presence values."
        )
    clean["original_presence"] = clean["original_presence"].astype(int)
    clean["stylized_presence"] = clean["stylized_presence"].astype(int)

    raw_style = clean["style"].copy()
    clean["style"] = raw_style.map(_normalize_style)
    invalid_style = clean["style"].isna()
    if invalid_style.any():
        examples = ", ".join(
            map(repr, raw_style.loc[invalid_style].astype(str).unique()[:5])
        )
        note(
            f"Dropped {int(invalid_style.sum()):,} row(s) whose style was not recognized as "
            f"Disney or Ghibli. Examples: {examples}."
        )
        clean = clean.loc[~invalid_style].copy()

    clean["style"] = pd.Categorical(clean["style"], categories=["Disney", "Ghibli"])
    clean["style_ghibli"] = clean["style"].eq("Ghibli").astype(int)
    clean["agreement"] = (
        clean["original_presence"].eq(clean["stylized_presence"]).astype(int)
    )

    for column in [
        "person_id",
        "base_image_id",
        "generated_image_id",
        "au",
        "option_id",
    ]:
        if column not in clean.columns:
            note(f"Optional column {column!r} was not found.")
            continue
        missing_ids = clean[column].isna()
        if missing_ids.any():
            note(
                f"Filled {int(missing_ids.sum()):,} missing {column} value(s) with a shared "
                "'__missing__' level. Review these rows if missingness is substantial."
            )
        clean[column] = clean[column].where(~missing_ids, "__missing__").astype(str)

    if clean.empty:
        raise ValueError("No usable rows remain after cleaning and requested filters.")
    return clean.reset_index(drop=True), messages


def transition_outputs(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build complete transition counts and style-level agreement summaries."""
    count_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for style in ["Disney", "Ghibli"]:
        subset = data.loc[data["style"] == style]
        counts: dict[tuple[int, int], int] = {}
        for original, stylized, transition, interpretation in TRANSITIONS:
            count = int(
                (
                    (subset["original_presence"] == original)
                    & (subset["stylized_presence"] == stylized)
                ).sum()
            )
            counts[(original, stylized)] = count
            count_rows.append(
                {
                    "style": style,
                    "transition": transition,
                    "original_au_presence": original,
                    "stylized_au_presence": stylized,
                    "interpretation": interpretation,
                    "count": count,
                    "proportion_within_style": (
                        count / len(subset) if len(subset) else np.nan
                    ),
                }
            )
        n11, n10 = counts[(1, 1)], counts[(1, 0)]
        n01, n00 = counts[(0, 1)], counts[(0, 0)]

        def ratio(numerator: int, denominator: int) -> float:
            return numerator / denominator if denominator else np.nan

        kappa = np.nan
        if len(subset) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kappa = float(
                    cohen_kappa_score(
                        subset["original_presence"], subset["stylized_presence"]
                    )
                )
        summary_rows.append(
            {
                "style": style,
                "n_au_observations": len(subset),
                "n_11_preserved_presence": n11,
                "n_10_lost": n10,
                "n_01_introduced": n01,
                "n_00_preserved_absence": n00,
                "overall_agreement": ratio(n11 + n00, n11 + n10 + n01 + n00),
                "preservation_rate": ratio(n11, n11 + n10),
                "preserved_absence_rate": ratio(n00, n00 + n01),
                "loss_rate": ratio(n10, n11 + n10),
                "introduction_rate": ratio(n01, n00 + n01),
                "cohens_kappa": kappa,
            }
        )
    return pd.DataFrame(count_rows), pd.DataFrame(summary_rows)


@dataclass
class MixedModelFit:
    model: BinomialBayesMixedGLM
    result: object
    formula: str
    random_effects: list[str]
    removed_random_effects: list[str]
    optimizer_messages: list[str]


def _optimizer_succeeded(result: object) -> bool:
    retvals = getattr(result, "optim_retvals", {})
    return bool(retvals.get("success", False))


def fit_bayesian_glmm(
    data: pd.DataFrame,
    formula: str,
    random_effects: Sequence[str],
    *,
    vcp_prior_sd: float = 0.5,
    fixed_prior_sd: float = 2.0,
    maxiter: int = 2000,
    near_zero_sd: float = 0.05,
    auto_simplify: bool = True,
) -> MixedModelFit:
    """Fit the requested Bayesian logistic GLMM, simplifying only after failed fitting."""
    usable: list[str] = []
    omitted: list[str] = []
    for column in random_effects:
        if column not in data.columns:
            warn_user(
                f"Random intercept {column!r} is unavailable and will be omitted."
            )
            omitted.append(column)
        elif data[column].nunique(dropna=False) < 2:
            warn_user(
                f"Random intercept {column!r} has fewer than two levels and will be omitted."
            )
            omitted.append(column)
        else:
            usable.append(column)
    if not usable:
        raise ValueError(
            "No requested grouping variable has at least two usable levels."
        )

    optimizer_messages: list[str] = []
    active = usable.copy()
    removal_priority = ["person_id", "base_image_id", "generated_image_id", "au"]

    while active:
        variance_formulas = {column: f"0 + C({column})" for column in active}
        try:
            model = BinomialBayesMixedGLM.from_formula(
                formula,
                variance_formulas,
                data,
                vcp_p=vcp_prior_sd,
                fe_p=fixed_prior_sd,
            )
            fit_results: list[object] = []
            for method in ["BFGS", "L-BFGS-B"]:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = model.fit_vb(
                        fit_method=method,
                        minim_opts={"maxiter": maxiter},
                        scale_fe=False,
                    )
                for item in caught:
                    optimizer_messages.append(f"{method}: {item.message}")
                fit_results.append(result)
                retvals = getattr(result, "optim_retvals", {})
                optimizer_messages.append(
                    f"{method}: success={retvals.get('success')}; "
                    f"message={retvals.get('message', 'not supplied')}"
                )
                if _optimizer_succeeded(result):
                    return MixedModelFit(
                        model=model,
                        result=result,
                        formula=formula,
                        random_effects=active,
                        removed_random_effects=omitted,
                        optimizer_messages=optimizer_messages,
                    )

            latest = fit_results[-1]
            sd_estimates = dict(zip(active, np.exp(latest.vcp_mean)))
            removable = [name for name in active if sd_estimates[name] < near_zero_sd]
            if not auto_simplify or not removable:
                raise RuntimeError(
                    "The full random-effects model did not converge, and no random-effect "
                    f"SD was below the configured near-zero threshold ({near_zero_sd})."
                )
            to_remove = next(
                (name for name in removal_priority if name in removable), removable[0]
            )
            active.remove(to_remove)
            omitted.append(to_remove)
            message = (
                f"The fit failed to converge and {to_remove!r} had a near-zero posterior "
                f"median SD ({sd_estimates[to_remove]:.4f}); refitting without it."
            )
            optimizer_messages.append(message)
            warn_user(message)
        except (MemoryError, KeyboardInterrupt):
            raise
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "did not converge" in str(exc):
                raise
            raise RuntimeError(
                "The mixed model could not reasonably be fit with random intercepts for "
                + ", ".join(active)
                + f". Original error: {type(exc).__name__}: {exc}"
            ) from exc

    raise RuntimeError("The model could not be fit with any requested random effects.")


def fixed_effect_table(fit: MixedModelFit, level: float = 0.95) -> pd.DataFrame:
    """Return posterior fixed-effect summaries and odds ratios."""
    alpha = 1 - level
    critical = norm.ppf(1 - alpha / 2)
    mean = np.asarray(fit.result.fe_mean)
    sd = np.asarray(fit.result.fe_sd)
    lower, upper = mean - critical * sd, mean + critical * sd
    return pd.DataFrame(
        {
            "term": fit.model.fep_names,
            "interpretation": [
                FIXED_TERM_LABELS.get(x, x) for x in fit.model.fep_names
            ],
            "posterior_mean_log_odds": mean,
            "posterior_sd": sd,
            "credible_interval_lower": lower,
            "credible_interval_upper": upper,
            "odds_ratio": np.exp(np.clip(mean, -700, 700)),
            "odds_ratio_interval_lower": np.exp(np.clip(lower, -700, 700)),
            "odds_ratio_interval_upper": np.exp(np.clip(upper, -700, 700)),
            "posterior_probability_positive": norm.cdf(mean / sd),
            "interval_level": level,
        }
    )


def random_effect_table(
    fit: MixedModelFit,
    data: pd.DataFrame,
    level: float = 0.95,
) -> pd.DataFrame:
    """Summarize random-intercept SDs/variances and grouping sizes."""
    critical = norm.ppf(1 - (1 - level) / 2)
    rows: list[dict[str, object]] = []
    for name, mean, sd in zip(
        fit.model.vcp_names, fit.result.vcp_mean, fit.result.vcp_sd
    ):
        lower_log, upper_log = mean - critical * sd, mean + critical * sd
        group_sizes = data.groupby(name, observed=True).size()
        rows.append(
            {
                "grouping_variable": name,
                "n_levels": int(group_sizes.size),
                "minimum_rows_per_level": int(group_sizes.min()),
                "median_rows_per_level": float(group_sizes.median()),
                "maximum_rows_per_level": int(group_sizes.max()),
                "posterior_mean_log_sd": mean,
                "posterior_sd_log_sd": sd,
                "posterior_median_sd": np.exp(mean),
                "sd_interval_lower": np.exp(lower_log),
                "sd_interval_upper": np.exp(upper_log),
                "posterior_median_variance": np.exp(2 * mean),
                "variance_interval_lower": np.exp(2 * lower_log),
                "variance_interval_upper": np.exp(2 * upper_log),
                "near_zero_sd_below_0_05": bool(np.exp(mean) < 0.05),
                "interval_level": level,
            }
        )
    return pd.DataFrame(rows)


def _scenario_design(
    fixed_names: Sequence[str], style_ghibli: int, original_presence: int
) -> np.ndarray:
    values = {
        "Intercept": 1.0,
        "style_ghibli": float(style_ghibli),
        "original_presence": float(original_presence),
        "style_ghibli:original_presence": float(style_ghibli * original_presence),
    }
    missing = [name for name in fixed_names if name not in values]
    if missing:
        raise ValueError(f"Prediction code does not recognize fixed term(s): {missing}")
    return np.asarray([values[name] for name in fixed_names], dtype=float)


def posterior_predictions_and_contrasts(
    fit: MixedModelFit,
    original_values: Sequence[int] = (0, 1),
    *,
    draws: int = 30000,
    random_seed: int = 20260714,
    level: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conditional (random intercepts = 0) probabilities and planned contrasts."""
    rng = np.random.default_rng(random_seed)
    means = np.asarray(fit.result.fe_mean)
    sds = np.asarray(fit.result.fe_sd)
    beta_draws = rng.normal(means, sds, size=(draws, len(means)))
    alpha = 1 - level
    probability_draws: dict[tuple[str, int], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for style, style_value in [("Disney", 0), ("Ghibli", 1)]:
        for original in original_values:
            design = _scenario_design(fit.model.fep_names, style_value, original)
            draws_for_cell = expit(beta_draws @ design)
            probability_draws[(style, original)] = draws_for_cell
            point = float(expit(design @ means))
            rows.append(
                {
                    "style": style,
                    "original_au_presence": int(original),
                    "original_au_label": "Present" if original else "Absent",
                    "estimated_probability": point,
                    "posterior_mean_probability": float(draws_for_cell.mean()),
                    "credible_interval_lower": float(
                        np.quantile(draws_for_cell, alpha / 2)
                    ),
                    "credible_interval_upper": float(
                        np.quantile(draws_for_cell, 1 - alpha / 2)
                    ),
                    "prediction_scale": "conditional; all random intercepts set to zero",
                    "interval_level": level,
                }
            )

    contrast_specs: list[tuple[str, tuple[str, int], tuple[str, int]]] = []
    if 0 in original_values:
        contrast_specs.append(
            (
                "Disney minus Ghibli among originally absent AUs",
                ("Disney", 0),
                ("Ghibli", 0),
            )
        )
    if 1 in original_values:
        contrast_specs.append(
            (
                "Disney minus Ghibli among originally present AUs",
                ("Disney", 1),
                ("Ghibli", 1),
            )
        )
    if set(original_values) == {0, 1}:
        contrast_specs.extend(
            [
                ("Present minus absent within Disney", ("Disney", 1), ("Disney", 0)),
                ("Present minus absent within Ghibli", ("Ghibli", 1), ("Ghibli", 0)),
            ]
        )
    contrast_rows: list[dict[str, object]] = []
    for label, first, second in contrast_specs:
        difference = probability_draws[first] - probability_draws[second]
        contrast_rows.append(
            {
                "contrast": label,
                "scale": "probability difference (first minus second)",
                "posterior_mean_difference": float(difference.mean()),
                "credible_interval_lower": float(np.quantile(difference, alpha / 2)),
                "credible_interval_upper": float(
                    np.quantile(difference, 1 - alpha / 2)
                ),
                "posterior_probability_greater_than_zero": float(
                    (difference > 0).mean()
                ),
                "interval_level": level,
                "multiplicity_adjustment": (
                    "Not applicable: Bayesian posterior contrasts; no frequentist p-values"
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(contrast_rows)


def model_diagnostics(
    fit: MixedModelFit,
    data: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> list[str]:
    """Return practical GLMM diagnostics as human-readable lines."""
    lines: list[str] = []
    retvals = getattr(fit.result, "optim_retvals", {})
    lines.append(f"Optimizer convergence flag: {bool(retvals.get('success', False))}")
    lines.append(f"Optimizer message: {retvals.get('message', 'not supplied')}")
    lines.append(f"Rows analyzed: {len(data):,}")
    lines.append(f"Agreement=1 rows: {int(data['agreement'].sum()):,}")
    lines.append(f"Agreement=0 rows: {int((1 - data['agreement']).sum()):,}")
    for group in fit.random_effects:
        sizes = data.groupby(group, observed=True).size()
        lines.append(
            f"{group}: {sizes.size:,} levels; rows/level min={sizes.min()}, "
            f"median={sizes.median():.1f}, max={sizes.max()}"
        )

    linear = np.asarray(fit.model.exog @ fit.result.fe_mean).ravel()
    linear += np.asarray(fit.model.exog_vc @ fit.result.vc_mean).ravel()
    fitted = expit(linear)
    denominator = np.maximum(fitted * (1 - fitted), np.finfo(float).eps)
    pearson = float(np.sum((data["agreement"].to_numpy() - fitted) ** 2 / denominator))
    df = max(len(data) - len(fit.result.fe_mean), 1)
    lines.append(
        f"Heuristic conditional Pearson dispersion ratio: {pearson / df:.3f}. "
        "For Bernoulli mixed models this is descriptive, not a definitive test."
    )
    extreme = int(((fitted < 0.01) | (fitted > 0.99)).sum())
    lines.append(
        f"Extreme conditional fitted probabilities (<0.01 or >0.99): {extreme:,} "
        f"of {len(fitted):,}."
    )
    scenario_extreme = int(
        (
            (probabilities["estimated_probability"] < 0.01)
            | (probabilities["estimated_probability"] > 0.99)
        ).sum()
    )
    lines.append(f"Extreme fixed-effect scenario probabilities: {scenario_extreme}.")

    for (style, original), subset in data.groupby(
        ["style", "original_presence"], observed=True
    ):
        successes = int(subset["agreement"].sum())
        lines.append(
            f"Cell {style}, originally {'present' if original else 'absent'}: "
            f"n={len(subset):,}, agreement=1 count={successes:,}, "
            f"agreement=0 count={len(subset) - successes:,}."
        )
        if successes in (0, len(subset)):
            lines.append(
                "  Separation warning: this fixed-effect cell has only one outcome value."
            )
    if fit.removed_random_effects:
        lines.append(
            "Random effects omitted from the final fit: "
            + ", ".join(fit.removed_random_effects)
        )
    else:
        lines.append("All four requested random intercepts were retained.")
    lines.extend(f"Fit log: {message}" for message in fit.optimizer_messages)
    return lines


def plain_language_interpretation(
    coefficients: pd.DataFrame,
    probabilities: pd.DataFrame,
    retained_random_effects: Sequence[str] | None = None,
) -> str:
    """Generate a short interpretation tied to the fitted estimates."""

    def probability(style: str, original: int) -> float | None:
        match = probabilities.loc[
            (probabilities["style"] == style)
            & (probabilities["original_au_presence"] == original),
            "estimated_probability",
        ]
        return float(match.iloc[0]) if len(match) else None

    sentences: list[str] = []
    requested = ["person_id", "base_image_id", "generated_image_id", "au"]
    retained = list(retained_random_effects or [])
    all_requested_retained = all(name in retained for name in requested)
    if all_requested_retained:
        clustering_phrase = (
            "repeated observations from the same individuals, source images, generated "
            "images, and Action Units"
        )
    elif retained:
        clustering_phrase = "clustering represented by " + ", ".join(retained)
    else:
        clustering_phrase = "the fixed effects in the fitted model"
    disney_present, ghibli_present = probability("Disney", 1), probability("Ghibli", 1)
    disney_absent, ghibli_absent = probability("Disney", 0), probability("Ghibli", 0)
    if disney_present is not None and ghibli_present is not None:
        sentences.append(
            f"After accounting for {clustering_phrase}, the estimated conditional probability "
            f"of agreement was {disney_present:.3f} for Disney and {ghibli_present:.3f} for "
            "Ghibli when the AU was originally present."
        )
    if disney_absent is not None and ghibli_absent is not None:
        prefix = (
            "When"
            if disney_present is not None
            else f"After accounting for {clustering_phrase}, when"
        )
        sentences.append(
            f"{prefix} the AU was originally absent, the corresponding estimated conditional "
            f"probabilities were {disney_absent:.3f} for Disney and {ghibli_absent:.3f} for "
            "Ghibli."
        )
    interaction = coefficients.loc[
        coefficients["term"] == "style_ghibli:original_presence"
    ]
    if len(interaction):
        row = interaction.iloc[0]
        supported = (
            row["credible_interval_lower"] > 0 or row["credible_interval_upper"] < 0
        )
        if supported:
            sentences.append(
                "The 95% credible interval for the interaction excluded zero, supporting the "
                "conclusion that the difference between Disney and Ghibli depended on whether "
                "the AU was originally present or absent."
            )
        else:
            sentences.append(
                "The 95% credible interval for the interaction included zero, so there was "
                "insufficient evidence that the Disney-Ghibli difference changed according to "
                "original AU presence."
            )
    sentences.append(
        "Agreement here means agreement with the original-image Py-Feat estimate; it is not a "
        "measure of ground-truth facial-expression accuracy."
    )
    return " ".join(sentences)


def plot_predicted_probabilities(
    probabilities: pd.DataFrame, output_base: Path
) -> None:
    """Plot conditional agreement probabilities with credible intervals."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x_lookup = {"Absent": 0, "Present": 1}
    colors = {"Disney": "#386CB0", "Ghibli": "#F28E2B"}
    offsets = {"Disney": -0.04, "Ghibli": 0.04}
    for style in ["Disney", "Ghibli"]:
        subset = probabilities.loc[probabilities["style"] == style].copy()
        subset["x"] = subset["original_au_label"].map(x_lookup).astype(float)
        subset = subset.sort_values("x")
        x = subset["x"].to_numpy() + offsets[style]
        y = subset["estimated_probability"].to_numpy()
        yerr = np.vstack(
            [
                y - subset["credible_interval_lower"].to_numpy(),
                subset["credible_interval_upper"].to_numpy() - y,
            ]
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=4,
            label=style,
            color=colors[style],
        )
    present_labels = [
        label
        for label in ["Absent", "Present"]
        if label in probabilities["original_au_label"].tolist()
    ]
    ticks = [x_lookup[label] for label in present_labels]
    ax.set_xticks(ticks, present_labels)
    ax.set_xlim(min(ticks, default=0) - 0.3, max(ticks, default=1) + 0.3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Original-image Py-Feat AU presence")
    ax.set_ylabel("Estimated probability of agreement")
    ax.set_title("RQ3: Model-estimated AU agreement")
    ax.legend(title="Stylization style", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(
            output_base.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


def plot_transition_distribution(counts: pd.DataFrame, output_base: Path) -> None:
    """Plot within-style percentages for all four transition categories."""
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    labels = [item[3] for item in TRANSITIONS]
    x = np.arange(len(labels))
    width = 0.36
    colors = {"Disney": "#386CB0", "Ghibli": "#F28E2B"}
    for index, style in enumerate(["Disney", "Ghibli"]):
        style_rows = counts.loc[counts["style"] == style].set_index("interpretation")
        values = style_rows.reindex(labels)["proportion_within_style"].to_numpy() * 100
        ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=style,
            color=colors[style],
        )
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Percentage of AU observations within style")
    ax.set_title("RQ3: Original-to-stylized AU transitions")
    ax.legend(title="Stylization style", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(
            output_base.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


def au_specific_clustered_models(
    data: pd.DataFrame,
    cluster_column: str = "person_id",
    minimum_rows: int = 30,
) -> pd.DataFrame:
    """Optional AU-specific logistic GLMs with person-clustered standard errors.

    This is a sensitivity analysis, not a replacement for the primary GLMM. It is used
    because each AU-specific fit cannot retain an AU random intercept and a collection of
    many Bayesian interaction intervals does not produce frequentist p-values for BH control.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    if "au" not in data.columns:
        raise ValueError("AU-specific models require the configured au column.")
    if cluster_column not in data.columns:
        raise ValueError(f"Cluster column {cluster_column!r} is unavailable.")
    rows: list[dict[str, object]] = []
    for au, subset in data.groupby("au", observed=True):
        row: dict[str, object] = {
            "au": au,
            "n": len(subset),
            "cluster_variable": cluster_column,
            "n_clusters": subset[cluster_column].nunique(),
            "method": "logistic GLM with person-clustered sandwich SE (sensitivity approximation)",
        }
        if (
            len(subset) < minimum_rows
            or subset[cluster_column].nunique() < 2
            or subset["style_ghibli"].nunique() < 2
            or subset["original_presence"].nunique() < 2
            or subset["agreement"].nunique() < 2
        ):
            row["status"] = "skipped: insufficient variation, rows, or clusters"
            rows.append(row)
            continue
        try:
            result = smf.glm(
                "agreement ~ style_ghibli * original_presence",
                data=subset,
                family=sm.families.Binomial(),
            ).fit(cov_type="cluster", cov_kwds={"groups": subset[cluster_column]})
            term = "style_ghibli:original_presence"
            estimate = float(result.params[term])
            se = float(result.bse[term])
            row.update(
                {
                    "status": "fit",
                    "interaction_log_odds": estimate,
                    "interaction_clustered_se": se,
                    "interaction_z": float(result.tvalues[term]),
                    "interaction_p_value": float(result.pvalues[term]),
                    "interaction_odds_ratio": float(np.exp(estimate)),
                    "confidence_interval_lower": float(result.conf_int().loc[term, 0]),
                    "confidence_interval_upper": float(result.conf_int().loc[term, 1]),
                }
            )
        except Exception as exc:
            row["status"] = f"failed: {type(exc).__name__}: {exc}"
        rows.append(row)
    output = pd.DataFrame(rows)
    valid = (
        output["interaction_p_value"].notna() if "interaction_p_value" in output else []
    )
    output["interaction_p_value_bh"] = np.nan
    output["interaction_reject_bh_0_05"] = False
    if len(output) and np.any(valid):
        rejected, adjusted, _, _ = multipletests(
            output.loc[valid, "interaction_p_value"], alpha=0.05, method="fdr_bh"
        )
        output.loc[valid, "interaction_p_value_bh"] = adjusted
        output.loc[valid, "interaction_reject_bh_0_05"] = rejected
    return output


def write_text(path: Path, sections: Iterable[str]) -> None:
    path.write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
