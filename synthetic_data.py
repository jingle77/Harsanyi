# synthetic_data.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid


# --- Feature definitions -----------------------------------------------------

# Channels
CHANNEL_FEATURES: List[str] = [
    "email",
    "seo",
    "sem",
    "direct",
    "display",
    "social",
    "affiliate",
]

# Pages
PAGE_FEATURES: List[str] = [
    "product_page_a",
    "product_page_b",
    "product_page_c",
    "product_page_d",
    "product_page_e",
    "product_page_f",
    "deals_page",
    "search_page",
    "homepage",
    "account_page",
    "support_page",
]

# Device (mutually exclusive per session)
DEVICE_FEATURES: List[str] = [
    "device_desktop",
    "device_mobile",
]

ALL_BINARY_FEATURES: List[str] = CHANNEL_FEATURES + PAGE_FEATURES + DEVICE_FEATURES


# --- Propensity scores (1–10 scale) -----------------------------------------
# These encode the "propensity to convert" intuition per feature. Some
# features have a range (e.g., 5–7) to allow small variation per run.

FEATURE_PROPENSITY_RANGES: Dict[str, Tuple[float, float]] = {
    # Channels
    "email": (2.0, 2.0),
    "seo": (6.0, 6.0),
    "sem": (6.0, 6.0),
    "direct": (5.0, 5.0),
    "display": (1.0, 1.0),
    "social": (1.0, 1.0),
    "affiliate": (7.0, 7.0),
    # Pages (A/F with ranges where specified)
    "product_page_a": (5.0, 7.0),
    "product_page_b": (4.0, 8.0),
    "product_page_c": (5.0, 7.0),
    "product_page_d": (4.0, 8.0),
    "product_page_e": (5.0, 7.0),
    "product_page_f": (4.0, 8.0),
    "deals_page": (6.0, 6.0),
    "search_page": (5.0, 5.0),
    "homepage": (4.0, 4.0),
    "account_page": (7.0, 7.0),
    "support_page": (3.0, 3.0),
    # Device
    "device_desktop": (6.0, 6.0),
    "device_mobile": (3.0, 3.0),
}


# --- Helper dataclasses ------------------------------------------------------


@dataclass
class LogisticSpec:
    intercept: float
    main_effects: Dict[str, float]
    interactions_2: Dict[Tuple[str, str], float]
    interactions_3: Dict[Tuple[str, str, str], float]


# --- Internal helpers --------------------------------------------------------


def _sample_feature_scores(
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Sample an effective "propensity score" for each feature from the
    provided 1–10 ranges. For single-value ranges, we just use that value.
    """
    scores: Dict[str, float] = {}
    for feature, (lo, hi) in FEATURE_PROPENSITY_RANGES.items():
        if lo == hi:
            scores[feature] = float(lo)
        else:
            scores[feature] = float(rng.uniform(lo, hi))
    return scores


def _coef_range_for_score(score: float) -> Tuple[float, float]:
    """
    Map a 1–10 propensity score into a coefficient range for the logistic model.

    - 1–2  -> negative effect
    - 3–4  -> near neutral
    - 5–6  -> moderate positive
    - 7–8  -> strong positive
    - 9–10 -> very strong positive
    """
    if score <= 2.0:
        return (-1.0, -0.3)
    elif score <= 4.0:
        return (-0.3, 0.3)
    elif score <= 6.0:
        return (0.3, 1.0)
    elif score <= 8.0:
        return (1.0, 2.5)
    else:
        return (2.5, 4.0)


def _sample_marginal_probabilities(
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], float]:
    """
    Sample marginal activation probabilities for channels and pages, plus
    an overall desktop share (mobile share = 1 - desktop_share).

    Returns
    -------
    probs: Dict[str, float]
        Marginal probabilities for CHANNEL_FEATURES and PAGE_FEATURES.
        Device features are handled separately.
    desktop_share: float
        Fraction of sessions that are desktop (mobile is complementary).
    """
    probs: Dict[str, float] = {}

    # Channels – fairly sparse, some more common (SEO, Direct)
    probs["email"] = rng.uniform(0.03, 0.15)
    probs["seo"] = rng.uniform(0.10, 0.60)
    probs["sem"] = rng.uniform(0.05, 0.40)
    probs["direct"] = rng.uniform(0.10, 0.50)
    probs["display"] = rng.uniform(0.01, 0.10)
    probs["social"] = rng.uniform(0.03, 0.20)
    probs["affiliate"] = rng.uniform(0.02, 0.15)

    # Product pages – moderate likelihood
    probs["product_page_a"] = rng.uniform(0.20, 0.60)
    probs["product_page_b"] = rng.uniform(0.10, 0.50)
    probs["product_page_c"] = rng.uniform(0.20, 0.60)
    probs["product_page_d"] = rng.uniform(0.10, 0.50)
    probs["product_page_e"] = rng.uniform(0.20, 0.60)
    probs["product_page_f"] = rng.uniform(0.10, 0.50)

    # Other pages
    probs["deals_page"] = rng.uniform(0.10, 0.40)
    probs["search_page"] = rng.uniform(0.15, 0.50)
    probs["homepage"] = rng.uniform(0.50, 0.95)
    probs["account_page"] = rng.uniform(0.05, 0.30)
    probs["support_page"] = rng.uniform(0.01, 0.10)

    # Device split – realistic desktop vs mobile share
    desktop_share = rng.uniform(0.40, 0.70)

    return probs, desktop_share


def _build_logistic_spec(
    rng: np.random.Generator,
) -> LogisticSpec:
    """
    Build a random-but-sensible logistic model specification based on
    the 1–10 propensity scores.
    """
    scores = _sample_feature_scores(rng)

    # Main effects
    main_effects: Dict[str, float] = {}
    for feature in ALL_BINARY_FEATURES:
        score = scores[feature]
        lo, hi = _coef_range_for_score(score)
        main_effects[feature] = rng.uniform(lo, hi)

    interactions_2: Dict[Tuple[str, str], float] = {}
    interactions_3: Dict[Tuple[str, str, str], float] = {}

    strong_2 = (1.0, 3.0)
    moderate_2 = (0.5, 1.5)
    weak_2 = (-0.3, 0.3)
    strong_3 = (1.5, 3.5)
    moderate_3 = (0.7, 2.0)

    def add_interaction_2(a: str, b: str, coef_range: Tuple[float, float]) -> None:
        key = tuple(sorted((a, b)))
        interactions_2[key] = rng.uniform(*coef_range)

    def add_interaction_3(a: str, b: str, c: str, coef_range: Tuple[float, float]) -> None:
        key = tuple(sorted((a, b, c)))
        interactions_3[key] = rng.uniform(*coef_range)

    # 2-way interactions (intuitive combos)
    add_interaction_2("email", "deals_page", strong_2)
    add_interaction_2("email", "account_page", moderate_2)

    add_interaction_2("seo", "search_page", strong_2)
    add_interaction_2("sem", "search_page", strong_2)
    add_interaction_2("sem", "product_page_a", moderate_2)
    add_interaction_2("sem", "product_page_b", moderate_2)

    add_interaction_2("affiliate", "deals_page", strong_2)
    add_interaction_2("direct", "homepage", moderate_2)
    add_interaction_2("social", "product_page_b", moderate_2)

    add_interaction_2("device_desktop", "account_page", moderate_2)
    add_interaction_2("device_mobile", "deals_page", moderate_2)

    # A negative interaction: display -> support (frustrated users)
    add_interaction_2("display", "support_page", (-1.5, -0.3))

    # 3-way interactions
    add_interaction_3("sem", "product_page_a", "deals_page", strong_3)
    add_interaction_3("seo", "product_page_c", "search_page", moderate_3)

    # Start intercept near logit(0.05); calibration will refine it.
    intercept = float(np.log(0.05 / (1.0 - 0.05)))

    return LogisticSpec(
        intercept=intercept,
        main_effects=main_effects,
        interactions_2=interactions_2,
        interactions_3=interactions_3,
    )


def _compute_linear_predictor(
    df: pd.DataFrame,
    spec: LogisticSpec,
) -> np.ndarray:
    """
    Compute the linear predictor:

        z = β0 + Σ β_i x_i + Σ β_ij x_i x_j + Σ β_ijk x_i x_j x_k
    """
    z = np.full(shape=len(df), fill_value=spec.intercept, dtype=float)

    # Main effects
    for f, beta in spec.main_effects.items():
        if f in df.columns:
            z += beta * df[f].values

    # 2-way interactions
    for (a, b), beta in spec.interactions_2.items():
        if a in df.columns and b in df.columns:
            z += beta * (df[a].values * df[b].values)

    # 3-way interactions
    for (a, b, c), beta in spec.interactions_3.items():
        if a in df.columns and b in df.columns and c in df.columns:
            z += beta * (df[a].values * df[b].values * df[c].values)

    return z


def _calibrate_intercept_to_global_rate(
    df: pd.DataFrame,
    spec: LogisticSpec,
    target_rate: float = 0.05,
    max_iter: int = 8,
) -> LogisticSpec:
    """
    Adjust the intercept so that the average predicted conversion rate
    is close to target_rate (~5%).

    Uses a simple fixed-point update on the intercept:

        Δβ0 ≈ log(target_odds / current_odds)
    """
    for _ in range(max_iter):
        z = _compute_linear_predictor(df, spec)
        p = expit(z)
        mean_p = float(p.mean())
        if mean_p <= 0 or mean_p >= 1:
            break  # degenerate; bail out

        current_odds = mean_p / (1.0 - mean_p)
        target_odds = target_rate / (1.0 - target_rate)
        delta = np.log(target_odds / current_odds)

        spec.intercept += float(delta)

        if abs(mean_p - target_rate) < 0.002:
            break

    return spec


# --- Public API --------------------------------------------------------------


def generate_synthetic_ecommerce_data(
    n_rows: int = 100_000,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic ecommerce-style session data with:

    - Binary features for channels, pages, and device.
    - A binary 'converted' target.

    Steps:
        1. Sample feature-wise marginal probabilities for channels and pages.
        2. Sample a realistic desktop vs mobile share (device features are
           mutually exclusive per row).
        3. Generate binary features as Bernoulli draws.
        4. Build a logistic model with:
            - Random main effects informed by 1–10 propensity scores.
            - A few explicit 2-way and 3-way interactions.
            - An intercept calibrated to yield ~5% global conversion rate.
        5. Draw conversions Y ~ Bernoulli(p) and append as 'converted'.
    """
    rng = np.random.default_rng(random_state)

    # 1) Marginals for channels/pages and overall device split
    probs, desktop_share = _sample_marginal_probabilities(rng)

    # 2) Generate features
    data: Dict[str, np.ndarray] = {}

    # Channels and pages
    for f in CHANNEL_FEATURES + PAGE_FEATURES:
        p = probs[f]
        data[f] = rng.binomial(1, p, size=n_rows).astype("int8")

    # Device: mutually exclusive per session
    desktop = rng.binomial(1, desktop_share, size=n_rows).astype("int8")
    mobile = (1 - desktop).astype("int8")
    data["device_desktop"] = desktop
    data["device_mobile"] = mobile

    df = pd.DataFrame(data)

    # 3) Build logistic model spec & calibrate intercept
    spec = _build_logistic_spec(rng)
    spec = _calibrate_intercept_to_global_rate(df, spec, target_rate=0.05)

    # 4) Compute probabilities and sample conversions
    z = _compute_linear_predictor(df, spec)
    p = expit(z)
    converted = rng.binomial(1, p).astype("int8")
    df["converted"] = converted

    return df
