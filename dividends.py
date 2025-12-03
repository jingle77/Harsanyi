# dividends.py

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

Coalition = Tuple[str, ...]


@dataclass
class CoalitionStats:
    """
    Holds empirical statistics for a coalition S.

    - support_count: number of rows where all features in S are 1.
    - support_fraction: support_count / N.
    - conversion_rate: mean(target | S active).
    """
    coalition: Coalition
    support_count: int
    support_fraction: float
    conversion_rate: float


def estimate_num_coalitions(num_features: int, max_size: int) -> int:
    """
    Estimate total number of coalitions up to max_size given num_features.
    """
    max_size = min(max_size, num_features)
    total = 0
    for k in range(1, max_size + 1):
        total += math.comb(num_features, k)
    return total


def enumerate_coalitions(features: Sequence[str], max_size: int) -> List[Coalition]:
    """
    Enumerate all non-empty coalitions up to given size.

    Coalitions are represented as sorted tuples of feature names.
    """
    features_sorted = sorted(features)
    coalitions: List[Coalition] = []
    max_size = min(max_size, len(features_sorted))

    for k in range(1, max_size + 1):
        for combo in combinations(features_sorted, k):
            coalitions.append(combo)

    return coalitions


def _compute_single_coalition_stats(
    coalition: Coalition,
    data: pd.DataFrame,
    target_col: str,
) -> CoalitionStats:
    """
    Compute support and conversion rate for a single coalition S:

        v(S) = mean(target | X_i = 1 for all i in S)

    v(S) is undefined (NaN) if support_count == 0.
    """
    if len(coalition) == 1:
        # Slight micro-optimization: single column
        col = coalition[0]
        mask = data[col].values == 1
    else:
        mask = np.ones(len(data), dtype=bool)
        for col in coalition:
            mask &= (data[col].values == 1)

    support_count = int(mask.sum())
    if support_count == 0:
        return CoalitionStats(coalition, 0, 0.0, float("nan"))

    support_fraction = support_count / len(data)
    conv_rate = float(data.loc[mask, target_col].mean())

    return CoalitionStats(coalition, support_count, support_fraction, conv_rate)


def compute_coalition_values(
    data: pd.DataFrame,
    target_col: str,
    coalitions: Sequence[Coalition],
    min_support_fraction: float,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[Coalition, CoalitionStats]:
    """
    Compute coalition values v(S) for all given coalitions in parallel.

    Only coalitions with support_fraction >= min_support_fraction are retained.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing player features and target_col.
    target_col : str
        Name of the binary target column.
    coalitions : Sequence[Coalition]
        Coalitions to evaluate.
    min_support_fraction : float
        Minimum support as fraction of total rows (0–1).
    max_workers : Optional[int]
        Number of worker threads. If None, uses ThreadPoolExecutor default.
    progress_callback : Optional[Callable[[int, int], None]]
        Called as progress_callback(done, total) as tasks complete.
        Useful for updating a Streamlit progress bar.

    Returns
    -------
    Dict[Coalition, CoalitionStats]
        Mapping coalition -> stats for all coalitions that pass support.
    """
    n_rows = len(data)
    min_support_count = math.ceil(min_support_fraction * n_rows)
    total = len(coalitions)

    results: Dict[Coalition, CoalitionStats] = {}
    if total == 0:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_coalition = {
            executor.submit(
                _compute_single_coalition_stats, coalition, data, target_col
            ): coalition
            for coalition in coalitions
        }

        done = 0
        for future in as_completed(future_to_coalition):
            stats = future.result()
            done += 1

            if stats.support_count >= min_support_count and not math.isnan(
                stats.conversion_rate
            ):
                results[stats.coalition] = stats

            if progress_callback is not None:
                progress_callback(done, total)

    return results


def _non_empty_subcoalitions(coalition: Coalition) -> Iterable[Coalition]:
    """
    Yield all non-empty subcoalitions T ⊆ S, including S itself.
    """
    n = len(coalition)
    for r in range(1, n + 1):
        for sub in combinations(coalition, r):
            yield sub


def compute_harsanyi_dividends(
    coalition_values: Mapping[Coalition, CoalitionStats],
) -> Dict[Coalition, float]:
    """
    Compute Harsanyi dividends Δ(S) for all coalitions S that:

    - Have v(S) defined (i.e., appear in coalition_values), and
    - All non-empty subcoalitions T ⊆ S also appear in coalition_values
      (i.e., meet the support threshold).

    We use the standard Möbius inversion formula:

        v(∅) = 0  (by convention)
        Δ(S) = Σ_{T ⊆ S} (-1)^{|S| - |T|} * v(T)

    Since v(∅) = 0, we only sum over non-empty T.
    """
    dividends: Dict[Coalition, float] = {}

    # Sort coalitions by size for readability (not strictly necessary)
    coalitions_sorted = sorted(coalition_values.keys(), key=len)

    for S in coalitions_sorted:
        v_S = coalition_values[S].conversion_rate
        size_S = len(S)

        # Check that all non-empty subcoalitions T ⊆ S have v(T)
        subcoalitions = list(_non_empty_subcoalitions(S))
        if any(T not in coalition_values for T in subcoalitions):
            # Skip Harsanyi dividend for S if any subcoalition lacks support
            continue

        # Compute Δ(S)
        delta = 0.0
        for T in subcoalitions:
            v_T = coalition_values[T].conversion_rate
            size_T = len(T)
            sign = (-1) ** (size_S - size_T)
            delta += sign * v_T

        dividends[S] = float(delta)

    return dividends


def build_results_dataframe(
    coalition_values: Mapping[Coalition, CoalitionStats],
    dividends: Mapping[Coalition, float],
) -> pd.DataFrame:
    """
    Build a tidy DataFrame with one row per coalition S that has a Harsanyi dividend.

    Columns:
        - coalition (str)
        - coalition_size (int)
        - support_count (int)
        - support_fraction (float)
        - conversion_rate (float)
        - harsanyi_dividend (float)
    """
    rows = []
    for S, delta in dividends.items():
        stats = coalition_values[S]
        rows.append(
            {
                "coalition": " & ".join(S),
                "coalition_size": len(S),
                "support_count": stats.support_count,
                "support_fraction": stats.support_fraction,
                "conversion_rate": stats.conversion_rate,
                "harsanyi_dividend": delta,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("harsanyi_dividend", ascending=False).reset_index(drop=True)

    return df
