# app.py

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from synthetic_data import generate_synthetic_ecommerce_data, ALL_BINARY_FEATURES
from dividends import (
    estimate_num_coalitions,
    enumerate_coalitions,
    compute_coalition_values,
    compute_harsanyi_dividends,
    build_results_dataframe,
)


def _detect_binary_columns(df: pd.DataFrame, max_unique: int = 3) -> List[str]:
    """
    Detect columns that look binary (values in {0,1} or boolean).
    """
    binary_cols: List[str] = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue

        if series.dtype == bool:
            series = series.astype(int)

        unique_vals = series.unique()
        if len(unique_vals) <= max_unique and set(unique_vals).issubset({0, 1}):
            binary_cols.append(col)

    return binary_cols


def main() -> None:
    st.set_page_config(
        page_title="Harsanyi Dividends for Ecommerce Behaviors",
        layout="wide",
    )
    st.title("Harsanyi Dividend Explorer for Ecommerce Behaviors")

    if "data" not in st.session_state:
        st.session_state.data = None
    if "results_df" not in st.session_state:
        st.session_state.results_df = None
    if "target_col" not in st.session_state:
        st.session_state.target_col = None
    if "player_features" not in st.session_state:
        st.session_state.player_features = []

    with st.sidebar:
        st.header("1. Data source")

        data_source = st.radio(
            "Choose data source",
            ("Generate synthetic data", "Upload CSV"),
        )

        if data_source == "Generate synthetic data":
            n_rows = st.number_input(
                "Number of rows",
                min_value=10_000,
                max_value=500_000,
                value=100_000,
                step=10_000,
            )
            if st.button("Generate synthetic data"):
                df = generate_synthetic_ecommerce_data(n_rows=int(n_rows))
                st.session_state.data = df
                st.session_state.results_df = None
                st.session_state.target_col = "converted"
                st.session_state.player_features = [
                    c for c in ALL_BINARY_FEATURES if c in df.columns
                ]
                conv_rate = float(df["converted"].mean())
                st.success(
                    f"Generated synthetic dataset with {len(df):,} rows, "
                    f"{len(df.columns)} columns. "
                    f"Overall conversion rate: {conv_rate:.2%}."
                )

        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.session_state.results_df = None

        if st.session_state.data is not None:
            df = st.session_state.data

            st.header("2. Configuration")

            binary_cols = _detect_binary_columns(df)
            if not binary_cols:
                st.warning(
                    "No binary columns detected. "
                    "Please ensure your features and target are encoded as 0/1."
                )
            else:
                target_default = None
                if st.session_state.target_col in binary_cols:
                    target_default = st.session_state.target_col
                elif "converted" in binary_cols:
                    target_default = "converted"
                else:
                    target_default = binary_cols[0]

                target_col = st.selectbox(
                    "Target column (binary)",
                    options=binary_cols,
                    index=binary_cols.index(target_default)
                    if target_default in binary_cols
                    else 0,
                )
                st.session_state.target_col = target_col

                candidate_players = [c for c in binary_cols if c != target_col]

                if st.session_state.player_features:
                    default_players = [
                        c
                        for c in st.session_state.player_features
                        if c in candidate_players
                    ]
                    if not default_players:
                        default_players = candidate_players
                else:
                    default_players = candidate_players

                player_features = st.multiselect(
                    "Player features (binary)",
                    options=candidate_players,
                    default=default_players,
                    help="These features define the coalitions for Harsanyi dividends.",
                )
                st.session_state.player_features = player_features

                min_support_pct = st.slider(
                    "Minimum coalition support (% of rows)",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                )

                max_coalition_size = st.slider(
                    "Maximum coalition size",
                    min_value=1,
                    max_value=5,
                    value=3,
                )

                n_players = len(player_features)
                est_coalitions = (
                    estimate_num_coalitions(n_players, max_coalition_size)
                    if n_players > 0
                    else 0
                )
                st.caption(
                    f"Estimated number of coalitions up to size {max_coalition_size}: "
                    f"{est_coalitions:,}"
                )

                too_many = est_coalitions > 50_000
                if too_many:
                    st.error(
                        "Too many coalitions (> 50,000). "
                        "Reduce the number of player features or the maximum coalition size."
                    )

                st.session_state.config = {
                    "min_support_pct": float(min_support_pct),
                    "max_coalition_size": int(max_coalition_size),
                    "too_many": bool(too_many),
                }

    df = st.session_state.data
    if df is None:
        st.info("Select a data source in the sidebar to get started.")
        return

    target_col = st.session_state.target_col
    player_features = st.session_state.player_features
    config = st.session_state.get("config", {})
    min_support_pct = config.get("min_support_pct", 1.0)
    max_coalition_size = config.get("max_coalition_size", 3)
    too_many = config.get("too_many", False)

    st.subheader("Current data summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", f"{len(df.columns):,}")
    with col3:
        if target_col in df.columns:
            conv_rate = float(df[target_col].mean())
            st.metric("Overall conversion rate", f"{conv_rate:.2%}")
        else:
            st.metric("Overall conversion rate", "N/A")

    st.write("**Selected target:**", target_col)
    st.write("**Player features (coalition players):**", ", ".join(player_features))

    if not player_features:
        st.warning("Please select at least one player feature in the sidebar.")
        return
    if target_col is None or target_col not in df.columns:
        st.warning("Please select a valid binary target column in the sidebar.")
        return

    st.subheader("Harsanyi dividends")

    compute_disabled = too_many

    if st.button(
        "Calculate Harsanyi dividends",
        type="primary",
        disabled=compute_disabled,
    ):
        if too_many:
            st.stop()

        for col in player_features + [target_col]:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)

        coalitions = enumerate_coalitions(
            features=player_features,
            max_size=max_coalition_size,
        )
        min_support_fraction = min_support_pct / 100.0

        progress_bar = st.progress(0.0)
        progress_text = st.empty()

        def progress_callback(done: int, total: int) -> None:
            fraction = done / total
            progress_bar.progress(fraction)
            progress_text.text(f"Computed coalition values: {done}/{total}")

        st.write("Computing coalition conversion rates (v(S))...")
        coalition_values = compute_coalition_values(
            data=df,
            target_col=target_col,
            coalitions=coalitions,
            min_support_fraction=min_support_fraction,
            progress_callback=progress_callback,
        )

        if not coalition_values:
            st.warning(
                "No coalitions passed the minimum support threshold. "
                "Try lowering the minimum support or selecting more frequent features."
            )
            st.session_state.results_df = None
            return

        progress_text.text(
            f"Computed coalition values for {len(coalition_values):,} coalitions."
        )

        st.write("Computing Harsanyi dividends Δ(S)...")
        dividends = compute_harsanyi_dividends(coalition_values)
        if not dividends:
            st.warning(
                "No coalitions had all subcoalitions above the support threshold, "
                "so no Harsanyi dividends could be computed. "
                "Try lowering the minimum support or limiting coalition size."
            )
            st.session_state.results_df = None
            return

        results_df = build_results_dataframe(coalition_values, dividends)
        st.session_state.results_df = results_df

        st.success(
            f"Computed Harsanyi dividends for {len(results_df):,} coalitions."
        )

    results_df = st.session_state.results_df
    if results_df is not None and not results_df.empty:
        st.subheader("Results")
        st.dataframe(results_df, use_container_width=True)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results as CSV",
            data=csv_bytes,
            file_name="harsanyi_dividends.csv",
            mime="text/csv",
        )
    elif results_df is not None and results_df.empty:
        st.info("No Harsanyi dividends to display (empty result set).")


if __name__ == "__main__":
    main()
