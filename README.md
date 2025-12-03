# Harsanyi

This repo contains a Streamlit app for analyzing **interaction effects** between ecommerce behaviors using **Harsanyi dividends**.

The app is designed for **page-journey stakeholders** (product, UX, marketing) who want to understand how combinations of channels, pages, and devices relate to conversion — beyond simple single-feature lift.

---

## What this app does

- Loads data from:
  - **Synthetic ecommerce sessions** (one click).
  - **User-uploaded CSVs** (real data).
- Treats selected **binary features** as “players” in a cooperative game.
- Computes:
  - **Coalition values**: empirical conversion rates for feature combinations.
  - **Harsanyi dividends**: pure synergy/interaction terms via Möbius inversion.
- Handles sparsity with:
  - **Minimum support threshold** (% of total rows).
  - **Maximum coalition size** (e.g. up to 3–5 features).
- Uses **parallel processing** + a **progress bar** so you can see compute progress.
- Outputs a sortable **results table** and **CSV download** of dividends.

---

## Synthetic data design

The synthetic generator creates session-level data with:

- **Channels**
  - `email`
  - `seo`
  - `sem`
  - `direct`
  - `display`
  - `social`
  - `affiliate`

- **Pages**
  - `product_page_a`, `product_page_b`, `product_page_c`
  - `product_page_d`, `product_page_e`, `product_page_f`
  - `deals_page`
  - `search_page`
  - `homepage`
  - `account_page`
  - `support_page`

- **Device**
  - `device_desktop`
  - `device_mobile` (mutually exclusive with desktop)

- **Target**
  - `converted` ∈ {0, 1}, with a global conversion rate calibrated to ~**5%**.

Internally:

- Each feature has a **marginal probability** (how often it occurs) drawn from realistic ranges.
- Each feature has a **“propensity to convert” score** on a 1–10 scale (based on domain intuition).
- The app maps those scores to **logistic coefficients**:
  - Low scores → neutral or negative effect.
  - Mid scores → moderate positive effect.
  - High scores → strong positive effect.
- A few intuitive **interaction terms** are added, e.g.:
  - `email × deals_page`
  - `seo × search_page`
  - `sem × product_page_a`
  - `affiliate × deals_page`
  - `device_desktop × account_page`
- The intercept is calibrated so the average predicted conversion rate is ≈ 5%, and `converted` is drawn as Bernoulli(p).

Each run of the synthetic generator:
- Uses new random draws for marginals and coefficients,
- But stays within realistic, interpretable ranges.

---

## Harsanyi dividends (math in plain language)

Let the user select a set of **binary features** as players (e.g. pages, channels, device flags).

For any **coalition** \( S \) (subset of these features):

- **Coalition value** \( v(S) \) is defined as the **empirical conversion rate** among rows where all features in \( S \) are 1:

\[
v(S) = \mathbb{E}[\text{converted} \mid X_i = 1 \ \forall i \in S]
\]

- We set \( v(\emptyset) = 0 \) by convention.

The **Harsanyi dividend** \( \Delta(S) \) is:

\[
\Delta(S) = \sum_{T \subseteq S} (-1)^{|S| - |T|} v(T)
\]

This is a Möbius inversion on the subset lattice and can be interpreted as the **pure synergy** attributable to the coalition \( S \) that cannot be explained by any of its subcoalitions.

### Sparsity handling

To avoid noisy estimates:

- Let **support_count(S)** = number of rows where all features in S are 1.
- Let **support_fraction(S)** = support_count(S) / N.

We:

1. Only keep v(S) if `support_fraction(S) >= min_support` (user slider, in %).
2. Only compute Δ(S) if **all non-empty subcoalitions** T ⊆ S:
   - also pass the support threshold, and
   - have defined v(T).

This is intentionally conservative: higher-order synergies are only reported when all lower-order pieces are reasonably well-estimated.

---

## Running the app

### 1. Environment

Dependencies are specified in `requirements.txt`, roughly:

```txt
streamlit
pandas
numpy
scipy
