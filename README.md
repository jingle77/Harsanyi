# Harsanyi Dividend Explorer for Ecommerce Journeys

![Alt text](harsanyi_gif.mp4)

This repo contains a Streamlit app for analyzing **interaction effects** between ecommerce behaviors using **Harsanyi dividends**.

The app is designed for **page-journey stakeholders** (product, UX, marketing) who want to understand how combinations of channels, pages, and devices relate to conversion 

---

## What this app does

- Loads data from:
  - **Synthetic ecommerce sessions** (one click).
  - **User-uploaded CSVs** (real data).
- Treats selected **binary features** as “players” in a cooperative game.
- Computes:
  - **Coalition values**: empirical conversion rates for feature combinations.
  - **Harsanyi dividends**: pure synergy/interaction terms via a Möbius inversion on subsets.
- Handles sparsity with:
  - **Minimum support threshold** (% of total rows).
  - **Maximum coalition size** (e.g. up to 3–5 features).
- Uses **parallel processing** + a **progress bar** so you can see compute progress.
- Outputs a sortable **results table** and **CSV download** of dividends.

---

## Synthetic data design

The synthetic generator creates session-level data with:

**Channels**

- `email`
- `seo`
- `sem`
- `direct`
- `display`
- `social`
- `affiliate`

**Pages**

- `product_page_a`, `product_page_b`, `product_page_c`
- `product_page_d`, `product_page_e`, `product_page_f`
- `deals_page`
- `search_page`
- `homepage`
- `account_page`
- `support_page`

**Device**

- `device_desktop`
- `device_mobile` (mutually exclusive with desktop)

**Target**

- `converted` ∈ {0, 1}, with a global conversion rate calibrated to around **5%**.

Internally:

- Each feature has a **marginal probability** (how often it occurs) drawn from realistic ranges.
- Each feature has a **“propensity to convert” score** on a 1–10 scale (based on domain intuition).
- The app maps those scores to **logistic coefficients**:
  - Low scores → neutral or negative effect.
  - Mid scores → moderate positive effect.
  - High scores → strong positive effect.
- A few intuitive **interaction terms** are added, for example:
  - `email` × `deals_page`
  - `seo` × `search_page`
  - `sem` × `product_page_a`
  - `affiliate` × `deals_page`
  - `device_desktop` × `account_page`
- The intercept is calibrated so the average predicted conversion rate is ≈ 5%, and `converted` is drawn as `Bernoulli(p)`.

Each run of the synthetic generator:

- Uses new random draws for marginals and coefficients,
- But stays within realistic, interpretable ranges.

---

## Harsanyi dividends

Let the user select a set of **binary features** as players (e.g. pages, channels, device flags).

### Coalition value: v(S)
For any **coalition** S (subset of these features), define the **coalition value**:
v(S) = mean(converted | all features in S are 1)

### Harsanyi Dividend: Dividend(S)
Δ(S) = sum over all T ⊆ S of [ (-1)^(|S| - |T|) * v(T) ]

---
## Running the app

### 1. Environment

Dependencies are specified in `requirements.txt`:

```txt
streamlit
pandas
numpy
scipy
```

Install Dependencies:
```bash
pip install -r requirements.txt
```
Run the application in Streamlit:
```bash
streamlit run app.py
```

