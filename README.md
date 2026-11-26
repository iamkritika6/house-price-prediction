# Bengaluru House Price Prediction

Predicting the **asking price** of residential property listings in Bengaluru, India.

> **Status: in progress.** The data pipeline and environment are set up and the raw
> data has been profiled. No model has been trained yet, so this README contains no
> model results. Every number below was produced by a script in this repository and
> can be regenerated.

## What this project predicts, and what it does not

The target variable is the **price a property was listed at**, not the price it sold
for. These are listings scraped from a property portal, not registered transactions.

This distinction matters and is not a technicality:

- Asking prices are set by sellers and agents, and in a soft market they sit above
  what properties actually achieve.
- There is no official, openly published, transaction-level record of Indian
  residential property sales. The Reserve Bank of India publishes a city-level
  housing price *index*, but that is a quarterly aggregate and cannot be used to
  predict the price of an individual property.

So this project is honest about its scope: it models what sellers ask, which is a
useful and well-defined question, rather than pretending to model what buyers pay.

## Data

| | |
|---|---|
| Rows | 13,320 |
| Columns | 9 |
| City | Bengaluru |
| Period | Listings described in the source as reflecting the 2017 market |
| Licence | CC0 (public domain) as published on OpenML |

### Provenance

The dataset originates as `Bengaluru_House_Data.csv`, published on Kaggle
(`amitabhajoy/bengaluru-house-price-data`), scraped from property portal listings.

Kaggle requires an account to download, so the file used here was obtained from a
public mirror and then **verified against an independent copy** before being used:

| Check | Result |
|---|---|
| Row count vs OpenML dataset 46284 | 13,320 in both — match |
| Column names | `area_type, availability, location, size, society, total_sqft, bath, balcony, price` |
| SHA-256 of `Bengaluru_House_Data.csv` | `23ddf71be7d79a259b62e125b9d56cb531f9d051b0da650c0fe7cc82365a1015` |
| SHA-256 of `bengaluru_real_estate_price.arff` (OpenML copy) | `b0175b86e375176846c1e0b80f709f403129688274d420346fd589547d3c122f` |

**A deliberate choice about which copy to use.** The OpenML copy is more convenient:
`total_sqft` arrives already numeric with no missing values. It was rejected as the
primary source precisely *because* of that. In the published data `total_sqft` is a
text column containing values such as `2100 - 2850` and `34.46Sq. Meter`; OpenML has
converted it by a method that is not documented anywhere in the dataset record. Using
it would have meant building on a transformation that could not be described or
defended.

The OpenML copy is kept as a **cross-check**: once `total_sqft` is parsed here, the
result can be compared against OpenML's independently produced numeric column.

Data files are not committed to this repository. See [Reproducing](#reproducing).

### What the raw data actually looks like

Produced by `scripts/explore_raw.py`:

**Missing values**

| Column | Missing | % |
|---|---|---|
| `society` | 5,502 | 41.31% |
| `balcony` | 609 | 4.57% |
| `bath` | 73 | 0.55% |
| `size` | 16 | 0.12% |
| `location` | 1 | 0.01% |
| `area_type`, `availability`, `total_sqft`, `price` | 0 | 0.00% |

**Distinct values**

| Column | Distinct |
|---|---|
| `location` | 1,305 |
| `society` | 2,688 |
| `total_sqft` | 2,117 |
| `price` | 1,994 |
| `availability` | 81 |
| `size` | 31 |
| `bath` | 19 |
| `area_type` | 4 |
| `balcony` | 4 |

**Known problems in the raw data, all of which the pipeline must handle**

1. **529 exactly duplicated rows.** These must be removed *before* the train/test
   split. If they are not, the same listing can appear in both training and test
   data, and the test score becomes optimistic — a leak.
2. **`total_sqft` is not numeric.** 13,073 values are plain numbers; 247 are not.
   Of those, 201 are ranges (`2100 - 2850`) and 46 carry units
   (`34.46Sq. Meter`, `4125Perch`, plus Acres, Cents, Guntha, Grounds, Sq. Yards).
3. **`size` is free text** with two different conventions for the same thing:
   `2 BHK` (5,199 rows) and `2 Bedroom` (329 rows) both mean two bedrooms.
4. **`location` has a long tail** — 1,305 distinct values over 13,320 rows, so many
   locations appear only once or twice.
5. **`society` is 41% missing**, which is itself likely to be informative rather
   than random.
6. **`area_type` is not comparable across rows.** Carpet area, built-up area and
   super built-up area measure different things, so the same `total_sqft` figure
   means different amounts of usable space depending on this column.

**`price` as published**

| | |
|---|---|
| Count | 13,320 |
| Min | 8.0 |
| 25th percentile | 50.0 |
| Median | 72.0 |
| Mean | 112.57 |
| 75th percentile | 120.0 |
| Max | 3,600.0 |

The source documents this column as being in **lakhs** (1 lakh = ₹100,000), which is
consistent with the distribution above: a median of ₹72 lakh for Bengaluru property.
The mean sits well above the median, so the distribution is strongly right-skewed.

## Environment

The project runs in Docker rather than directly on the host machine. This is not
decoration: on the Windows machine this was developed on, Windows Smart App Control
blocks SciPy's unsigned `_flapack` binary from loading, which makes scikit-learn and
SHAP fail to import. Running in a Linux container avoids that without disabling a
host security feature, and has the side benefit that the results reproduce on any
machine with Docker.

Verified working inside the container:

| Package | Version |
|---|---|
| Python | 3.12.16 |
| numpy | 2.5.5 |
| pandas | 3.1.1 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| lightgbm | 4.7.0 |
| shap | 0.54.1 |
| matplotlib | 3.11.2 |
| pytest | 9.2.0 |

Exact pinned versions are in `requirements-lock.txt`.

## Reproducing

```bash
docker compose build
```

Download the raw data into `data/raw/` (not committed):

```bash
curl -L -o data/raw/Bengaluru_House_Data.csv \
  https://huggingface.co/Mridul2003/Auto_Analysis/resolve/main/autoanalysis/Automate_Analysis/AutoML/Bengaluru_House_Data.csv
```

Verify you have the same bytes this project was built on:

```bash
sha256sum data/raw/Bengaluru_House_Data.csv
# expected: 23ddf71be7d79a259b62e125b9d56cb531f9d051b0da650c0fe7cc82365a1015
```

Profile the raw data:

```bash
docker compose run --rm hpp python scripts/explore_raw.py
```

## Repository layout

```
data/raw/          raw data, not committed
src/hpp/           importable package: cleaning, features, modelling
scripts/           entry points that produce numbers and figures
tests/             pytest suite
reports/figures/   generated plots, not committed
reports/metrics/   generated metrics, not committed
```

## Licence

Code in this repository is available under the MIT Licence. The dataset is CC0 as
published on OpenML; it is not redistributed here.
