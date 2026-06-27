# Using Data to Verify Whether Ann Taylor Clothes Are True to Size for Women in Washington State

**Author:** Elly Lee | Student #2163622
**Course:** CSE 163 | Spring 2026

---

## Overview

This project investigates whether women in Washington state have
systematically different body measurements from the U.S. national
average, and whether Ann Taylor's published size guide adequately
serves the Washington state female population. The analysis combines
three public datasets (ANSUR II, NHANES, ACS PUMS) with a manually
transcribed Ann Taylor size guide to answer four research questions
through demographic weighting, proxy filtering, size distribution
analysis, and statistical significance testing.

---

## Required Installations

All dependencies are available via Anaconda. If not already installed,
run the following in your terminal:

```bash
pip install pandas matplotlib seaborn scipy
```

**Python version:** 3.13 (Anaconda environment recommended)

---

## Data Downloads

Certain datasets are not included in the submission due to file size.
Download each file and place it in a folder named `data/` inside
your project directory.

### 1. ANSUR II Women (2012)
- URL: https://www.openlab.psu.edu/ansur2/
- Download: `ANSUR II FEMALE Public.csv`
- Rename to: `ANSUR_II_FEMALE.csv`

### 2. NHANES 2015–2018 Women
- URL: https://www.openlab.psu.edu/nhanes/
- Download the pre-merged women's file (ages 20+)
- Rename to: `NHANES15-18_womenAge20YearsAndOver.csv`

### 3. ACS PUMS 2020–2024 — Washington State
- URL: https://www.census.gov/programs-surveys/acs/microdata/access.html
- Navigate to: 2024 → 5-Year → Person Files → Washington
- Download: `csv_pwa.zip`, extract to get `psam_p53.csv`

### 4. ACS PUMS 2020–2024 — National
- Same URL as above
- Navigate to: 2024 → 5-Year → Person Files → National
- Download: `csv_pus.zip`, extract to get four files:
  `psam_pusa.csv`, `psam_pusb.csv`, `psam_pusc.csv`, `psam_pusd.csv`

Your `data/` folder should contain:

```
data/
  ANSUR_II_FEMALE.csv
  NHANES15-18_womenAge20YearsAndOver.csv
  psam_p53.csv
  psam_pusa.csv
  psam_pusb.csv
  psam_pusc.csv
  psam_pusd.csv
```

---

## File Descriptions

| File | Description |
|---|---|
| `analysis.py` | Helper module containing all data loading, filtering, computation, and visualization functions. Not runnable on its own. |
| `eda.py` | Main runnable pipeline. Loads all datasets, runs EDA on each, and executes the analyses for RQ1, RQ2, and RQ4. |
| `test_eda.py` | Testing module with 9 unit tests verifying core functions in `analysis.py` using synthetic DataFrames. Runnable independently. |
| `report.pdf` | Full project report (10–15 pages) documenting methodology, EDA findings, results, and conclusions. |
| `README.md` | This file. |

---

## Setup

**Step 1 — Update the data and plots paths in `eda.py`.**

Open `eda.py` and update the two constants near the top to match
where you placed the data folder on your computer:

```python
DATA_DIR = '/path/to/your/data'
PLOTS_DIR = '/path/to/your/plots'
```

The `plots/` directory will be created automatically if it does
not already exist.

**Step 2 — Place all Python files in the same directory.**

`analysis.py`, `eda.py`, and `test_eda.py` must all be in the
same folder so that `eda.py` and `test_eda.py` can import from
`analysis.py`.

---

## Running the Project

### Run the full analysis

From the terminal, navigate to the project folder and run:

```bash
python eda.py
```

This will:
1. Load all datasets from `DATA_DIR`
2. Run EDA on ANSUR II, NHANES, ACS PUMS, and Ann Taylor size guides
3. Run the joined NHANES + PUMS analysis (RQ1, RQ3)
4. Run the ANSUR II proxy structural comparison (RQ1)
5. Run the Ann Taylor size distribution alignment (RQ2)
6. Run the statistical significance tests (RQ4)
7. Save all plots as `.png` files to `PLOTS_DIR`

**Expected runtime:** 3–5 minutes (dominated by loading the
8.2M-row national PUMS file).

### Run the tests

```bash
python test_eda.py
```

All 9 tests should pass. Two RuntimeWarnings from scipy and numpy
may appear during `test_run_ttests` — these are expected and result
from the test using perfectly identical synthetic values; they do not
affect correctness.

---

## Output

All plots are saved to `PLOTS_DIR`. The following files are generated:

| Plot file | Contents |
|---|---|
| `ansur_distributions.png` | Histograms of ANSUR II key measurements |
| `ansur_boxplots.png` | Box plots of ANSUR II key measurements |
| `nhanes_distributions.png` | Histograms of NHANES waist, height, weight |
| `nhanes_by_race.png` | Mean measurements by race/ethnicity (NHANES) |
| `pums_demographics.png` | WA vs. U.S. demographic composition |
| `pums_divergence.png` | Demographic divergence by group |
| `joined_nhanes_pums.png` | WA vs. U.S. weighted mean measurements |
| `anntaylor_tops_comparison.png` | Classic vs. Petite tops measurements |
| `anntaylor_bottoms_comparison.png` | Classic vs. Petite bottoms measurements |
| `rq1_ansur_proxy.png` | RQ1: structural measurements WA vs. U.S. |
| `rq1_ansur_differences.png` | RQ1: structural measurement differences |
| `rq2_classic_tops.png` | RQ2: Classic tops size distribution |
| `rq2_petite_tops.png` | RQ2: Petite tops size distribution |
| `rq2_classic_bottoms.png` | RQ2: Classic bottoms size distribution |
| `rq2_petite_bottoms.png` | RQ2: Petite bottoms size distribution |
| `rq4_significance.png` | RQ4: Cohen's d effect sizes by measurement |
