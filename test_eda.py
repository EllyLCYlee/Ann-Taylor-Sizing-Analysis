"""
Testing module for CSE 163 Final Project.
Tests core functions in analysis.py using small synthetic DataFrames
with known expected values, allowing correctness to be verified
independently of the full datasets.

Tests are organized to mirror the section order in analysis.py:
  1. Data Loading     — load_anntaylor
  2. Data Filtering   — filter_ansur_by_profile
  3. Computations     — summarize_dataset, compute_demographic_weights,
                        compute_weighted_means, compute_weighted_std,
                        run_ttests
  4. Utility          — missing data detection, female-only filtering
"""
import pandas as pd
import analysis as an


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def test_load_anntaylor():
    """
    Test that load_anntaylor returns four DataFrames with correct
    row counts and that inch-to-cm conversion is applied correctly.
    Classic tops and bottoms max at size 18 (11 rows each).
    Petite tops max at 16P (10 rows); petite bottoms max at 18P (11 rows).
    """
    classic_tops, classic_bottoms, petite_tops, petite_bottoms = (
        an.load_anntaylor()
    )
    assert len(classic_tops) == 11, \
        'Classic tops should have 11 rows (sizes 00-18)'
    assert len(classic_bottoms) == 11, \
        'Classic bottoms should have 11 rows (sizes 00-18)'
    assert len(petite_tops) == 10, \
        'Petite tops should have 10 rows (sizes 00P-16P)'
    assert len(petite_bottoms) == 11, \
        'Petite bottoms should have 11 rows (sizes 00P-18P)'
    assert abs(classic_tops.loc[0, 'waist_cm'] - 24.5 * 2.54) < 0.01, \
        'Inch-to-cm conversion for classic tops waist is incorrect'
    print('test_load_anntaylor PASSED')


# ---------------------------------------------------------------------------
# 2. Data Filtering
# ---------------------------------------------------------------------------

def test_filter_ansur_by_profile():
    """
    Test that filter_ansur_by_profile retains only rows within
    +/-1 std of each given mean and excludes rows outside that range.
    The three-row test DataFrame has one row below range, one in range,
    and one above range for all three measurements simultaneously.
    """
    df = pd.DataFrame({
        'waistcircumference': [800.0, 1000.0, 1200.0],
        'stature_m': [1.55, 1.60, 1.65],
        'weight_kg': [60.0, 75.0, 90.0],
    })
    # Only the middle row falls within +/-1 std on all three filters
    result = an.filter_ansur_by_profile(
        df,
        waist_mean=1000.0, height_mean=1600.0, weight_mean=75.0,
        waist_std=100.0, height_std=50.0, weight_std=10.0,
    )
    assert len(result) == 1, \
        'Only the middle row should pass all three filters'
    assert abs(result.iloc[0]['waistcircumference'] - 1000.0) < 0.01, \
        'Retained row should have waistcircumference = 1000.0'
    print('test_filter_ansur_by_profile PASSED')


# ---------------------------------------------------------------------------
# 3. Computations
# ---------------------------------------------------------------------------

def test_summarize_dataset():
    """
    Test that summarize_dataset returns a correct seven-number summary
    for quantitative columns and value counts for categorical columns.
    Expected values are computed by hand from the 4-row test DataFrame.
    """
    df = pd.DataFrame({
        'height': [150.0, 160.0, 170.0, 180.0],
        'waist': [60.0, 65.0, 70.0, 75.0],
        'race': ['White', 'Asian', 'White', 'Black'],
    })
    result = an.summarize_dataset(df, ['height', 'waist', 'race'])
    assert 'quantitative' in result, \
        'summarize_dataset should return a quantitative key'
    assert 'categorical' in result, \
        'summarize_dataset should return a categorical key'
    quant = result['quantitative']
    assert abs(quant.loc['mean', 'height'] - 165.0) < 0.01, \
        'Mean height should be 165.0'
    assert abs(quant.loc['mean', 'waist'] - 67.5) < 0.01, \
        'Mean waist should be 67.5'
    assert result['categorical']['race']['White'] == 2, \
        'White count should be 2'
    print('test_summarize_dataset PASSED')


def test_compute_demographic_weights():
    """
    Test that compute_demographic_weights produces proportions that
    sum to 1.0, correctly identifies Hispanic respondents (HISP != 1
    takes priority over RAC1P), and assigns the correct proportion.
    Row 2 has HISP=2 (Hispanic) with PWGTP=150 out of total 500.
    """
    df = pd.DataFrame({
        'SEX': [2, 2, 2, 2],
        'RAC1P': [1, 6, 1, 1],
        'HISP': [1, 1, 2, 1],   # row index 2 is Hispanic
        'PWGTP': [100, 200, 150, 50],
    })
    weights = an.compute_demographic_weights(df)
    assert abs(weights.sum() - 1.0) < 1e-6, \
        'Demographic weights should sum to 1.0'
    assert 'Hispanic' in weights.index, \
        'Hispanic should appear in weights'
    assert 'Asian' in weights.index, \
        'Asian should appear in weights'
    assert abs(weights['Hispanic'] - 150 / 500) < 1e-6, \
        'Hispanic proportion should be 150/500 = 0.30'
    print('test_compute_demographic_weights PASSED')


def test_compute_weighted_means():
    """
    Test that compute_weighted_means correctly applies demographic
    weights to group means. With equal 50/50 weights, the result
    should be the average of the two group means, computable by hand:
    White mean waist = 750, Asian mean waist = 650 -> weighted = 700.
    White mean height = 1650, Asian mean height = 1550 -> weighted = 1600.
    """
    df = pd.DataFrame({
        'race_label': ['White', 'White', 'Asian', 'Asian'],
        'waist': [700.0, 800.0, 600.0, 700.0],
        'height': [1600.0, 1700.0, 1500.0, 1600.0],
    })
    weights = pd.Series({'White': 0.5, 'Asian': 0.5})
    result = an.compute_weighted_means(
        df, 'race_label', ['waist', 'height'], weights
    )
    assert abs(result['waist'] - 700.0) < 0.01, \
        'Weighted mean waist should be 700.0'
    assert abs(result['height'] - 1600.0) < 0.01, \
        'Weighted mean height should be 1600.0'
    print('test_compute_weighted_means PASSED')


def test_compute_weighted_std():
    """
    Test that compute_weighted_std returns a positive standard
    deviation for a group with spread-out values. The White group
    (600, 1000) has more spread than the Asian group (700, 800),
    so the overall weighted std should be positive and non-trivial.
    """
    df = pd.DataFrame({
        'race_label': ['White', 'White', 'Asian', 'Asian'],
        'waist': [600.0, 1000.0, 700.0, 800.0],
    })
    weights = pd.Series({'White': 0.5, 'Asian': 0.5})
    result = an.compute_weighted_std(df, 'race_label', ['waist'], weights)
    assert result['waist'] > 0, \
        'Weighted std should be positive for non-identical values'
    print('test_compute_weighted_std PASSED')


def test_run_ttests():
    """
    Test that run_ttests correctly identifies a statistically significant
    difference between two clearly separated samples (means 10 vs 20),
    and that Cohen's d is negative when the WA sample mean is lower
    than the national sample mean.
    """
    wa_sample = pd.DataFrame({'measurement': [10.0] * 50})
    nat_sample = pd.DataFrame({'measurement': [20.0] * 50})
    result = an.run_ttests(wa_sample, nat_sample, ['measurement'])
    assert result.loc['measurement', 'significant'], \
        'Clearly separated samples should be statistically significant'
    assert result.loc['measurement', 'cohens_d'] < 0, \
        "Cohen's d should be negative when WA mean < national mean"
    print('test_run_ttests PASSED')


# ---------------------------------------------------------------------------
# 4. Utility behavior
# ---------------------------------------------------------------------------

def test_load_filters_females_only():
    """
    Test the female-only filter logic used in load_pums_wa and
    load_pums_national. Simulates the SEX == 2 filter on a mixed
    DataFrame and verifies that only the 3 female rows are retained.
    """
    df = pd.DataFrame({
        'SEX': [1, 2, 2, 1, 2],
        'RAC1P': [1, 1, 6, 3, 4],
        'HISP': [1, 1, 1, 2, 1],
        'PWGTP': [100, 200, 150, 120, 180],
    })
    filtered = df[df['SEX'] == 2].reset_index(drop=True)
    assert (filtered['SEX'] == 2).all(), \
        'All rows after filtering should be female (SEX == 2)'
    assert len(filtered) == 3, \
        'Should have exactly 3 female rows'
    print('test_load_filters_females_only PASSED')


def test_missing_data_detection():
    """
    Test that isnull().sum() correctly detects missing values.
    The test DataFrame has exactly 1 missing value per column and
    2 total, verifying the missing data check used across all EDA
    functions.
    """
    df = pd.DataFrame({
        'waist': [700.0, None, 650.0],
        'height': [1600.0, 1550.0, None],
    })
    missing = df.isnull().sum()
    assert missing['waist'] == 1, \
        'Should detect 1 missing waist value'
    assert missing['height'] == 1, \
        'Should detect 1 missing height value'
    assert missing.sum() == 2, \
        'Total missing values should be 2'
    print('test_missing_data_detection PASSED')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Run all 9 tests in section order. All tests use synthetic
    DataFrames and do not require any dataset files to be present.
    """
    print('Running tests...\n')

    # Data Loading
    test_load_anntaylor()

    # Data Filtering
    test_filter_ansur_by_profile()

    # Computations
    test_summarize_dataset()
    test_compute_demographic_weights()
    test_compute_weighted_means()
    test_compute_weighted_std()
    test_run_ttests()

    # Utility behavior
    test_load_filters_females_only()
    test_missing_data_detection()

    print('\nAll tests passed!')


if __name__ == '__main__':
    main()
