"""
Main pipeline for CSE 163 Final Project.
Loads all datasets, runs exploratory data analysis, and answers all
four research questions:

  RQ1 — How do WA women's body measurements compare to the U.S. average?
  RQ2 — How do WA women's measurements align with Ann Taylor's size guide?
  RQ3 — Is WA's demographic composition distinct enough to explain
         any sizing differences?
  RQ4 — Are the measurement differences statistically significant?

All helper functions (loading, computation, plotting) live in analysis.py.
Run this file directly to reproduce all results and saved plots.
"""
import os
import pandas as pd
import analysis as an

# Paths to data and plots directories
DATA_DIR = '/Users/ichaeyun/Downloads/Final Project/data'
PLOTS_DIR = '/Users/ichaeyun/Downloads/Final Project/plots'


def setup_plots_dir():
    """Create the plots output directory if it does not exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# EDA — individual dataset exploration
# ---------------------------------------------------------------------------

def run_ansur_eda(ansur):
    """
    Run EDA on the ANSUR II Women dataset.
    Prints shape, missing data summary, and variable summaries.
    Saves a distribution histogram grid and a box plot grid.
    """
    print('=' * 60)
    print('ANSUR II FEMALE - EDA')
    print('=' * 60)

    print(f'Rows: {ansur.shape[0]}, Columns: {ansur.shape[1]}')
    print('Each row represents one U.S. Army servicewoman.')

    # Missing data check
    missing = ansur.isnull().sum()
    if missing.sum() == 0:
        print('No missing data found in ANSUR II dataset.')
    else:
        print(f'Missing values found:\n{missing[missing > 0]}')

    # Seven-number summary for key structural measurements
    cols = [
        'sleeveoutseam', 'crotchheight',
        'biacromialbreadth', 'waistcircumference',
    ]
    print('\nVariables of interest summary:')
    print(an.summarize_dataset(ansur, cols))

    # Visualization 1: histograms showing measurement distributions
    an.plot_measurement_distributions(
        ansur, cols,
        title='ANSUR II: Distribution of Key Body Measurements',
        filename=os.path.join(PLOTS_DIR, 'ansur_distributions.png')
    )

    # Visualization 2: box plots showing spread and outliers
    an.plot_boxplots(
        ansur, cols,
        title='ANSUR II: Spread of Key Body Measurements',
        filename=os.path.join(PLOTS_DIR, 'ansur_boxplots.png')
    )


def run_nhanes_eda(nhanes):
    """
    Run EDA on the NHANES 2015-2018 Women dataset.
    Prints shape, missing data summary, variable summaries, and
    race/ethnicity counts. Saves distribution and grouped bar plots.
    """
    print('=' * 60)
    print('NHANES 2015-2018 WOMEN - EDA')
    print('=' * 60)

    print(f'Rows: {nhanes.shape[0]}, Columns: {nhanes.shape[1]}')
    print('Each row represents one civilian U.S. woman aged 20+.')

    # Missing data check — WTMEC4YR and RIDEXPRG are expectedly missing
    missing = nhanes.isnull().sum()
    if missing.sum() == 0:
        print('No missing data found in NHANES dataset.')
    else:
        print(f'Total missing values: {missing.sum()}')
        print(missing[missing > 0])

    # Seven-number summary for key measurement and weight columns
    cols = ['BMXWAIST', 'BMXHT', 'BMXWT', 'RIDRETH3', 'WTMEC2YR']
    print('\nVariables of interest summary:')
    print(an.summarize_dataset(nhanes, cols))

    # Descriptive race/ethnicity labels for printing and plotting
    race_map = {
        1: 'Mexican American', 2: 'Other Hispanic',
        3: 'Non-Hispanic White', 4: 'Non-Hispanic Black',
        6: 'Non-Hispanic Asian', 7: 'Other',
    }
    nhanes['race_label'] = nhanes['RIDRETH3'].map(race_map)
    print('\nRace/ethnicity counts:')
    print(nhanes['race_label'].value_counts())

    meas_cols = ['BMXWAIST', 'BMXHT', 'BMXWT']

    # Visualization 1: histograms of waist, height, and weight
    an.plot_measurement_distributions(
        nhanes, meas_cols,
        title='NHANES: Distribution of Key Body Measurements',
        filename=os.path.join(PLOTS_DIR, 'nhanes_distributions.png')
    )

    # Visualization 2: mean measurements grouped by race/ethnicity
    an.plot_grouped_bars(
        nhanes, group_col='race_label',
        value_cols=meas_cols,
        title='NHANES: Mean Body Measurements by Race/Ethnicity',
        filename=os.path.join(PLOTS_DIR, 'nhanes_by_race.png')
    )


def run_pums_eda(pums_wa, pums_national):
    """
    Run EDA on the ACS PUMS Washington state and national datasets.
    Prints shape, missing data, and demographic proportions for both.
    Saves a side-by-side demographic comparison chart and a divergence
    bar chart showing which groups differ most between WA and the U.S.
    """
    print('=' * 60)
    print('ACS PUMS - EDA')
    print('=' * 60)

    print(f'WA PUMS   - Rows: {pums_wa.shape[0]}, '
          f'Columns: {pums_wa.shape[1]}')
    print(f'National  - Rows: {pums_national.shape[0]}, '
          f'Columns: {pums_national.shape[1]}')
    print('Each row represents one survey respondent.')

    # Missing data check for the four columns we actually use
    for name, df in [('WA', pums_wa), ('National', pums_national)]:
        missing = df[['SEX', 'RAC1P', 'HISP', 'PWGTP']].isnull().sum()
        print(f'\n{name} PUMS missing values in key columns:')
        print(missing)

    # Compute PWGTP-weighted racial/ethnic proportions
    wa_weights = an.compute_demographic_weights(pums_wa)
    nat_weights = an.compute_demographic_weights(pums_national)

    print('\nWA state female racial/ethnic proportions:')
    print(wa_weights)
    print('\nNational female racial/ethnic proportions:')
    print(nat_weights)

    # Visualization 1: side-by-side demographic comparison
    an.plot_demographic_comparison(
        wa_weights, nat_weights,
        title='Female Racial/Ethnic Composition: Washington State vs. U.S.',
        filename=os.path.join(PLOTS_DIR, 'pums_demographics.png')
    )

    # Visualization 2: divergence chart sorted by magnitude
    divergence = (
        (wa_weights - nat_weights).abs().sort_values(ascending=False)
    )
    an.plot_divergence_bar(
        divergence,
        title='Demographic Divergence: Washington State vs. U.S. National',
        xlabel='Racial/Ethnic Group',
        ylabel='Absolute Proportion Difference',
        filename=os.path.join(PLOTS_DIR, 'pums_divergence.png')
    )


def run_anntaylor_eda(classic_tops, classic_bottoms,
                      petite_tops, petite_bottoms):
    """
    Run EDA on the four Ann Taylor size guide DataFrames.
    Prints shape, missing data, and variable summaries for each.
    Saves Classic vs. Petite comparison charts for tops and bottoms.
    """
    print('=' * 60)
    print('ANN TAYLOR SIZING - EDA')
    print('=' * 60)

    # Shape and missing data for each of the four DataFrames
    datasets = {
        'Classic Tops': classic_tops,
        'Classic Bottoms': classic_bottoms,
        'Petite Tops': petite_tops,
        'Petite Bottoms': petite_bottoms,
    }
    for name, df in datasets.items():
        print(f'{name} - Rows: {df.shape[0]}, Columns: {df.shape[1]}')
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print(f'No missing data found in {name} dataset.')
        else:
            print(f'Missing values in {name}:\n{missing[missing > 0]}')

    # Seven-number summary for cm measurement columns
    print(an.summarize_dataset(
        classic_tops, ['bust_cm', 'sleeve_cm', 'waist_cm']
    ))
    print(an.summarize_dataset(
        classic_bottoms,
        ['waist_regular_cm', 'waist_curvy_cm',
         'hip_regular_cm', 'hip_curvy_cm']
    ))
    print(an.summarize_dataset(
        petite_tops, ['bust_cm', 'sleeve_cm', 'waist_cm']
    ))
    print(an.summarize_dataset(
        petite_bottoms,
        ['waist_regular_cm', 'waist_curvy_cm',
         'hip_regular_cm', 'hip_curvy_cm']
    ))

    # Visualization: Classic vs. Petite for tops and bottoms
    an.plot_anntaylor_comparison(
        classic_tops, petite_tops,
        title=('Ann Taylor Tops: '
               'Classic vs. Petite Measurement Distributions'),
        filename=os.path.join(PLOTS_DIR, 'anntaylor_tops_comparison.png')
    )
    an.plot_anntaylor_comparison(
        classic_bottoms, petite_bottoms,
        title=('Ann Taylor Bottoms: '
               'Classic vs. Petite Measurement Distributions'),
        filename=os.path.join(PLOTS_DIR, 'anntaylor_bottoms_comparison.png')
    )


# ---------------------------------------------------------------------------
# EDA — joined datasets
# ---------------------------------------------------------------------------

def run_joined_eda(nhanes, pums_wa, pums_national):
    """
    Run EDA on the joined NHANES + ACS PUMS datasets.
    Applies PUMS demographic weights to NHANES group means to produce
    Washington-representative vs. national average body measurements.
    Prints a weighted summary table (means and stds) and saves a
    side-by-side means comparison chart.

    Returns wa_means, wa_stds, nat_means, nat_stds as Series so that
    downstream functions (run_rq1_ansur_proxy, run_rq4_significance)
    can use the correct demographically-weighted parameters.

    Note: ANSUR II is excluded from the joined EDA because it contains
    no race/ethnicity variable and cannot be demographically weighted.
    """
    print('=' * 60)
    print('JOINED DATASETS EDA (NHANES + PUMS)')
    print('=' * 60)

    wa_weights = an.compute_demographic_weights(pums_wa)
    nat_weights = an.compute_demographic_weights(pums_national)

    nhanes_cols = ['BMXWAIST', 'BMXHT', 'BMXWT']

    # Map NHANES race codes to PUMS race labels for weight alignment
    nhanes_race_map = {
        1: 'Hispanic', 2: 'Hispanic',
        3: 'White', 4: 'Black or African American',
        6: 'Asian', 7: 'Other',
    }
    nhanes['race_label'] = nhanes['RIDRETH3'].map(nhanes_race_map)

    # Weighted means and standard deviations for WA and national
    wa_means = an.compute_weighted_means(
        nhanes, 'race_label', nhanes_cols, wa_weights
    )
    nat_means = an.compute_weighted_means(
        nhanes, 'race_label', nhanes_cols, nat_weights
    )
    wa_stds = an.compute_weighted_std(
        nhanes, 'race_label', nhanes_cols, wa_weights
    )
    nat_stds = an.compute_weighted_std(
        nhanes, 'race_label', nhanes_cols, nat_weights
    )

    # Print full summary including stds
    summary = pd.DataFrame({
        'WA Mean': wa_means,
        'WA Std': wa_stds,
        'U.S. Mean': nat_means,
        'U.S. Std': nat_stds,
    })
    print('\nNHANES + PUMS: WA vs. U.S. weighted summary:')
    print(summary)

    # Means-only comparison for the visualization
    comparison_n = pd.DataFrame({
        'Washington State': wa_means,
        'U.S. National': nat_means,
    })
    an.plot_wa_vs_national(
        comparison_n,
        title='NHANES + PUMS: WA vs. U.S. Mean Body Measurements',
        filename=os.path.join(PLOTS_DIR, 'joined_nhanes_pums.png')
    )

    return wa_means, wa_stds, nat_means, nat_stds


# ---------------------------------------------------------------------------
# Research question analyses
# ---------------------------------------------------------------------------

def run_rq1_ansur_proxy(ansur, wa_means_nhanes, wa_stds_nhanes,
                        nat_means_nhanes, nat_stds_nhanes):
    """
    RQ1 — ANSUR II structural measurement proxy comparison.

    Since ANSUR II has no race/ethnicity variable, filters servicewomen
    whose waist, height, and weight fall within one WA-weighted (or
    national-weighted) standard deviation of the NHANES means. Treats
    the resulting subgroups as proxies for WA and U.S. women and
    compares seven structural measurements between them.

    Saves a side-by-side bar chart and a horizontal difference chart.
    """
    print('=' * 60)
    print('ANSUR II PROXY COMPARISON - RQ1')
    print('=' * 60)

    # Filter ANSUR II to WA-representative and national-representative
    # subgroups using demographically-weighted NHANES means and stds
    wa_filtered = an.filter_ansur_by_profile(
        ansur,
        waist_mean=wa_means_nhanes['BMXWAIST'],
        height_mean=wa_means_nhanes['BMXHT'],
        weight_mean=wa_means_nhanes['BMXWT'],
        waist_std=wa_stds_nhanes['BMXWAIST'],
        height_std=wa_stds_nhanes['BMXHT'],
        weight_std=wa_stds_nhanes['BMXWT'],
    )
    nat_filtered = an.filter_ansur_by_profile(
        ansur,
        waist_mean=nat_means_nhanes['BMXWAIST'],
        height_mean=nat_means_nhanes['BMXHT'],
        weight_mean=nat_means_nhanes['BMXWT'],
        waist_std=nat_stds_nhanes['BMXWAIST'],
        height_std=nat_stds_nhanes['BMXHT'],
        weight_std=nat_stds_nhanes['BMXWT'],
    )

    # Structural measurements relevant to clothing fit
    struct_cols = [
        'sleeveoutseam',       # full sleeve length
        'shoulderlength',      # neck to shoulder tip (seam placement)
        'biacromialbreadth',   # full shoulder width
        'crotchheight',        # inseam / leg length
        'thighcircumference',  # thigh fit (bottoms)
        'calfcircumference',   # calf fit (fitted pants)
        'hipbreadth',          # hip width (bottoms)
    ]

    print(f'WA proxy subgroup:       {len(wa_filtered)} women')
    print(f'National proxy subgroup: {len(nat_filtered)} women')

    comparison = pd.DataFrame({
        'Washington State': wa_filtered[struct_cols].mean(),
        'U.S. National': nat_filtered[struct_cols].mean(),
    })
    print('\nRQ1 ANSUR II proxy comparison (mm):')
    print(comparison)

    # Side-by-side bar chart (hard to read at different scales but
    # useful for showing absolute magnitudes)
    an.plot_wa_vs_national(
        comparison,
        title=(
            'RQ1: Estimated Structural Measurements — '
            'WA State vs. U.S. National Proxy'
        ),
        filename=os.path.join(PLOTS_DIR, 'rq1_ansur_proxy.png')
    )

    # Difference chart — plots WA minus national on a single scale,
    # making small differences readable regardless of measurement size
    an.plot_measurement_differences(
        comparison,
        title=(
            'RQ1: WA vs. U.S. Structural Measurement Differences'
            ' (ANSUR II Proxy)'
        ),
        filename=os.path.join(PLOTS_DIR, 'rq1_ansur_differences.png')
    )


def run_rq2_size_alignment(nhanes, classic_tops, classic_bottoms,
                           petite_tops, petite_bottoms,
                           wa_weights, nat_weights):
    """
    RQ2 — Ann Taylor size distribution alignment.

    Assigns each NHANES respondent to an Ann Taylor size bucket based
    on waist circumference, using PUMS demographic weights. Computes
    the proportion of WA and U.S. women in each bucket and an 'Above
    Range' bucket for women exceeding Ann Taylor's maximum size.

    Saves four size distribution charts (Classic/Petite x Tops/Bottoms).
    """
    print('=' * 60)
    print('ANN TAYLOR SIZE ALIGNMENT - RQ2')
    print('=' * 60)

    for name, size_guide, waist_col in [
        ('Classic Tops', classic_tops, 'waist_cm'),
        ('Petite Tops', petite_tops, 'waist_cm'),
        ('Classic Bottoms', classic_bottoms, 'waist_regular_cm'),
        ('Petite Bottoms', petite_bottoms, 'waist_regular_cm'),
    ]:
        wa_dist = an.assign_size_distribution(
            nhanes, size_guide, waist_col, wa_weights
        )
        nat_dist = an.assign_size_distribution(
            nhanes, size_guide, waist_col, nat_weights
        )
        comparison = pd.DataFrame({
            'Washington State': wa_dist,
            'U.S. National': nat_dist,
        })
        print(f'\n{name} size distribution:')
        print(comparison)

        an.plot_size_distribution(
            comparison,
            title=f'RQ2: {name} Size Distribution — WA vs. U.S.',
            filename=os.path.join(
                PLOTS_DIR,
                f'rq2_{name.lower().replace(" ", "_")}.png'
            )
        )


def run_rq4_significance(nhanes, ansur, wa_weights, nat_weights,
                         wa_means_nhanes, wa_stds_nhanes,
                         nat_means_nhanes, nat_stds_nhanes):
    """
    RQ4 — Statistical significance testing (Result Validity goal).

    For NHANES measurements: resamples NHANES proportionally to WA and
    national PUMS weights (n=2,000 each), then runs independent samples
    t-tests on BMXWAIST, BMXHT, and BMXWT.

    For ANSUR II structural measurements: runs t-tests directly between
    the WA and national proxy subgroups from filter_ansur_by_profile().

    Computes Cohen's d effect sizes alongside p-values to distinguish
    statistical significance from practical significance. Saves a
    horizontal bar chart of effect sizes colored by significance.
    """
    print('=' * 60)
    print('STATISTICAL SIGNIFICANCE TESTING - RQ4')
    print('=' * 60)

    # Map NHANES race codes to PUMS labels for demographic resampling
    nhanes_race_map = {
        1: 'Hispanic', 2: 'Hispanic',
        3: 'White', 4: 'Black or African American',
        6: 'Asian', 7: 'Other',
    }
    nhanes['race_label'] = nhanes['RIDRETH3'].map(nhanes_race_map)

    # Resample NHANES to match WA and national demographic profiles
    wa_sample = an.resample_by_weights(
        nhanes, 'race_label', wa_weights, n=2000
    )
    nat_sample = an.resample_by_weights(
        nhanes, 'race_label', nat_weights, n=2000
    )

    # T-tests on NHANES circumference measurements
    nhanes_cols = ['BMXWAIST', 'BMXHT', 'BMXWT']
    print('\nNHANES t-test results:')
    nhanes_results = an.run_ttests(wa_sample, nat_sample, nhanes_cols)
    print(nhanes_results)

    # T-tests on ANSUR II structural measurements via proxy subgroups
    struct_cols = [
        'sleeveoutseam', 'shoulderlength', 'biacromialbreadth',
        'crotchheight', 'thighcircumference',
        'calfcircumference', 'hipbreadth',
    ]
    wa_ansur = an.filter_ansur_by_profile(
        ansur,
        waist_mean=wa_means_nhanes['BMXWAIST'],
        height_mean=wa_means_nhanes['BMXHT'],
        weight_mean=wa_means_nhanes['BMXWT'],
        waist_std=wa_stds_nhanes['BMXWAIST'],
        height_std=wa_stds_nhanes['BMXHT'],
        weight_std=wa_stds_nhanes['BMXWT'],
    )
    nat_ansur = an.filter_ansur_by_profile(
        ansur,
        waist_mean=nat_means_nhanes['BMXWAIST'],
        height_mean=nat_means_nhanes['BMXHT'],
        weight_mean=nat_means_nhanes['BMXWT'],
        waist_std=nat_stds_nhanes['BMXWAIST'],
        height_std=nat_stds_nhanes['BMXHT'],
        weight_std=nat_stds_nhanes['BMXWT'],
    )
    print('\nANSUR II proxy t-test results:')
    ansur_results = an.run_ttests(wa_ansur, nat_ansur, struct_cols)
    print(ansur_results)

    an.plot_significance_results(
        nhanes_results, ansur_results,
        title='RQ4: Statistical Significance of WA vs. U.S. Differences',
        filename=os.path.join(PLOTS_DIR, 'rq4_significance.png')
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Load all datasets and run the full analysis pipeline in order:
      1. EDA for each individual dataset
      2. Joined EDA (NHANES + PUMS) — returns weighted means and stds
      3. RQ1: ANSUR II structural measurement proxy comparison
      4. RQ2: Ann Taylor size distribution alignment
      5. RQ4: Statistical significance testing
    """
    setup_plots_dir()

    print('Loading datasets...')
    ansur = an.load_ansur(DATA_DIR)
    nhanes = an.load_nhanes(DATA_DIR)
    pums_wa = an.load_pums_wa(DATA_DIR)
    pums_national = an.load_pums_national(DATA_DIR)
    classic_tops, classic_bottoms, petite_tops, petite_bottoms = (
        an.load_anntaylor()
    )
    print('All datasets loaded successfully.\n')

    # Individual dataset EDA
    run_ansur_eda(ansur)
    run_nhanes_eda(nhanes)
    run_pums_eda(pums_wa, pums_national)
    run_anntaylor_eda(classic_tops, classic_bottoms,
                      petite_tops, petite_bottoms)

    # Joined EDA — returns weighted means and stds for downstream use
    wa_means, wa_stds, nat_means, nat_stds = run_joined_eda(
        nhanes, pums_wa, pums_national
    )

    # Recompute demographic weight vectors for RQ2 and RQ4
    wa_weights = an.compute_demographic_weights(pums_wa)
    nat_weights = an.compute_demographic_weights(pums_national)

    # Research question analyses
    run_rq1_ansur_proxy(ansur, wa_means, wa_stds, nat_means, nat_stds)
    run_rq2_size_alignment(nhanes, classic_tops, classic_bottoms,
                           petite_tops, petite_bottoms,
                           wa_weights, nat_weights)
    run_rq4_significance(nhanes, ansur, wa_weights, nat_weights,
                         wa_means, wa_stds, nat_means, nat_stds)

    print('\nAnalysis complete. All plots saved to:', PLOTS_DIR)


if __name__ == '__main__':
    main()
