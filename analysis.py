"""
Analysis helper module for CSE 163 Final Project.
Contains functions for loading, filtering, summarizing, and
visualizing the ANSUR II, NHANES, ACS PUMS, and Ann Taylor datasets.

Organized into four sections:
  1. Data Loading     — load each dataset from disk
  2. Data Filtering   — filter/transform datasets for analysis
  3. Computations     — summaries, demographic weighting, statistical tests
  4. Visualizations   — all plot-generating functions
"""
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_ansur(data_dir):
    """
    Load and return the ANSUR II Women dataset as a DataFrame.
    Each row is one U.S. Army servicewoman; all measurements in mm.
    """
    path = os.path.join(data_dir, 'ANSUR_II_FEMALE.csv')
    return pd.read_csv(path)


def load_nhanes(data_dir):
    """
    Load and return the Penn State NHANES 2015-2018 Women dataset.
    Pre-merged CSV already filtered to women aged 20+.
    Key columns: BMXWAIST, BMXHT, BMXWT (mm), RIDRETH3, WTMEC2YR.
    """
    path = os.path.join(
        data_dir, 'NHANES15-18_womenAge20YearsAndOver.csv'
    )
    return pd.read_csv(path)


def load_pums_wa(data_dir):
    """
    Load and return the ACS PUMS 2024 5-Year Washington state person
    file, filtered to female respondents (SEX == 2).
    """
    path = os.path.join(data_dir, 'psam_p53.csv')
    df = pd.read_csv(path, usecols=['SEX', 'RAC1P', 'HISP', 'PWGTP'])
    return df[df['SEX'] == 2].reset_index(drop=True)


def load_pums_national(data_dir):
    """
    Load and return the ACS PUMS 2024 5-Year national person file by
    concatenating the four part files (psam_pusa-d.csv).
    Filtered to female respondents (SEX == 2).
    """
    parts = ['psam_pusa.csv', 'psam_pusb.csv',
             'psam_pusc.csv', 'psam_pusd.csv']
    dfs = []
    for part in parts:
        path = os.path.join(data_dir, part)
        dfs.append(
            pd.read_csv(path, usecols=['SEX', 'RAC1P', 'HISP', 'PWGTP'])
        )
    df = pd.concat(dfs, ignore_index=True)
    return df[df['SEX'] == 2].reset_index(drop=True)


def load_anntaylor():
    """
    Return Ann Taylor's Classic and Petite size guides as four
    DataFrames: classic_tops, classic_bottoms, petite_tops,
    petite_bottoms. Data manually transcribed from the size chart
    modal at anntaylor.com in May 2026. All measurements are
    converted from inches to cm (* 2.54) and stored alongside the
    original inch columns.
    """
    classic_tops = pd.DataFrame({
        'size_label': ['XXS', 'XS', 'XS', 'S', 'S',
                       'M', 'M', 'L', 'L', 'XL', 'XXL'],
        'numeric_size': ['00', '0', '2', '4', '6',
                         '8', '10', '12', '14', '16', '18'],
        'bust_in': [31.5, 32.5, 33.5, 34.5, 35.5,
                    36.5, 37.5, 39.0, 40.5, 42.5, 44.5],
        'sleeve_in': [29.75, 30.0, 30.25, 30.5, 30.75,
                      31.0, 31.25, 31.625, 32.0, 32.375, 32.75],
        'waist_in': [24.5, 25.5, 26.5, 27.5, 28.5,
                     29.5, 30.5, 32.0, 33.5, 35.5, 37.5],
    })

    classic_bottoms = pd.DataFrame({
        'size_label': ['XXS', 'XS', 'XS', 'S', 'S',
                       'M', 'M', 'L', 'L', 'XL', 'XXL'],
        'numeric_size': ['00', '0', '2', '4', '6',
                         '8', '10', '12', '14', '16', '18'],
        'waist_regular_in': [24.5, 25.5, 26.5, 27.5, 28.5,
                             29.5, 30.5, 32.0, 33.5, 35.5, 37.5],
        'waist_curvy_in': [24.0, 25.0, 26.0, 27.0, 28.0,
                           29.0, 30.0, 31.5, 33.0, 35.0, 37.0],
        'hip_regular_in': [34.5, 35.5, 36.5, 37.5, 38.5,
                           39.5, 40.5, 42.0, 43.5, 45.5, 47.5],
        'hip_curvy_in': [36.0, 37.0, 38.0, 39.0, 40.0,
                         41.0, 42.0, 43.5, 45.0, 47.0, 49.0],
    })

    petite_tops = pd.DataFrame({
        'size_label': ['XXSP', 'XSP', 'XSP', 'SP', 'SP',
                       'MP', 'MP', 'LP', 'LP', 'XLP'],
        'numeric_size': ['00P', '0P', '2P', '4P', '6P',
                         '8P', '10P', '12P', '14P', '16P'],
        'bust_in': [30.5, 31.5, 32.5, 33.5, 34.5,
                    35.5, 36.5, 38.0, 39.5, 41.5],
        'sleeve_in': [28.25, 28.5, 28.75, 29.0, 29.25,
                      29.5, 29.75, 30.125, 30.5, 30.875],
        'waist_in': [23.5, 24.5, 26.5, 26.5, 27.5,
                     28.5, 29.5, 31.0, 32.5, 34.5],
    })

    petite_bottoms = pd.DataFrame({
        'size_label': ['XXSP', 'XSP', 'XSP', 'SP', 'SP',
                       'MP', 'MP', 'LP', 'LP', 'XLP', 'XXLP'],
        'numeric_size': ['00P', '0P', '2P', '4P', '6P',
                         '8P', '10P', '12P', '14P', '16P', '18P'],
        'waist_regular_in': [23.5, 24.5, 25.5, 26.5, 27.5,
                             28.5, 29.5, 31.0, 32.5, 34.5, 36.5],
        'waist_curvy_in': [23.0, 24.0, 25.0, 26.0, 27.0,
                           28.0, 29.0, 30.5, 32.0, 34.0, 36.0],
        'hip_regular_in': [33.5, 34.5, 35.5, 36.5, 37.5,
                           38.5, 39.5, 41.0, 42.5, 44.0, 46.5],
        'hip_curvy_in': [35.0, 36.0, 37.0, 38.0, 39.0,
                         40.0, 41.0, 42.5, 44.0, 45.5, 48.0],
    })

    # Convert all inch measurements to cm and store as new columns
    for col in ['bust_in', 'sleeve_in', 'waist_in']:
        classic_tops[col.replace('_in', '_cm')] = (
            classic_tops[col] * 2.54
        )
        petite_tops[col.replace('_in', '_cm')] = (
            petite_tops[col] * 2.54
        )
    for col in ['waist_regular_in', 'waist_curvy_in',
                'hip_regular_in', 'hip_curvy_in']:
        classic_bottoms[col.replace('_in', '_cm')] = (
            classic_bottoms[col] * 2.54
        )
        petite_bottoms[col.replace('_in', '_cm')] = (
            petite_bottoms[col] * 2.54
        )

    return classic_tops, classic_bottoms, petite_tops, petite_bottoms


# ---------------------------------------------------------------------------
# 2. Data Filtering
# ---------------------------------------------------------------------------

def filter_ansur_by_profile(ansur, waist_mean, height_mean, weight_mean,
                            waist_std, height_std, weight_std):
    """
    Filter ANSUR II to women whose waist, height, and weight all fall
    within one standard deviation of the given means.

    Used as a proxy for a target demographic when ANSUR II lacks a
    race/ethnicity variable. Means and stds are computed from the
    NHANES + PUMS demographic-weighted analysis.

    waist_mean and waist_std are in mm.
    height_mean and height_std are in mm.
    weight_mean and weight_std are in kg.
    Returns a filtered DataFrame.
    """
    ansur = ansur.copy()
    # Convert ANSUR II stature from meters to mm for comparison
    ansur['stature_mm'] = ansur['stature_m'] * 1000

    # Keep only women within +/-1 std of all three body measurements
    filtered = ansur[
        (ansur['waistcircumference'] >= waist_mean - waist_std) &
        (ansur['waistcircumference'] <= waist_mean + waist_std) &
        (ansur['stature_mm'] >= height_mean - height_std) &
        (ansur['stature_mm'] <= height_mean + height_std) &
        (ansur['weight_kg'] >= weight_mean - weight_std) &
        (ansur['weight_kg'] <= weight_mean + weight_std)
    ]
    return filtered


# ---------------------------------------------------------------------------
# 3. Computations
# ---------------------------------------------------------------------------

def summarize_dataset(df, cols):
    """
    Return a summary dict for the given columns in df.
    Quantitative columns get a seven-number summary (mean, std, min,
    25%, 50%, 75%, max) via DataFrame.describe(). Categorical columns
    get value counts.
    """
    quant_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols
                if not pd.api.types.is_numeric_dtype(df[c])]
    results = {}
    if quant_cols:
        results['quantitative'] = df[quant_cols].describe().loc[
            ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
        ]
    if cat_cols:
        results['categorical'] = {
            c: df[c].value_counts() for c in cat_cols
        }
    return results


def compute_demographic_weights(pums_df):
    """
    Compute weighted racial/ethnic proportions for female respondents
    in a PUMS DataFrame. Uses PWGTP as person weight. Hispanic origin
    takes priority over RAC1P per Census Bureau convention, producing
    a unified race/ethnicity label. Returns a Series of proportions
    that sum to 1.0, indexed by race/ethnicity label.
    """
    race_map = {
        1: 'White', 2: 'Black or African American',
        3: 'Native American', 4: 'Alaska Native',
        5: 'Native American/Alaska Native combo',
        6: 'Asian', 7: 'Native Hawaiian/Pacific Islander',
        8: 'Other', 9: 'Two or More Races',
    }
    df = pums_df.copy()
    df['race_eth'] = df.apply(
        lambda r: 'Hispanic' if r['HISP'] != 1
        else race_map.get(r['RAC1P'], 'Other'),
        axis=1
    )
    grouped = df.groupby('race_eth')['PWGTP'].sum()
    return grouped / grouped.sum()


def compute_weighted_means(df, race_col, meas_cols, weight_series):
    """
    Compute demographic-weighted mean for each measurement column.
    Groups df by race_col, computes group means, then applies
    weight_series (a Series indexed by race label). Returns a Series
    of weighted means indexed by measurement name.
    """
    group_means = df.groupby(race_col)[meas_cols].mean()
    # Align weights to only the groups present in both datasets
    common = group_means.index.intersection(weight_series.index)
    aligned_weights = weight_series.loc[common]
    aligned_weights = aligned_weights / aligned_weights.sum()
    weighted = group_means.loc[common].multiply(aligned_weights, axis=0)
    return weighted.sum()


def compute_weighted_std(df, race_col, meas_cols, weight_series):
    """
    Compute demographic-weighted standard deviation for each
    measurement column, using the same weighting approach as
    compute_weighted_means. Returns a Series of weighted standard
    deviations indexed by measurement name.
    """
    # Weighted mean needed for variance calculation
    weighted_means = compute_weighted_means(
        df, race_col, meas_cols, weight_series
    )

    # Align weights to groups present in both datasets
    group_means = df.groupby(race_col)[meas_cols].mean()
    common = group_means.index.intersection(weight_series.index)
    aligned_weights = weight_series.loc[common]
    aligned_weights = aligned_weights / aligned_weights.sum()

    # Assign each row its group's demographic weight, then compute
    # weighted variance: sum(w * (x - weighted_mean)^2) / sum(w)
    df = df.copy()
    df = df[df[race_col].isin(common)]
    df['_weight'] = df[race_col].map(aligned_weights)

    weighted_vars = {}
    for col in meas_cols:
        diff_sq = (df[col] - weighted_means[col]) ** 2
        weighted_vars[col] = (
            (df['_weight'] * diff_sq).sum() / df['_weight'].sum()
        )

    return pd.Series(weighted_vars).apply(lambda v: v ** 0.5)


def assign_size_distribution(nhanes, size_guide_df,
                             waist_col, weight_series):
    """
    Compute the weighted proportion of women falling into each Ann
    Taylor size bucket based on waist circumference. An 'Above Range'
    bucket captures women whose waist exceeds the largest published size.

    nhanes: NHANES DataFrame with BMXWAIST (mm) and race_label columns.
    size_guide_df: Ann Taylor size guide with waist_col values in cm.
    waist_col: name of the waist column in size_guide_df (cm).
    weight_series: PUMS demographic weights indexed by race label.
    Returns a Series of weighted proportions indexed by size label.
    """
    # Assign each NHANES row a demographic weight via its race_label
    df = nhanes.copy()
    df = df[df['race_label'].isin(weight_series.index)].copy()
    df['_weight'] = df['race_label'].map(weight_series)

    # Convert NHANES waist from mm to cm to match the size guide
    df['waist_cm'] = df['BMXWAIST'] / 10

    # Sort size guide by waist value to build sequential size buckets
    size_guide = size_guide_df[['numeric_size', waist_col]].copy()
    size_guide = size_guide.sort_values(waist_col).reset_index(drop=True)

    # Assign women to size buckets; last defined bucket uses its own
    # interval width rather than extending to infinity
    results = {}
    for i, row in size_guide.iterrows():
        lower = row[waist_col]
        if i + 1 < len(size_guide):
            upper = size_guide.loc[i + 1, waist_col]
        else:
            upper = row[waist_col] + (
                row[waist_col] - size_guide.loc[i - 1, waist_col]
            )
        mask = (df['waist_cm'] >= lower) & (df['waist_cm'] < upper)
        results[row['numeric_size']] = (
            df.loc[mask, '_weight'].sum() / df['_weight'].sum()
        )

    # Women whose waist exceeds Ann Taylor's maximum published size
    largest_waist = size_guide[waist_col].max()
    above_mask = df['waist_cm'] >= largest_waist
    results['Above\nRange'] = (
        df.loc[above_mask, '_weight'].sum() / df['_weight'].sum()
    )

    return pd.Series(results)


def resample_by_weights(df, race_col, weight_series, n=1000,
                        random_state=42):
    """
    Resample df to match the target demographic proportions in
    weight_series. Draws n total rows, with each racial/ethnic group
    contributing proportionally to its weight. Used to construct
    WA-representative and national-representative samples for t-testing.
    Returns a resampled DataFrame of approximately size n.
    """
    samples = []
    df = df[df[race_col].isin(weight_series.index)].copy()
    aligned = weight_series.reindex(df[race_col].unique()).dropna()
    aligned = aligned / aligned.sum()

    for group, proportion in aligned.items():
        group_df = df[df[race_col] == group]
        n_sample = max(1, int(round(proportion * n)))
        samples.append(
            group_df.sample(
                n=n_sample, replace=True,
                random_state=random_state
            )
        )
    return pd.concat(samples, ignore_index=True)


def run_ttests(wa_sample, nat_sample, meas_cols, alpha=0.05):
    """
    Run independent samples t-tests between wa_sample and nat_sample
    for each measurement column in meas_cols. Computes Cohen's d
    effect size alongside each p-value. Returns a DataFrame with one
    row per measurement summarizing the test results.
    """
    results = []
    for col in meas_cols:
        wa_vals = wa_sample[col].dropna()
        nat_vals = nat_sample[col].dropna()
        t_stat, p_val = stats.ttest_ind(wa_vals, nat_vals)

        # Cohen's d = difference in means / pooled standard deviation
        pooled_std = (
            (wa_vals.std() ** 2 + nat_vals.std() ** 2) / 2
        ) ** 0.5
        cohens_d = (wa_vals.mean() - nat_vals.mean()) / pooled_std

        results.append({
            'measurement': col,
            'wa_mean': wa_vals.mean(),
            'nat_mean': nat_vals.mean(),
            'difference': wa_vals.mean() - nat_vals.mean(),
            't_statistic': round(t_stat, 4),
            'p_value': round(p_val, 4),
            'significant': p_val < alpha,
            'cohens_d': round(cohens_d, 4),
        })
    return pd.DataFrame(results).set_index('measurement')


# ---------------------------------------------------------------------------
# 4. Visualizations
# ---------------------------------------------------------------------------

def add_bar_labels(ax):
    """
    Add value labels on top of each bar in a seaborn bar chart.
    Values are rounded to 2 decimal places for readability.
    Called by most plot functions below after sns.barplot().
    """
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8, padding=3)


def plot_measurement_distributions(df, cols, title, filename):
    """
    Save a grid of histograms for each column in cols from df.
    Each subplot shows the frequency distribution of one body
    measurement. All x-axes are labelled in mm.
    """
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=30, color='steelblue',
                edgecolor='white', alpha=0.85)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel('mm')
        ax.set_ylabel('Count')
    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved: {os.path.basename(filename)}')


def plot_boxplots(df, cols, title, filename):
    """
    Save a grid of box plots for each column in cols from df.
    Each subplot shows the spread and outliers of one body
    measurement more clearly than a histogram alone.
    """
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.boxplot(y=df[col], ax=ax, color='steelblue')
        ax.set_title(col, fontsize=10)
        ax.set_ylabel('mm')
    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved: {os.path.basename(filename)}')


def plot_grouped_bars(df, group_col, value_cols, title, filename):
    """
    Save a grouped bar chart comparing value_cols across group_col
    categories. Reshapes the DataFrame from wide to long format via
    melt() before passing to seaborn so that each value column
    appears as a separate hue.
    """
    if group_col in df.columns:
        means = df.groupby(group_col)[value_cols].mean().reset_index()
        melted = means.melt(id_vars=group_col, var_name='Measurement',
                            value_name='Mean (mm)')
        x_col = group_col
    else:
        melted = df.melt(id_vars=value_cols[0], var_name='Group',
                         value_name='Mean (mm)')
        x_col = value_cols[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=melted, x=x_col, y='Mean (mm)',
        hue='Measurement' if group_col in df.columns else 'Group',
        ax=ax, palette='muted'
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=25)
    add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved: {os.path.basename(filename)}')


def plot_demographic_comparison(wa_weights, nat_weights, title, filename):
    """
    Save a grouped bar chart comparing racial/ethnic proportions
    between Washington state and the U.S. national population.
    wa_weights and nat_weights are Series indexed by race/ethnicity
    label, as returned by compute_demographic_weights().
    """
    all_groups = wa_weights.index.union(nat_weights.index)
    wa = wa_weights.reindex(all_groups, fill_value=0)
    nat = nat_weights.reindex(all_groups, fill_value=0)

    # Reshape to long format for seaborn hue grouping
    comp = pd.DataFrame({'Washington State': wa, 'U.S. National': nat})
    comp.index.name = 'Race/Ethnicity'
    comp = comp.reset_index()
    melted = comp.melt(id_vars='Race/Ethnicity',
                       var_name='Region', value_name='Proportion')

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=melted, x='Race/Ethnicity', y='Proportion',
                hue='Region', ax=ax, palette='muted')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('')
    ax.set_ylabel('Proportion of Female Population')
    ax.tick_params(axis='x', rotation=30)
    add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved: {os.path.basename(filename)}')


def plot_divergence_bar(series, title, xlabel, ylabel, filename):
    """
    Save a bar chart from a pre-computed divergence Series, sorted
    in descending order. Used to visualize the absolute demographic
    proportion difference between Washington state and the U.S.
    national female population.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=series.index, y=series.values, color='salmon', ax=ax)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45)
    add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved: {os.path.basename(filename)}')


def plot_wa_vs_national(comparison_df, title, filename):
    """
    Save a grouped bar chart comparing mean body measurements between
    Washington state and the U.S. national population. comparison_df
    should have 'Washington State' and 'U.S. National' columns with
    measurement names as the index.
    """
    comp_reset = comparison_df.reset_index().rename(
        columns={'index': 'Measurement'}
    )
    # Reshape from wide to long format so seaborn renders side-by-side
    # bars for Washington state vs. U.S. national
    melted = comp_reset.melt(
        id_vars='Measurement',
        var_name='Region',
        value_name='Mean (mm)'
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=melted, x='Measurement', y='Mean (mm)',
                hue='Region', ax=ax, palette='muted')
    ax.set_title(title, fontsize=13)
    add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved: {os.path.basename(filename)}')


def plot_measurement_differences(comparison_df, title, filename):
    """
    Save a horizontal bar chart showing WA minus U.S. national for
    each structural measurement. Bars extending left indicate WA is
    smaller than the national average; bars extending right indicate
    WA is larger. Used for RQ1 to show differences on a common scale
    when measurements span very different absolute magnitudes.
    """
    diff = comparison_df['Washington State'] - comparison_df['U.S. National']
    diff = diff.sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ['steelblue' if v < 0 else 'salmon' for v in diff]
    ax.barh(diff.index, diff.values, color=bar_colors, edgecolor='white')

    # Reference line at zero
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')

    # Value labels outside each bar
    for i, v in enumerate(diff.values):
        ax.text(v - 0.03, i, f'{v:.2f} mm',
                va='center', ha='right', fontsize=9)

    # Padding so labels are not clipped on the left
    ax.set_xlim(diff.min() * 1.2, 0.3)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('WA minus U.S. National (mm)')
    ax.set_ylabel('Measurement')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved: {os.path.basename(filename)}')


def plot_anntaylor_comparison(classic_df, petite_df, title, filename):
    """
    Save a grouped bar chart comparing Ann Taylor's Classic vs. Petite
    size guides. classic_df and petite_df must have a 'size_label'
    column and measurement columns ending in '_cm'. Only _cm columns
    are plotted so that inch columns are excluded automatically.
    """
    cm_cols_c = [c for c in classic_df.columns if c.endswith('_cm')]
    classic_melted = classic_df[['size_label'] + cm_cols_c].melt(
        id_vars='size_label',
        var_name='Measurement',
        value_name='Size (cm)'
    )
    classic_melted['Fit'] = 'Classic'

    cm_cols_p = [c for c in petite_df.columns if c.endswith('_cm')]
    petite_melted = petite_df[['size_label'] + cm_cols_p].melt(
        id_vars='size_label',
        var_name='Measurement',
        value_name='Size (cm)'
    )
    petite_melted['Fit'] = 'Petite'

    combined = pd.concat([classic_melted, petite_melted], ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=combined, x='Measurement', y='Size (cm)',
                hue='Fit', ax=ax, palette='muted')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved: {os.path.basename(filename)}')


def plot_size_distribution(comparison_df, title, filename):
    """
    Save a grouped bar chart comparing the proportion of Washington
    state vs. U.S. national women in each Ann Taylor size bucket.
    comparison_df should have 'Washington State' and 'U.S. National'
    columns with Ann Taylor size labels as the index.
    """
    comp_reset = comparison_df.reset_index().rename(
        columns={'index': 'Size'}
    )
    # Reshape from wide to long format for seaborn hue grouping
    melted = comp_reset.melt(
        id_vars='Size', var_name='Region', value_name='Proportion'
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x='Size', y='Proportion',
                hue='Region', ax=ax, palette='muted')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Ann Taylor Size')
    ax.set_ylabel('Proportion of Female Population')
    add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved: {os.path.basename(filename)}')


def plot_significance_results(nhanes_results, ansur_results,
                              title, filename):
    """
    Save a horizontal bar chart of Cohen's d effect sizes for all
    measurements, colored by statistical significance. Steelblue
    bars are significant (p < 0.05); salmon bars are not. Dashed
    reference lines mark the conventional small (0.2), medium (0.5),
    and large (0.8) effect size benchmarks.
    """
    from matplotlib.patches import Patch

    # Combine NHANES and ANSUR II results and sort by effect size
    combined = pd.concat([nhanes_results, ansur_results])
    combined = combined.sort_values('cohens_d')

    bar_colors = [
        'steelblue' if sig else 'salmon'
        for sig in combined['significant']
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(combined.index, combined['cohens_d'],
            color=bar_colors, edgecolor='white')

    # Dashed reference lines at standard effect size benchmarks
    for d in [0.2, 0.5, 0.8]:
        ax.axvline(x=-d, color='gray', linewidth=0.7,
                   linestyle='--', alpha=0.6)
        ax.axvline(x=d, color='gray', linewidth=0.7,
                   linestyle='--', alpha=0.6)
    ax.axvline(x=0, color='black', linewidth=0.8)

    # Cohen's d value labels on each bar
    for i, (idx, row) in enumerate(combined.iterrows()):
        ax.text(row['cohens_d'] - 0.002, i,
                f'd={row["cohens_d"]:.4f}',
                va='center', ha='right', fontsize=8.5)

    legend_elements = [
        Patch(facecolor='steelblue', label='Significant (p < 0.05)'),
        Patch(facecolor='salmon', label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Cohen's d (WA minus U.S. National)")
    ax.set_ylabel('Measurement')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved: {os.path.basename(filename)}')
