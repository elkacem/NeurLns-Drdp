import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, ttest_rel
import os
import sys

# Configure plotting style for research papers
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze DRDP Solver Results for Research Paper")

    # Allow passing files as arguments: --full results_full.csv --random results_random.csv
    parser.add_argument("--full", type=str, help="Path to 'full' mode results", default="results_full.csv")
    parser.add_argument("--random", type=str, help="Path to 'random' mode results", default="results_random.csv")
    parser.add_argument("--no_learning", type=str, help="Path to 'no_learning' mode results", default="results_nolearn.csv")
    parser.add_argument("--no_cpsat", type=str, help="Path to 'no_cpsat' mode results", default="results_nocpsat.csv")
    parser.add_argument("--classical", type=str, help="Path to classical metaheuristic results (optional)", default=None)

    parser.add_argument("--out_dir", type=str, default="analysis_output", help="Directory to save plots and tables")
    return parser.parse_args()

def load_data(args):
    """Loads and standardizes data from valid files."""
    file_map = {
        'NeuroCP-LNS (Full)': args.full,
        'Random-LNS': args.random,
        'No Learning': args.no_learning,
        'No CP-SAT': args.no_cpsat,
    }

    if args.classical:
        file_map['Classical'] = args.classical

    dfs = []
    print(f"{'Method':<25} | {'Status':<10} | {'Path'}")
    print("-" * 60)

    for method, path in file_map.items():
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Standardize columns just in case
                df.columns = [c.strip() for c in df.columns]

                # Check for essential columns
                required = ['Graph', 'Cost', 'Time']
                if not all(col in df.columns for col in required):
                    print(f"{method:<25} | {'SKIPPED':<10} | Missing columns in {path}")
                    continue

                # Overwrite 'Method' column with our standardized label
                df['Method'] = method

                # Handle infeasible results (Cost = -1) by setting to NaN for stats
                df.loc[df['Cost'] == -1, 'Cost'] = np.nan

                dfs.append(df)
                print(f"{method:<25} | {'LOADED':<10} | {path} ({len(df)} rows)")
            except Exception as e:
                print(f"{method:<25} | {'ERROR':<10} | {e}")
        else:
            print(f"{method:<25} | {'MISSING':<10} | {path if path else 'None'}")

    if not dfs:
        raise ValueError("No result files loaded. Please run the solver to generate .csv files first.")

    return pd.concat(dfs, ignore_index=True)

def compute_metrics(df):
    """computes relative gaps and best known solutions."""
    # Pivot to find best known solution per graph across ALL methods
    # We use min() ignoring NaNs.
    pivot = df.pivot_table(index='Graph', columns='Method', values='Cost', aggfunc='min')

    # Calculate Best Known Solution (BKS) for each graph
    bks = pivot.min(axis=1)

    # creates a copy to avoid SettingWithCopy warning on slice
    df_aug = df.copy()

    # Map BKS back to the main dataframe
    df_aug['BKS'] = df_aug['Graph'].map(bks)

    # Calculate Gap (%)
    # Gap = (Cost - BKS) / BKS * 100
    df_aug['Gap (%)'] = ((df_aug['Cost'] - df_aug['BKS']) / df_aug['BKS']) * 100.0

    # Calculate Is_Best (Boolean)
    # We use a small epsilon for float comparison safety, though costs are likely int
    df_aug['Is_Best'] = (df_aug['Cost'] <= df_aug['BKS'] + 1e-9)

    return df_aug

def generate_table_statistics(df, out_dir):
    """Generates summary tables (Table 1 & 2 in paper)."""

    # Aggregation
    # Mean Cost, Std Cost, Mean Gap, Mean Time, Success Rate
    summary = df.groupby('Method').agg(
        Avg_Cost=('Cost', 'mean'),
        Std_Cost=('Cost', 'std'),
        Avg_Gap=('Gap (%)', 'mean'),
        Avg_Time=('Time', 'mean'),
        Std_Time=('Time', 'std'),
        Feasible_Count=('Cost', 'count'), # Count non-NaNs
        Best_Count=('Is_Best', 'sum'),
        Total_Count=('Graph', 'nunique') # Should be same for all if aligned
    ).reset_index()

    # Calculate Success Rate %
    summary['Success_Rate (%)'] = (summary['Best_Count'] / summary['Total_Count']) * 100

    # Sort by Avg Gap (Ascending) -> best method top
    summary = summary.sort_values(by='Avg_Gap')

    # Save to CSV
    table_path = os.path.join(out_dir, 'table_summary.csv')
    summary.to_csv(table_path, index=False, float_format="%.2f")
    print(f"\n[INFO] Summary table saved to {table_path}")

    # Print simplified view
    print("\n=== Table 1: Algorithm Performance Summary ===")
    print(summary[['Method', 'Avg_Gap', 'Avg_Time', 'Success_Rate (%)']].to_string(index=False))

    return summary

def run_significance_tests(df, out_dir, baseline='NeuroCP-LNS (Full)'):
    """Runs Wilcoxon signed-rank tests against the baseline."""

    pivot_cost = df.pivot_table(index='Graph', columns='Method', values='Cost')

    if baseline not in pivot_cost.columns:
        print(f"\n[WARN] Baseline method '{baseline}' not in results. Skipping statistical tests.")
        return

    test_results = []
    print(f"\n=== Statistical Significance (vs {baseline}) ===")

    for method in pivot_cost.columns:
        if method == baseline:
            continue

        # Get paired data, dropping instances where either method failed (NaN)
        data = pivot_cost[[baseline, method]].dropna()
        x = data[baseline]
        y = data[method]

        if len(x) < 2:
            p_val = np.nan
            stat = np.nan
            verdict = "N/A"
        else:
            # Check if differences exist (Wilcoxon fails if all diffs are zero)
            if np.allclose(x, y):
                p_val = 1.0
                stat = 0.0
                verdict = "Identical"
            else:
                try:
                    stat, p_val = wilcoxon(x, y, alternative='two-sided')
                    verdict = "Significant" if p_val < 0.05 else "Not Significant"
                except Exception:
                    p_val = 1.0; verdict="Error"

        test_results.append({
            'Comparison': f"{baseline} vs {method}",
            'N_Samples': len(x),
            'P-Value': p_val,
            'Verdict (p<0.05)': verdict
        })
        print(f"{baseline} vs {method:<20} | p={p_val:.4f} | {verdict}")

    pd.DataFrame(test_results).to_csv(os.path.join(out_dir, 'statistical_tests.csv'), index=False)

def plot_results(df, out_dir):
    """Generates publication-quality figures."""

    # 1. Box Plot: Gap Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Method', y='Gap (%)', data=df, showfliers=False, palette="viridis")
    sns.stripplot(x='Method', y='Gap (%)', data=df, color='black', alpha=0.3, size=3)
    plt.title("Optimality Gap Distribution by Method")
    plt.ylabel("Gap to Best Known Solution (%)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_gap_boxplot.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "plot_gap_boxplot.pdf"))
    plt.close()

    # 2. Scatter Plot: Time vs Gap
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Time', y='Gap (%)', hue='Method', style='Method', data=df, alpha=0.7, s=80)
    plt.xscale('log') # Runtime usually varies by orders of magnitude
    plt.title("Trade-off: Runtime vs. Solution Quality")
    plt.xlabel("Runtime (s) [Log Scale]")
    plt.ylabel("Gap to Best Known (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_time_vs_quality.png"), dpi=300)
    plt.close()

    # 3. Bar Chart: Success Rate (best solutions found)
    # Re-calculate summary just for this plot
    summary = df.groupby('Method').agg(
        Success_Rate=('Is_Best', 'mean')
    ).reset_index()
    summary['Success_Rate'] *= 100

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x='Method', y='Success_Rate', data=summary, palette='magma')
    plt.title("Success Rate (% Instances with Best Solution Found)")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 105)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.1f%%')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_success_rate.png"), dpi=300)
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Loading Data ===")
    try:
        raw_df = load_data(args)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    print("\n=== Computing Metrics ===")
    processed_df = compute_metrics(raw_df)

    # Save full processed data for debugging
    processed_df.to_csv(os.path.join(args.out_dir, "full_processed_results.csv"), index=False)

    print("\n=== Generating Tables ===")
    generate_table_statistics(processed_df, args.out_dir)

    print("\n=== Running Statistical Tests ===")
    run_significance_tests(processed_df, args.out_dir)

    print("\n=== Generating Plots ===")
    plot_results(processed_df, args.out_dir)

    print(f"\n[DONE] All results saved to directory: {args.out_dir}")

if __name__ == "__main__":
    main()
