import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import os
import sys

# --- NEURIPS / AAAI Publication Style Configuration ---
# Use standard LaTeX-like fonts and sizes
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": False,    # True requires latex installed, False uses mostly-compatible matplotlib internals
    "font.size": 10,         # Match typical paper font size (9-10pt)
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
    "pdf.fonttype": 42,      # TrueType fonts for editing in Illustrator/Inkscape
    "ps.fonttype": 42
})

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze DRDP Solver Results for Research Paper")

    # Auto-detection defaults
    default_results = "results.txt.csv"

    parser.add_argument("--results", type=str, help="Path to main solver results", default=default_results)
    parser.add_argument("--classical", type=str, help="Path to classical results (for comparison)", default=None)

    parser.add_argument("--out_dir", type=str, default="analysis_output", help="Directory for paper artifacts")
    return parser.parse_args()

def load_and_clean_data(args):
    """Loads CSVs, normalizes headers, handles infeasible (-1) entries."""
    # Define method names for paper
    file_map = {
        'NeuroCP-LNS': args.results,
    }

    if args.classical:
        file_map['Metaheuristic-Ref'] = args.classical

    dfs = []
    print(f"{'Method':<25} | {'Path':<40} | {'Status'}")
    print("-" * 80)

    valid_methods = []

    for method, path in file_map.items():
        # Exclusion logic
        if args.exclude and any(exc.lower() in method.lower() for exc in args.exclude):
            print(f"{method:<25} | {'-- excluded --':<40} | SKIPPED (User request)")
            continue

        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns] # Remove extra spaces

                # Check required columns
                if not {'Graph', 'Cost', 'Time'}.issubset(df.columns):
                    print(f"{method:<25} | {path:<40} | SKIPPED (Missing cols)")
                    continue

                # Filter out infeasible runs (-1) or invalid
                df['Feasible'] = df['Cost'] > 0
                # Set invalid costs to NaN for stats calculation
                df.loc[~df['Feasible'], 'Cost'] = np.nan
                df.loc[~df['Feasible'], 'Time'] = np.nan

                df['Method'] = method
                dfs.append(df)
                valid_methods.append(method)
                print(f"{method:<25} | {path:<40} | OK ({len(df)} rows)")
            except Exception as e:
                print(f"{method:<25} | {path:<40} | ERROR: {e}")
        else:
            print(f"{method:<25} | {str(path):<40} | MISSING")

    if not dfs:
        raise ValueError("No valid result files found! Please run the experiments first.")

    return pd.concat(dfs, ignore_index=True), valid_methods

def compute_comparative_metrics(df):
    """Computes Best Known Solution (BKS) and Optimality Gaps."""

    # 1. Pivot to find Min Cost per Graph across all methods -> BKS
    pivot = df.pivot_table(index='Graph', columns='Method', values='Cost', aggfunc='min')
    bks = pivot.min(axis=1) # Series: Graph -> MinCost

    df = df.copy()
    df['BKS'] = df['Graph'].map(bks)

    # 2. Compute Gap: (Cost - BKS) / BKS * 100
    # Handle NaNs (infeasible) -> Gap remains NaN
    df['Gap'] = ((df['Cost'] - df['BKS']) / df['BKS']) * 100.0

    # 3. Identify if 'Best Found' (within numeric tolerance)
    df['Is_Best'] = (df['Cost'] <= df['BKS'] + 1e-6)

    print(f"\n[INFO] Processed {df['Graph'].nunique()} unique instances.")
    return df

def generate_latex_table(stats_df, out_path, caption="Experimental Results"):
    """Generates a professional Booktabs-style LaTeX table."""

    # Sort by Average Gap to put best methods first (or ensure Ours is highlighted)
    stats_df = stats_df.sort_values(by='Mean Gap %')

    latex_str = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{tab:results}",
        "\\begin{tabular}{lrrrrr}", # l c c c c
        "\\toprule",
        "Method & Cost (Mean $\\pm$ Std) & Time (s) & Gap (\\%) & Best Found \\\\",
        "\\midrule"
    ]

    # Determine the best method (lowest gap) to bold
    best_gap = stats_df['Mean Gap %'].min()

    for _, row in stats_df.iterrows():
        method = row['Method']
        mean_cost = row['Mean Cost']
        std_cost = row['Std Cost']
        time = row['Mean Time']
        gap = row['Mean Gap %']
        success = row['Success Rate %']

        # Bold the method name if it's ours or best
        method_str = method
        if "NeuroCP-LNS" in method or abs(gap - best_gap) < 0.01:
            method_str = f"\\textbf{{{method}}}"
            gap_str = f"\\textbf{{{gap:.2f}}}"
        else:
            gap_str = f"{gap:.2f}"

        line = (f"{method_str} & "
                f"{mean_cost:.1f} $\\pm$ {std_cost:.1f} & "
                f"{time:.2f} & "
                f"{gap_str} & "
                f"{success:.1f}\\% \\\\")
        latex_str.append(line)

    latex_str.append("\\bottomrule")
    latex_str.append("\\end{tabular}")
    latex_str.append("\\end{table}")

    with open(out_path, "w") as f:
        f.write("\n".join(latex_str))
    print(f"[INFO] LaTeX table saved to {out_path}")

def run_analysis(df, out_dir):
    """Main analysis driver."""

    # --- A. Aggregated Statistics ---
    summary = df.groupby('Method').agg(
        Mean_Cost=('Cost', 'mean'),
        Median_Cost=('Cost', 'median'),
        Std_Cost=('Cost', 'std'),
        Mean_Time=('Time', 'mean'),
        Mean_Gap_Pct=('Gap', 'mean'),
        Success_Rate=('Is_Best', 'mean'),
        Feasible_Count=('Feasible', 'sum'),
        Total_Instances=('Graph', 'count')
    ).reset_index()

    # Rename columns for clarity
    summary.rename(columns={'Mean_Cost': 'Mean Cost', 'Std_Cost': 'Std Cost',
                            'Mean_Time': 'Mean Time', 'Mean_Gap_Pct': 'Mean Gap %',
                            'Success_Rate': 'Success Rate %'}, inplace=True)
    summary['Success Rate %'] *= 100

    # Save CSV summary
    summary.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False, float_format="%.2f")

    # Generate LaTeX Table
    generate_latex_table(summary, os.path.join(out_dir, "table_1_results.tex"))

    print("\n=== Aggregated Results ===")
    print(summary[['Method', 'Mean Cost', 'Mean Time', 'Mean Gap %', 'Success Rate %']].to_string(index=False))

    # --- A.2 Head-to-Head Win/Tie/Loss (vs Ours) ---
    baseline = 'NeuroCP-LNS (Ours)'
    if baseline in df['Method'].unique():
        pivot_cost = df.pivot_table(index='Graph', columns='Method', values='Cost')
        if len(pivot_cost.columns) > 1:
            print(f"\n=== Head-to-Head vs {baseline} ===")
            h2h_data = []
            for method in pivot_cost.columns:
                if method == baseline: continue

                # Compare method vs baseline (lower cost is better)
                # Diff = Method - Baseline
                # Diff > 0 => Baseline wins (Method is worse)
                # Diff < 0 => Method wins (Baseline is worse)
                diffs = pivot_cost[method] - pivot_cost[baseline]

                wins = (diffs < 0).sum() # Method beats Baseline
                losses = (diffs > 0).sum() # Baseline beats Method
                ties = (diffs == 0).sum()

                print(f"{baseline} vs {method:<20} | Wins: {losses} | Ties: {ties} | Losses: {wins}")
                h2h_data.append({
                    'Opponent': method,
                    'Ours_Wins': losses,
                    'Ties': ties,
                    'Ours_Losses': wins
                })

            if h2h_data:
                pd.DataFrame(h2h_data).to_csv(os.path.join(out_dir, "head_to_head.csv"), index=False)
        else:
            print(f"\n[INFO] Skipping Head-to-Head (Only 1 method present)")

    # --- B. Statistical Tests (Wilcoxon) ---
    pivot_gap = df.pivot_table(index='Graph', columns='Method', values='Gap')
    baseline = 'NeuroCP-LNS (Ours)'

    stats_log = []
    if baseline in pivot_gap.columns and len(pivot_gap.columns) > 1:
        print(f"\n=== Statistical Significance (vs {baseline}) ===")
        for method in pivot_gap.columns:
            if method == baseline: continue

            # Pairwise removal of NaNs
            valid_data = pivot_gap[[baseline, method]].dropna()
            x, y = valid_data[baseline], valid_data[method]

            if len(x) > 1 and not np.allclose(x, y):
                try:
                    stat, p = wilcoxon(x, y)
                    verdict = "**Signif**" if p < 0.05 else "Not Signif"
                    print(f"{method:<20} | p={p:.2e} | {verdict}")
                    stats_log.append({'Comparison': method, 'p-value': p, 'Significant': p < 0.05})
                except Exception as e:
                    print(f"{method:<20} | Error: {e}")
            else:
                print(f"{method:<20} | N/A (Insufficient data or identical)")

    pd.DataFrame(stats_log).to_csv(os.path.join(out_dir, "statistical_tests.csv"), index=False)

    return summary

def create_plots(df, summary_df, out_dir):
    """Creates publication-ready plots."""

    # Define a consistent palette
    methods = sorted(df['Method'].unique())
    palette = {}
    colors = sns.color_palette("colorblind", n_colors=len(methods))

    for i, m in enumerate(methods):
        if "NeuroCP-LNS (Ours)" in m:
            palette[m] = "#d62728" # Highlight Red/Brick
        else:
            palette[m] = colors[i] # Default

    # 1. Box Plot of Optimality Gaps
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='Method', y='Gap', palette=palette,
                showfliers=False, width=0.5, linewidth=1.2)
    plt.title("Optimality Gap Distribution")
    plt.ylabel("Gap to Best Known Solution (%)")
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_boxplot_gap.pdf"))
    plt.close()

    # 2. Success Rate Bar Chart
    plt.figure(figsize=(6, 4))
    sns.barplot(data=summary_df, x='Method', y='Success Rate %', palette=palette)
    plt.title("Success Rate (Finding BKS)")
    plt.ylabel("Success Rate (%)")
    plt.xlabel("")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(out_dir, "fig_barplot_success.pdf"))
    except PermissionError:
        print(f"[WARN] Could not save 'fig_barplot_success.pdf' (Permission denied). Is it open?")
    plt.close()

    # 3. Runtime vs Quality Tradeoff (Scatter)
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=summary_df, x='Mean Time', y='Mean Gap %', hue='Method', style='Method',
                    palette=palette, s=150, legend=False)

    # labels
    for i, row in summary_df.iterrows():
        plt.text(row['Mean Time']*1.05, row['Mean Gap %'],
                 row['Method'].replace(" ", "\n"), fontsize=8)

    plt.title("Efficiency Frontier")
    plt.xlabel("Average Runtime (s)")
    plt.ylabel("Average Optimality Gap (%)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(out_dir, "fig_scatter_tradeoff.pdf"))
    except PermissionError:
        print(f"[WARN] Could not save 'fig_scatter_tradeoff.pdf' (Permission denied). Is it open?")
    plt.close()

    print("\n[INFO] Plots saved to", out_dir)

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("="*40)
    print("  DRDP EXPERIMENTAL ANALYSIS (NeurIPS Style)")
    print("="*40)

    try:
        df, methods = load_and_clean_data(args)

        # --- FILTERING: Only consider graphs solved by our approach ---
        our_method = 'NeuroCP-LNS (Ours)'
        if our_method in df['Method'].unique():
            # Identify graphs where our method has a valid feasible solution
            valid_runs = df[(df['Method'] == our_method) & (df['Feasible'] == True)]
            solved_graphs = set(valid_runs['Graph'].unique())
            all_graphs = set(df['Graph'].unique())

            unsolved_graphs = all_graphs - solved_graphs

            if unsolved_graphs:
                print(f"\n[FILTERING] Excluding {len(unsolved_graphs)} graphs not solved by {our_method} (Resource/Time limits):")

                with open(os.path.join(args.out_dir, "excluded_graphs.txt"), "w") as f:
                    f.write(f"The following {len(unsolved_graphs)} graphs were excluded because {our_method} did not produce a feasible solution (likely due to OOM or timeout):\n")
                    for g in sorted(unsolved_graphs):
                        print(f"  - {g}")
                        f.write(f"{g}\n")

                print(f"[INFO] List of excluded graphs saved to '{os.path.join(args.out_dir, 'excluded_graphs.txt')}'")

                # Keep only intersections
                df = df[df['Graph'].isin(solved_graphs)]
                print(f"[FILTERING] Analysis proceeding with {len(solved_graphs)} common graphs.")
            else:
                print(f"[FILTERING] {our_method} solved all {len(all_graphs)} graphs present in dataset.")
        else:
            print(f"[WARN] Method '{our_method}' not found. Skipping filtering.")

        df_processed = compute_comparative_metrics(df)
        summary = run_analysis(df_processed, args.out_dir)
        create_plots(df_processed, summary, args.out_dir)
        print(f"\n[DONE] Analysis complete. Check '{args.out_dir}'")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
