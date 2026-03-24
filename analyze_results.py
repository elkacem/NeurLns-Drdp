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
    default_aco = "aco_results.csv"

    parser.add_argument("--results", type=str, help="Path to main solver results", default=default_results)
    parser.add_argument("--classical", type=str, help="Path to classical results (ACO)", default=default_aco)
    parser.add_argument("--exact", type=str, help="Path to exact solver results", default="exact_results.csv")

    parser.add_argument("--out_dir", type=str, default="analysis_output", help="Directory for paper artifacts")
    return parser.parse_args()

def load_and_clean_data(args):
    """Loads CSVs, normalizes headers, handles infeasible (-1) entries."""
    # Define method names for paper
    file_map = {
        'NeuroCP-LNS': args.results,
    }

    if args.classical and os.path.exists(args.classical):
        file_map['Metaheuristic-ACO'] = args.classical
    elif os.path.exists("aco_results.csv"):
        file_map['Metaheuristic-ACO'] = "aco_results.csv"

    if args.exact and os.path.exists(args.exact):
        file_map['Exact-CP'] = args.exact
    elif os.path.exists("exact_results.csv"):
        file_map['Exact-CP'] = "exact_results.csv"

    dfs = []
    print(f"{'Method':<25} | {'Path':<40} | {'Status'}")
    print("-" * 80)

    valid_methods = []

    for method, path in file_map.items():
        if path and os.path.exists(path):
            try:
                # Header detection: Read first line
                with open(path, 'r') as f:
                    first_line = f.readline().strip()

                has_header = "Graph" in first_line and "Cost" in first_line

                if has_header:
                    df = pd.read_csv(path)
                else:
                    # Assume: Graph, Method, Cost, Time, Iterations (standard for results.txt.csv)
                    # But wait, results.txt.csv might have 4 or 5 columns.
                    # check column count
                    num_cols = len(first_line.split(','))
                    if num_cols == 5:
                        names = ['Graph', 'Method', 'Cost', 'Time', 'Iterations']
                    elif num_cols == 4:
                         names = ['Graph', 'Method', 'Cost', 'Time']
                    else:
                         # fallback
                         names = None

                    df = pd.read_csv(path, header=None, names=names)

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

def conduct_friedman_test(df, out_dir):
    """
    Conducts Friedman Rank Sum Test and outputs Average Ranks.
    Friedman test checks if there are statistically significant differences between methods.
    """
    # Pivot table: Graph x Method -> Cost
    pivot = df.pivot_table(index='Graph', columns='Method', values='Cost')
    pivot = pivot.dropna() # Requires complete data (all methods solved the instance)

    if len(pivot) < 5 or len(pivot.columns) < 2:
        print("\n[INFO] Skipping Friedman Test (insufficient overlapping data).")
        return

    print(f"\n=== Friedman Test ({len(pivot)} instances) ===")

    # Calculate Ranks (lower cost = rank 1)
    ranks = pivot.rank(axis=1, ascending=True)
    avg_ranks = ranks.mean().sort_values()

    print("Average Ranks (Lower is better):")
    print(avg_ranks)

    # Save ranks
    ranks.reset_index().to_csv(os.path.join(out_dir, "friedman_ranks.csv"), index=False)

    # Simple Friedman calculation
    from scipy.stats import friedmanchisquare
    # args for friedman: array of measurements for each method
    args = [pivot[col].values for col in pivot.columns]
    try:
        stat, p = friedmanchisquare(*args)
        print(f"Friedman Chi^2 = {stat:.2f}, p-value = {p:.4e}")
        if p < 0.05:
            print(">> Significant difference detected among methods.")
        else:
            print(">> No significant difference detected.")

        with open(os.path.join(out_dir, "statistical_tests_friedman.txt"), "w") as f:
            f.write(f"Friedman Test Results\n")
            f.write(f"Instances: {len(pivot)}\n")
            f.write(f"Chi^2: {stat:.4f}\n")
            f.write(f"p-value: {p:.4e}\n\n")
            f.write("Average Ranks:\n")
            f.write(avg_ranks.to_string())

    except Exception as e:
        print(f"Friedman Test Failed: {e}")

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

    # Friedman Test / Wilcoxon
    if len(df['Method'].unique()) > 2:
        conduct_friedman_test(df, out_dir)
    elif len(df['Method'].unique()) == 2:
        # Wilcoxon
        methods = df['Method'].unique()
        pivot = df.pivot_table(index='Graph', columns='Method', values='Cost').dropna()
        if len(pivot) > 0:
            stat, p = wilcoxon(pivot[methods[0]], pivot[methods[1]])
            print(f"\n=== Wilcoxon Signed-Rank Test ({methods[0]} vs {methods[1]}) ===")
            print(f"Statistic: {stat}, p-value: {p}")
            if p < 0.05:
                print(">> Significant difference detected.")
            else:
                print(">> No significant difference detected.")

    # Generate LaTeX Table
    generate_latex_table(summary, os.path.join(out_dir, "table_1_results.tex"))

    # --- Custom: Save Best Solution Per Graph ---
    # We want to identify the winner and save the solution vector if available.
    # Note: df currently might not have 'Solution' column if not all files had it.
    # If 'Solution' is in df, we can use it.
    if 'Solution' in df.columns:
        best_rows = df.loc[df['Is_Best']].copy()
        # Deduplicate if multiple methods found best, pick one (e.g. fastest)
        best_rows.sort_values(by=['Graph', 'Time'], inplace=True)
        best_unique = best_rows.drop_duplicates(subset=['Graph'])

        best_sol_path = os.path.join(out_dir, "best_solutions.csv")
        best_unique[['Graph', 'Method', 'Cost', 'Time', 'Solution']].to_csv(best_sol_path, index=False)
        print(f"\n[INFO] Saved best solutions with vectors to {best_sol_path}")

    # --- Custom: Where did ACO win? ---
    pivot = df.pivot_table(index='Graph', columns='Method', values='Cost')
    if 'Metaheuristic-ACO' in pivot.columns and 'NeuroCP-LNS' in pivot.columns:
        # ACO wins if Cost < Neuro
        aco_wins = pivot[pivot['Metaheuristic-ACO'] < pivot['NeuroCP-LNS']]
        if not aco_wins.empty:
            print(f"\n=== Instances where ACO outperforms NeuroCP-LNS ({len(aco_wins)}) ===")
            print(aco_wins[['Metaheuristic-ACO', 'NeuroCP-LNS']].to_string())
            aco_wins.to_csv(os.path.join(out_dir, "aco_wins.csv"))

    print("\n=== Aggregated Results ===")
    print(summary[['Method', 'Mean Cost', 'Mean Time', 'Mean Gap %', 'Success Rate %']].to_string(index=False))

    return summary

def create_plots(df, summary_df, out_dir):
    """Creates publication-ready plots."""

    # Define a consistent palette
    methods = sorted(df['Method'].unique())
    palette = {}
    colors = sns.color_palette("colorblind", n_colors=len(methods))

    for i, m in enumerate(methods):
        if "NeuroCP-LNS" in m:
            palette[m] = "#d62728" # Highlight Red/Brick
        else:
            palette[m] = colors[i] # Default

    def safe_save(fname):
        try:
            plt.savefig(fname)
            print(f"Saved {fname}")
        except PermissionError:
            print(f"[WARN] Could not save {fname} (Permission denied). Close the file.")
        except Exception as e:
            print(f"[WARN] Could not save {fname}: {e}")

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
    safe_save(os.path.join(out_dir, "fig_boxplot_gap.pdf"))
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
    safe_save(os.path.join(out_dir, "fig_barplot_success.pdf"))
    plt.close()

    # 4. Performance Profile (Cumulative Distribution of Ratios)
    if len(methods) > 1:
        plt.figure(figsize=(6, 4))
        # Calculate cost ratio per instance: r_p,s = Cost / min_Cost
        pivot = df.pivot_table(index='Graph', columns='Method', values='Cost', aggfunc='min')
        min_costs = pivot.min(axis=1)

        for method in methods:
            if method not in pivot.columns: continue
            ratios = pivot[method] / min_costs
            ratios = ratios.dropna().sort_values()

            # CDF
            y = np.arange(1, len(ratios) + 1) / len(ratios)
            plt.step(ratios, y, where='post', label=method, color=palette.get(method, 'k'))

        plt.title("Performance Profile (Cost)")
        plt.xlabel(r"Performance Ratio ($\tau$)")
        # Fix: use \leq or <=. pure matplotlib mathtext prefers \leq
        plt.ylabel(r"Probability ($P(r_{p,s} \leq \tau$)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(1.0, 1.05 if ratios.max() < 1.05 else min(1.5, ratios.max())) # Zoom in on near-optimal
        plt.tight_layout()
        safe_save(os.path.join(out_dir, "fig_perf_profile_v2.pdf"))
        plt.close()

    # 5. Box Plot of Runtimes (Log Scale)
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='Method', y='Time', palette=palette, showfliers=False)
    plt.yscale('log')
    plt.title("Runtime Distribution (Log Scale)")
    plt.ylabel("Time (s)")
    plt.xlabel("")
    plt.xticks(rotation=15)
    plt.tight_layout()
    safe_save(os.path.join(out_dir, "fig_boxplot_time_v2.pdf"))
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
    safe_save(os.path.join(out_dir, "fig_scatter_tradeoff_v2.pdf"))
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
        our_method = 'NeuroCP-LNS'
        if our_method in df['Method'].unique():
            # Identify graphs where our method has a valid feasible solution
            valid_runs = df[(df['Method'] == our_method) & (df['Feasible'] == True)]
            solved_graphs = set(valid_runs['Graph'].unique())
            all_graphs = set(df['Graph'].unique())

            unsolved_graphs = all_graphs - solved_graphs

            if unsolved_graphs:
                print(f"\n[FILTERING] Excluding {len(unsolved_graphs)} graphs not solved by {our_method} (Resource/Time limits):")

                # Check if we are accidentally excluding graphs solely because ACO solved them but NeuroCP-LNS failed
                # This is important for fair comparison. If NeuroCP-LNS failed, it's a loss.
                # However, for metric aggregation (mean cost), we can only compare on common subset.
                # Or we assign penalty (infinity) to failures.
                # The user asked to "consider on the analysis only the graphs that are solved in our approach".

                with open(os.path.join(args.out_dir, "excluded_graphs.txt"), "w") as f:
                    f.write(f"The following {len(unsolved_graphs)} graphs were excluded because {our_method} did not produce a feasible solution:\n")
                    for g in sorted(unsolved_graphs):
                        # print(f"  - {g}") # verbose
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
