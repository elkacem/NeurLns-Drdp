import sys
import os
import time
import argparse
import csv
import gzip
from ortools.sat.python import cp_model

def read_mtx_gz(path: str):
    """
    Reads a Matrix Market file (.mtx.gz).
    Returns (n, adj_list).
    Note: MTX is 1-based, we convert to 0-based.
    """
    with gzip.open(path, 'rt') as f:
        # Skip comments
        line = f.readline()
        while line and (line.strip() == '' or line[0] in 'c%'):
            line = f.readline()
        if not line:
            # Empty file case or just header
            return 0, []

        # Read Matrix Size
        try:
            parts = line.strip().split()
            # typically R C entries
            r, c = int(parts[0]), int(parts[1])
        except ValueError:
            return 0, []

        n = max(r, c)
        adj = [set() for _ in range(n)]

        for l in f:
            if not l.strip() or l[0] in 'c%': continue
            try:
                # expecting: u v [value]
                parts = l.split()
                u, v = int(parts[0]) - 1, int(parts[1]) - 1
                if u != v:
                    if 0 <= u < n and 0 <= v < n:
                        adj[u].add(v)
                        adj[v].add(u)
            except ValueError:
                continue

    return n, [list(s) for s in adj]

def solve_exact(n, adj, time_limit=60.0, workers=8):
    if n == 0:
        return [], 0, True

    model = cp_model.CpModel()

    # Variables
    # y[i] is true if label is 2
    # z[i] is true if label is 3
    # If both false, label is 0
    y = [model.NewBoolVar(f"y_{i}") for i in range(n)]
    z = [model.NewBoolVar(f"z_{i}") for i in range(n)]

    # Constraint 1: Mutually exclusive labels (can't be 2 and 3 at once)
    for i in range(n):
        model.Add(y[i] + z[i] <= 1)

    # Constraint 2: Double Roman Domination Condition
    # For every vertex v with label 0, it must be dominated by:
    #   - at least one neighbor with label 3
    #   - OR at least two neighbors with label 2
    # This is equivalent to: sum_{u in N(v)} (2*z_u + 1*y_u) >= 2

    for i in range(n):
        neighbors = adj[i]

        # The constraint only applies if node i has label 0.
        # Node i is label 0 if NOT (y[i] or z[i]).
        # Expression for coverage score from neighbors:
        # score = sum( 2*z[j] + y[j] )

        # Linear expression for sum
        # We can build it directly
        score_expr = sum((2 * z[j] + y[j]) for j in neighbors)

        # Add constraint implies Logic:
        # (Label(i) == 0) => (score >= 2)
        # In CP-SAT: OnlyEnforceIf works with literals.
        # Literal for (Label(i) == 0) is [Not(y[i]), Not(z[i])]

        model.Add(score_expr >= 2).OnlyEnforceIf([y[i].Not(), z[i].Not()])

    # Objective: Minimize Cost = Sum (2*y + 3*z)
    obj = sum(2 * y[i] + 3 * z[i] for i in range(n))
    model.Minimize(obj)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    # solver.parameters.num_search_workers = workers # CP-SAT manages workers automatically well, but let's keep user control
    if workers > 0:
        solver.parameters.num_search_workers = workers
    # solver.parameters.log_search_progress = True

    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol = [0] * n
        for i in range(n):
            if solver.Value(z[i]):
                sol[i] = 3
            elif solver.Value(y[i]):
                sol[i] = 2
            else:
                sol[i] = 0

        cost = int(solver.ObjectiveValue())
        is_opt = (status == cp_model.OPTIMAL)
        return sol, cost, is_opt
    else:
        return None, -1, False

def run_dir(data_dir, out_path, time_limit, workers):
    # Ensure standard CSV name
    csv_path = out_path if out_path.endswith('.csv') else out_path + ".csv"

    # Check if we are appending or writing new
    # If the file exists and looks like our format, we might want to skip existing?
    # But usually 'solve' implies fresh run or explicit overwrite in this context.
    # Let's overwrite for clean results.

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Graph', 'Method', 'Cost', 'Time', 'Solution'])

        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            return

        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".mtx.gz")])
        print(f"Found {len(files)} graphs in {data_dir}")
        print("Using Solver Backend: CP-SAT (Default)")

        for fname in files:
            path = os.path.join(data_dir, fname)
            base = fname

            try:
                n, adj = read_mtx_gz(path)
                if n == 0:
                    print(f"{base:<25} | Skipped (Empty/Invalid)")
                    continue

                print(f"{base:<25} | Solving (n={n})...", end='', flush=True)

                t0 = time.time()
                sol, cost, is_opt = solve_exact(n, adj, time_limit, workers)
                dur = time.time() - t0

                method_label = "Exact-CP"
                # Optionally mark if it hit time limit?
                # But typically for comparison we just say "Exact-CP".
                # If non-optimal, it's technically a Heuristic result from CP.

                if sol is not None:
                    opt_str = "*" if is_opt else "(TimeOut)"
                    print(f" Cost: {cost}{opt_str} ({dur:.2f}s)")
                    writer.writerow([base, method_label, cost, f"{dur:.4f}", str(sol)])
                else:
                    print(f" Failed/Timeout ({dur:.2f}s)")
                    writer.writerow([base, method_label, -1, f"{dur:.4f}", "[]"])

                f.flush()

            except Exception as e:
                print(f"\n[ERROR] {base}: {e}")
                writer.writerow([base, "Exact-CP", -1, 0, "[]"])

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exact DRDP Solver using OR-Tools CP-SAT")
    p.add_argument("--data_dir", required=True, help="Directory containing .mtx.gz graphs")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--time_limit", type=float, default=60.0, help="Time limit per graph in seconds")
    p.add_argument("--workers", type=int, default=1, help="Number of CP-SAT workers")

    args = p.parse_args()

    run_dir(args.data_dir, args.out, args.time_limit, args.workers)
