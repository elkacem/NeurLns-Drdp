import networkx as nx
import random
import numpy as np
import argparse
import os
import gzip
import time
import sys

# --- Helper for standalone execution ---

def read_mtx_gz_to_nx(path: str) -> nx.Graph:
    """Reads a .mtx.gz file and returns a NetworkX graph."""
    with gzip.open(path, 'rt') as f:
        line = f.readline()
        while line and (line.strip() == '' or line[0] in 'c%'):
            line = f.readline()
        if not line:
            raise ValueError(f"Invalid MTX header in {path}")
        parts = line.strip().split()
        r, c = int(parts[0]), int(parts[1])
        # n = max(r, c) # typically square

        G = nx.Graph()
        # Ensure all nodes exist (0 to n-1)
        # However, MTX is 1-based, we convert to 0-based.
        # Nodes are 1..M, 1..N.

        for l in f:
            if not l.strip() or l[0] in 'c%': continue
            try:
                # MTX format: u v [weight]
                # We assume unweighted or ignore weights
                parts = l.split()
                u, v = int(parts[0]) - 1, int(parts[1]) - 1
                if u != v:
                    G.add_edge(u, v)
            except ValueError:
                continue

    # Ensure graph is connected or handled as is?
    # The original main.py handled it. We just return G.
    # The ACO code assumes 0..n-1 indices matching list indices.
    # We should relabel nodes to integers 0..|V|-1 if they aren't already compact,
    # but MTX usually implies 1..N.
    # Let's ensure nodes 0..max_idx exist if they are isolated.
    # Actually, simplistic read is:
    return G

class GraphWrapper:
    """Adapts NetworkX graph to interface expected by ACO."""
    def __init__(self, nx_graph):
        self.graph = nx_graph
        # Ensure nodes are 0..n-1
        self.mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
        self.inv_mapping = {i: node for node, i in self.mapping.items()}
        # Relabel to integers 0..n-1 for internal ACO logic
        self.graph = nx.relabel_nodes(nx_graph, self.mapping)

class ACO:
    def __init__(self, graph, initial_pheromone, d_rate_aco, d_rate, d_min, d_max, k_max, max_itr, max_no_inpr,
                 evaporation_rate, num_iterations):
        self.graph = graph
        self.initial_pheromone = initial_pheromone
        self.d_rate_aco = d_rate_aco
        self.d_rate = d_rate
        self.d_min = d_min
        self.d_max = d_max
        self.k_max = k_max
        self.max_itr = max_itr
        self.max_no_inpr = max_no_inpr
        self.evaporation_rate = evaporation_rate
        self.num_iterations = num_iterations
        self.pheromone = [initial_pheromone] * graph.graph.number_of_nodes()

    def initialize_pheromone(self):
        self.pheromone = [self.initial_pheromone] * self.graph.graph.number_of_nodes()

    def choose_vertex(self, remaining_vertices, d_rate):
        r = random.random()
        if r <= d_rate:
            max_objective = -1
            selected_vertex = -1
            for u in remaining_vertices:
                objective = self.graph.graph.degree[u] * self.pheromone[u]
                if objective > max_objective:
                    max_objective = objective
                    selected_vertex = u
            return selected_vertex
        else:
            total_weight = sum(self.graph.graph.degree[u] * self.pheromone[u] for u in remaining_vertices)
            random_weight = random.random() * total_weight
            cumulative_weight = 0.0
            for u in remaining_vertices:
                cumulative_weight += self.graph.graph.degree[u] * self.pheromone[u]
                if cumulative_weight >= random_weight:
                    return u
        return -1

    def construct_solution(self):
        num_vertices = self.graph.graph.number_of_nodes()
        solution = [0] * num_vertices
        remaining_vertices = list(range(num_vertices))

        while remaining_vertices:
            selected_vertex = self.choose_vertex(remaining_vertices, self.d_rate_aco)
            solution[selected_vertex] = 3
            remaining_vertices.remove(selected_vertex)
            for neighbor in self.graph.graph.neighbors(selected_vertex):
                solution[neighbor] = 0
                if neighbor in remaining_vertices:
                    remaining_vertices.remove(neighbor)
        return solution

    def extend_solution(self, solution):
        S = solution[:]
        V02 = [u for u in range(len(solution)) if solution[u] == 0 or solution[u] == 2]
        random.shuffle(V02)
        iterations = int(0.05 * len(V02))

        while iterations:
            selected_vertex = self.choose_vertex(V02, self.d_rate)
            S[selected_vertex] = 3
            V02.remove(selected_vertex)
            iterations -= 1

        return S

    def feasibility_check(self, C):
        for u in range(len(C)):
            if C[u] == 0:
                has_neighbor_selected = False
                num_neighbors_with_c2 = 0
                for v in self.graph.graph.neighbors(u):
                    if C[v] == 3:
                        has_neighbor_selected = True
                        break
                    if C[v] == 2:
                        num_neighbors_with_c2 += 1
                if not has_neighbor_selected and num_neighbors_with_c2 < 2:
                    return False
            elif C[u] == 1:
                has_neighbor_with_c2_or_3 = False
                for v in self.graph.graph.neighbors(u):
                    if C[v] == 2 or C[v] == 3:
                        has_neighbor_with_c2_or_3 = True
                        break
                if not has_neighbor_with_c2_or_3:
                    return False
        return True

    def reduce_solution(self, solution):
        S = solution[:]
        degrees = dict(self.graph.graph.degree())
        sorted_vertices = sorted(range(len(S)), key=lambda x: degrees[x])

        for u in sorted_vertices:
            if S[u] == 3 or S[u] == 2:
                init_lab = S[u]
                S[u] = 0
                if not self.feasibility_check(S):
                    S[u] = 2
                    if not self.feasibility_check(S):
                        S[u] = init_lab

        return S

    def get_unlabeled_vertices(self, S):
        return [u for u in range(len(S)) if S[u] == 0 or S[u] == 2]

    def calculate_d(self, k):
        return self.d_min + (k - 1) * ((self.d_max - self.d_min) / (self.k_max - 1))

    def destroy_solution(self, S, k):
        V_prime = list(range(len(S)))
        num_iterations = int(len(S) * self.calculate_d(k))
        while num_iterations:
            u = random.choice(V_prime)
            if S[u] == 0 or S[u] == 2:
                S[u] = -1
            else:
                num_iterations += 1
            V_prime.remove(u)
            num_iterations -= 1
        return S

    def repair(self, solution):
        remaining_vertices = [u for u in range(len(solution)) if solution[u] != 3]
        while remaining_vertices:
            selected_vertex = self.choose_vertex(remaining_vertices, self.d_rate)
            solution[selected_vertex] = 3
            remaining_vertices.remove(selected_vertex)
            for neighbor in self.graph.graph.neighbors(selected_vertex):
                solution[neighbor] = 0
                if neighbor in remaining_vertices:
                    remaining_vertices.remove(neighbor)
        return solution

    def calculate_solution_sum(self, S_prime):
        return sum(S_prime)

    def random_variable_neighborhood_search(self, solution):
        S_prime = solution[:]
        k = 1
        cnoinpr = 0

        while self.max_itr and cnoinpr < self.max_no_inpr:
            S_primed = self.destroy_solution(S_prime[:], k)
            S_primer = self.repair(S_primed)
            S_primee = self.extend_solution(S_primer)
            S_primere = self.reduce_solution(S_primee)
            S_prime = S_primere

            if self.calculate_solution_sum(S_prime) < self.calculate_solution_sum(solution):
                solution = S_prime
                k = 1
                cnoinpr = 0
            else:
                k += 1
                cnoinpr += 1

            if k > self.k_max:
                k = 1

            self.max_itr -= 1

        return solution

    def delta(self, curr_best_solution, u):
        return 1.0 if curr_best_solution[u] != 0 else 0.0

    def update_pheromone(self, curr_best_solution_fitness, curr_best_solution, best_solution_fitness, best_solution):
        for u in range(self.graph.graph.number_of_nodes()):
            self.pheromone[u] += self.evaporation_rate * (
                    (curr_best_solution_fitness * self.delta(curr_best_solution, u) +
                     best_solution_fitness * self.delta(best_solution, u)) /
                    (curr_best_solution_fitness + best_solution_fitness) - self.pheromone[u]
            )

    def calculate_cf(self):
        tau_max = max(self.pheromone)
        tau_min = min(self.pheromone)
        sum_differences = sum(max(tau_max - tau, tau - tau_min) for tau in self.pheromone)
        cf = 2 * (sum_differences / len(self.pheromone)) * ((tau_max - tau_min) - 1)
        return cf

    def run_aco(self, termination_condition):
        self.initialize_pheromone()
        best_solution = []
        best_solution_fitness = float('inf')
        curr_best_solution = []
        curr_best_solution_fitness = float('inf')

        while termination_condition:
            for _ in range(self.num_iterations):
                solution = self.construct_solution()
                solution = self.extend_solution(solution)
                solution = self.reduce_solution(solution)
                solution = self.random_variable_neighborhood_search(solution)

                solution_fitness = self.calculate_solution_sum(solution)

                if solution_fitness < curr_best_solution_fitness:
                    curr_best_solution = solution
                    curr_best_solution_fitness = solution_fitness

            if curr_best_solution_fitness < best_solution_fitness:
                best_solution = curr_best_solution
                best_solution_fitness = curr_best_solution_fitness

            self.update_pheromone(curr_best_solution_fitness, curr_best_solution, best_solution_fitness, best_solution)
            cf = self.calculate_cf()
            if cf > 0.99:
                self.initialize_pheromone()
            termination_condition -= 1

        return best_solution_fitness, best_solution

# --- Main Execution Block ---

def solve_file(filepath, args):
    basename = os.path.basename(filepath)
    try:
        # Load Graph
        nx_graph = read_mtx_gz_to_nx(filepath)

        # If graph is empty or has issues
        if nx_graph.number_of_nodes() == 0:
            print(f"[ERROR] {basename}: Empty graph")
            return

        graph_wrapper = GraphWrapper(nx_graph)

        # Initialize ACO
        # Parameters from main.py default or args
        # defaults: d_rate_aco=0.8, d_rate=0.7, d_min=0.1, d_max=0.5, k_max=5,
        #           max_itr=150, max_no_inpr=10, evap=0.1, num_it=10, termination=50

        aco = ACO(
            graph=graph_wrapper,
            initial_pheromone=args.initial_pheromone,
            d_rate_aco=0.8,
            d_rate=0.7,
            d_min=0.1,
            d_max=0.5,
            k_max=5,
            max_itr=150,
            max_no_inpr=10,
            evaporation_rate=args.evaporation_rate,
            num_iterations=args.num_iterations
        )

        t0 = time.perf_counter()
        best_cost, best_sol = aco.run_aco(args.termination_condition)
        elapsed = time.perf_counter() - t0

        # Map solution back to original node ids if needed?
        # For labeling problems, usually just the label array matters if nodes are canonical.
        # But if we relabeled, we might need to be careful if we output explicit sets.
        # Here solution is a list of labels [0,1,2,3...].
        # Just convert list to simple string format.

        # Output strictly in format
        print(f"Graph: {basename}")
        print(f"Solution: {list(best_sol)}")
        print(f"Cost: {int(best_cost)}")
        print(f"Time(s): {elapsed:.6f}")
        print("-" * 20) # Separator

        # Also write to CSV if requested (or just redirect stdout)
        # We will follow the pattern of printing to stdout, user can redirect.
        # But user wants "results_full.txt.csv" style.
        # We can implement simple appending to a file.

        if args.out:
            # Check if file exists to write header
            file_exists = os.path.isfile(args.out)
            with open(args.out, 'a', newline='') as csvfile:
                fieldnames = ['Graph', 'Method', 'Cost', 'Time', 'Solution']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'Graph': basename,
                    'Method': 'ACO',
                    'Cost': int(best_cost),
                    'Time': f"{elapsed:.6f}",
                    'Solution': str(list(best_sol))
                })

    except Exception as e:
        print(f"[ERROR] {basename}: {e}")
        traceback.print_exc()

import csv
import traceback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACO Solver for DRDP")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .mtx.gz files")
    parser.add_argument("--out", type=str, default="aco_results.csv", help="Output CSV file")

    # ACO Hyperparameters
    parser.add_argument("--termination_condition", type=int, default=50, help="Outer loop iterations")
    parser.add_argument("--num_iterations", type=int, default=10, help="Inner ACO iterations")
    parser.add_argument("--evaporation_rate", type=float, default=0.1, help="Pheromone evaporation")
    parser.add_argument("--initial_pheromone", type=float, default=1.0, help="Initial pheromone")

    parser.add_argument("--limit", type=int, default=None, help="Limit number of graphs to solve")

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} not found.")
        sys.exit(1)

    files = [f for f in os.listdir(args.data_dir) if f.endswith('.mtx.gz')]
    files.sort()

    if args.limit:
        files = files[:args.limit]

    print(f"Found {len(files)} graphs in {args.data_dir}")

    for f in files:
        solve_file(os.path.join(args.data_dir, f), args)
