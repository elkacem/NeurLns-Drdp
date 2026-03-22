import argparse
import os
import csv
from graph import Graph  # Ensure to define or import the Graph class with necessary methods
from ilps import DRDPModel  # Import DRDPModel class
# from alns import ALNS
from ga import GA
from aco import ACO
from _alnsWithGa import ALNSWithGA
from newAlnsWithGenetic import GaAlns
import networkx as nx

import time



# def save_solution(file_name, method, solution, cost=None):
#     """Save the solution and optional cost to a file."""
#     output_dir = "D:/rl/pythonRDP/pythonProject/results/new_alns_local/"
#     os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
#     output_file = os.path.join(output_dir, f"{method}_solutions.txt")
#
#     with open(output_file, "a") as f:
#         f.write(f"Graph: {file_name}\n")
#         f.write(f"Method: {method}\n")
#         f.write(f"Solution: {solution}\n")
#         if cost is not None:
#             f.write(f"Cost: {cost}\n")
#         f.write("\n")

def save_solution(file_name, method, solution, cost=None, elapsed_sec=None):
    """
    Append the solution, optional cost, and optional runtime to a results file.
    Also saves to a CSV file for easier analysis.
    """
    output_dir = "D:/rl/pythonRDP/pythonProject/results/new_alns_local/"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Text Block Format (Original)
    output_file_txt = os.path.join(output_dir, f"{method}_sstmodel.txt")
    with open(output_file_txt, "a", encoding="utf-8") as f:
        f.write(f"Graph: {file_name}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Solution: {solution}\n")
        if cost is not None:
            f.write(f"Cost: {cost}\n")
        if elapsed_sec is not None:
            f.write(f"Time(s): {elapsed_sec:.6f}\n")
        f.write("\n")

    # 2. CSV Format (For Analysis)
    output_file_csv = os.path.join(output_dir, f"{method}_results.csv")
    file_exists = os.path.isfile(output_file_csv)

    with open(output_file_csv, "a", newline='', encoding="utf-8") as csvfile:
        fieldnames = ['Graph', 'Method', 'Cost', 'Time', 'Solution']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'Graph': file_name,
            'Method': method,
            'Cost': int(cost) if cost is not None else -1,
            'Time': f"{elapsed_sec:.6f}" if elapsed_sec is not None else "",
            'Solution': str(solution)
        })



def run_drdp(graph, file_name):
    drdp_model = DRDPModel(graph)
    print("Running DRDP Models...")

    # Optimize Model DRDP
    model_drdp, x, y, z = drdp_model.build_model_drdp()
    solution = drdp_model.optimize_model(model_drdp, x + y + z)
    save_solution(file_name, "DRDP", solution)


def run_ga(graph, file_name, generations, population_size):
    ga = GA(graph, generations, population_size)
    print("Running Genetic Algorithm...")
    solution = ga.genetic_algorithm(generations, population_size, mutation_rate=3)
    save_solution(file_name, "GA", solution)


# def run_aco(graph, file_name, num_iterations, evaporation_rate, initial_pheromone):
#
#     aco = ACO(
#         graph,
#         initial_pheromone=initial_pheromone,
#         d_rate_aco=0.8,
#         d_rate=0.7,
#         d_min=0.1,
#         d_max=0.5,
#         k_max=5,
#         max_itr=150,
#         max_no_inpr=10,
#         evaporation_rate=evaporation_rate,
#         num_iterations=num_iterations
#     )
#     print("Running Ant Colony Optimization...")
#     termination_condition = 50  # Number of iterations to perform
#     best_solution_fitness, best_solution = aco.run_aco(termination_condition)
#     save_solution(file_name, "ACO", best_solution, best_solution_fitness)

# def run_aco(graph, file_name, num_iterations, evaporation_rate, initial_pheromone):
#     aco = ACO(
#         graph,
#         initial_pheromone=initial_pheromone,
#         d_rate_aco=0.8,
#         d_rate=0.7,
#         d_min=0.1,
#         d_max=0.5,
#         k_max=5,
#         max_itr=150,
#         max_no_inpr=10,
#         evaporation_rate=evaporation_rate,
#         num_iterations=num_iterations
#     )
#     print("Running Ant Colony Optimization...")
#     termination_condition = 50
#
#     t0 = time.perf_counter()
#     best_solution_fitness, best_solution = aco.run_aco(termination_condition)
#     elapsed = time.perf_counter() - t0
#
#     save_solution(file_name, "ACO", best_solution, best_solution_fitness, elapsed)

def run_aco(graph, file_name, num_iterations, evaporation_rate, initial_pheromone):
    aco = ACO(
        graph,
        initial_pheromone=initial_pheromone,
        d_rate_aco=0.8,
        d_rate=0.7,
        d_min=0.1,
        d_max=0.5,
        k_max=5,
        max_itr=150,
        max_no_inpr=10,
        evaporation_rate=evaporation_rate,
        num_iterations=num_iterations
    )
    print("Running Ant Colony Optimization...")
    termination_condition = 50

    t0 = time.perf_counter()
    best_solution_fitness, best_solution = aco.run_aco(termination_condition)
    elapsed = time.perf_counter() - t0

    save_solution(file_name, "ACO", best_solution, cost=best_solution_fitness, elapsed_sec=elapsed)
    print(f"[ACO] {file_name}: {elapsed:.6f} s")




# def run_alns(graph, file_name, termination_condition):
#     alns = ALNS(graph)
#     print("Running ALNS...")
#     best_solution, best_cost = alns.run(termination_condition)
#     print(best_solution, best_cost)
#     save_solution(file_name, "ALNS", best_solution, best_cost)


# def run_ga_alns(graph, file_name, generations, population_size, min_population, max_population):
#     ga_alns = GaAlns(graph, generations, population_size, min_population, max_population)
#     print("Running GA-ALNS...")
#     solution, best_cost = ga_alns.run()
#     save_solution(file_name, "GA-ALNS", solution, best_cost)

def is_graph_connected(graph):
    """Check if the graph is connected using NetworkX."""
    nx_graph = graph.to_networkx()  # Convert your custom Graph to a NetworkX graph
    return nx.is_connected(nx_graph)

def main():
    parser = argparse.ArgumentParser(description="Graph-Based Optimization Algorithms")
    parser.add_argument("--method", type=str, required=True, choices=["drdp", "ga", "aco", "alns", "ga_alns"],
                        help="Choose the method to run: drdp, ga, aco, alns, ga_alns")
    parser.add_argument("--folder", type=str, default="D:/rl/pythonRDP/datasets/test", required=True, help="Folder path containing .gz files")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations for GA or GA-ALNS")
    parser.add_argument("--population_size", type=int, default=10, help="Population size for GA or GA-ALNS")
    parser.add_argument("--min_population", type=int, default=20, help="Population size for GA or GA-ALNS")
    parser.add_argument("--max_population", type=int, default=100, help="Population size for GA or GA-ALNS")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of iterations for ACO")
    parser.add_argument("--evaporation_rate", type=float, default=0.1, help="Evaporation rate for ACO")
    parser.add_argument("--initial_pheromone", type=float, default=1.0, help="Initial pheromone level for ACO")
    parser.add_argument("--termination_condition", type=int, default=200,
                        help="Termination condition for ALNS or GA-ALNS")
    args = parser.parse_args()

    # Iterate through all files in the folder
    for filename in os.listdir(args.folder):
        if filename.endswith('.gz'):  # Only process .gz files
            file_path = os.path.join(args.folder, filename)
            print(f"Processing file: {file_path}")

            # Create a new Graph instance and read the graph from the file
            graph = Graph()
            graph.read_graph_from_file(file_path)

            # Check connectivity using NetworkX
            # if not graph.is_graph_connected():
            #     print(f"Graph {filename} is not connected. Skipping.")
            #     continue

            if not graph.is_graph_connected():
                print(f"⚠️ Graph {filename} is not connected; processing anyway.")

            # Select the method to run
            if args.method == "drdp":
                run_drdp(graph, filename)
            elif args.method == "ga":
                run_ga(graph, filename, args.generations, args.population_size)
            elif args.method == "aco":
                run_aco(graph, filename, args.num_iterations, args.evaporation_rate, args.initial_pheromone)
            # elif args.method == "alns":
            #     run_alns(graph, filename, args.termination_condition)
            # elif args.method == "ga_alns":
            #     run_ga_alns(graph, filename, args.generations, args.population_size, args.min_population, args.max_population)


if __name__ == "__main__":
    main()
