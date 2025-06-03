# Input is 2-D array, example:
#                        D1  D2  D3 
#                  Ap1 [ 60  90  70 
#                  Ap2   20   0  20 
#                  Ap3   10   0   0 ]
# Potential food sources => Admisible paths (m')
# Number of best food sources => Best admisible paths to exploit (e)
# Quality assesment => Number of nodes (jumps) in the path - the less is better or loss function / target function (less is better).
# Hive => Demand (Di)
# Bees => Demand value (n)
import sys

# 1st strategy (aggregation) => all bees assigned to the best food source
# 2nd strategy (disaggregation) => some bees assigned to the best food source and rest randomly to the remaining ones


# demands_count = 3
# admisible_paths_count = 3

# custom flavor for the algorithm
# local_exp_step = 10

# chromosome = tf.zeros((admisible_paths_count, demands_count), dtype='float32') # Food Sources
# print(chromosome)

from structures.ObjectsDB import ObjectsDB
from structures.AdmisiblePaths import AdmisiblePaths
from structures.Demand import Demand
import numpy as np
import math
import random
import pandas as pd

randomness_seed = 77
random.seed(randomness_seed)

def calculate_link_encumbrance(bee_chromosome: pd.DataFrame | np.ndarray,
                              dpl_mapping: dict[str, dict[str, list[str]]]) -> dict[str, int]:
    path_ids = list(next(iter(dpl_mapping.values())).keys())
    demand_ids = list(dpl_mapping.keys())

    edge_encumbrance = {}  # temporary data_struct

    for col_idx, demand_id in enumerate(demand_ids):
        if demand_id not in dpl_mapping:
            continue
        path_to_links = dpl_mapping[demand_id]
        demand_values = bee_chromosome[:, col_idx]

        for row_idx, path_id in enumerate(path_ids):
            if path_id not in path_to_links or demand_values[row_idx] == 0:
                continue
            for link in path_to_links[path_id]:
                edge_encumbrance[link] = edge_encumbrance.get(link, 0) + demand_values[row_idx]

    return edge_encumbrance


def calculate_cost_function(edge_encumbrance: dict[str, int], modularity: int):
    return math.ceil(sum(edge_encumbrance.values()) / modularity)


def initialize_bee(db_context: ObjectsDB, distribution_strategy: int, as_ndarray: bool = True) -> pd.DataFrame | np.ndarray:
    """
    Creates a bee specimen with random chromosome values. Genes distribution is based on the distribution_strategy parameter.
    In other words, distribution strategy is related to number of splits of the demand's value gene.

    :param db_context: ObjectsDB instance
    :param distribution_strategy: 0 for deterministic aggregation (always first top path), 1 for random aggregation, 2 and more for random disaggregation
    :param as_ndarray: bool Should the output be given as numpy array?
    :return: DataFrame | np.ndarray
    """
    demands: dict[str, Demand] = db_context.get_demands()
    adm_paths: AdmisiblePaths = db_context.get_admisible_paths()
    max_paths = adm_paths.get_max_paths()
    data = pd.DataFrame()

    first_demand = next(iter(demands.values()))
    path_names = [path.get_path_id() for path in adm_paths.get_paths_for_demand(first_demand.get_demand_id())]
    data['path_id'] = path_names

    for dmd_id in demands.keys():
        demand_value = int(demands[dmd_id].get_demand_value())
        demand_distribution = _make_random_value_split(demand_value, max_paths, distribution_strategy)
        data[dmd_id] = demand_distribution

    if as_ndarray:
        # Return just the numeric data as a NumPy array (excluding the 'path_id' column)
        return data.drop(columns=['path_id']).to_numpy()

    # bees_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return data

def create_new_bee_population(db_context: ObjectsDB, population_size: int) -> list[pd.DataFrame] | list[np.ndarray]:
    bee_colony = []
    for i in range(population_size):
        bee_colony.append(initialize_bee(db_context, 1))

    return bee_colony

def assess_bee_solutions_quality(bee_population: list, dpl_mapping) -> tuple:
    bee_solutions = [ calculate_link_encumbrance(bee, dpl_mapping) for bee in bee_population ]
    bee_quality_grades = [ calculate_cost_function(solution) for solution in bee_solutions ]

    return (bee_solutions, bee_quality_grades)


def _make_random_value_split(total_value: int, list_size: int, n_splits: int) -> list:
    """
    Randomly splits a given integer total_value into a list_size-sized list of sub-values.
    If n_splits is less than list_size, then rest will be filled with zeroes.

    \n
    AGGREGATED SPLIT:
    If n_splits is equal 0, then always first value = total_value and the rest of the list is filled with zeroes.
    If n_splits is equal 1, then total_value is randomly assigned to position between 0 and list_size-1.

    DISAGGREGATED SPLIT:
    If n_splits is greater than 1, then total_value is randomly split into n_splits sub-values
    and randomly assigned to position between 0 and list_size-1.

    \nReturns list of sub-values based on n_splits parameter.
    """
    # Input validation
    if total_value < 0 or list_size < 0 or n_splits < 0:
        raise ValueError("total_value, list_size, and n_splits must be non-negative")
    if n_splits > list_size:
        raise ValueError("n_splits cannot be greater than list_size")

    if list_size == 0:
        return []

    values = [0] * list_size
    if n_splits == 0:
        # Put total_value in the first position of the list.
        values[0] = total_value
        return values
    elif n_splits == 1:
        values[random.randint(0, list_size - 1)] = total_value
        return values
    else:
        # Generate n_splits-1 random split points. Example: [20, 120, 160] for total_value = 300 and n_splits = 4
        split_points = sorted(random.sample(range(1, total_value), n_splits - 1))

        # Calculate non-zero parts
        first_part = [split_points[0]]  # example [20]
        mid_part = [split_points[i + 1] - split_points[i] for i in
                    range(len(split_points) - 1)]  # example [120-20,160-120]
        last_part = [total_value - split_points[-1]]  # example [300 - 160]
        parts = (first_part + mid_part + last_part)  # example [20, 100, 40, 140]

        # Create result list of list_size, initially containing all zeros
        values = [0] * list_size

        # Randomly assign non-zero parts to indices
        indices = random.sample(range(list_size), n_splits)
        for i, value in zip(indices, parts):
            values[i] = value

        return values




def train_model(data, colony_size=10, maximum_cycles=10, rr_search_areas, e_best_areas) -> list:

    iteration = 0
    best_solution_value: int = sys.maxsize
    last_n_best_solutions = []

    # 1. Initial population. Scouting phase I (send scouts).
    # Determine n_bees, rr'_prospect_areas, top_e_areas.
    # Randomly assign values (search areas)

    # 2. Assessment phase I (assess solutions' quality)
    # Assess scouts solutions with assessment function. Choose rr best areas for next step.

    while iteration < maximum_cycles:

        # 3. Exploration phase I (assign more bees for top M areas - top e from them are explored much stronger)
        # Assign additional bees for rr best areas (all bees were working from the beginning but now they are moving to best areas)

        # 4. Selection phase I (from rr explored areas, select 'e' the best ones which will be exploited in next iteration)
        # choose e top solutions from rr areas -> copy them to the next round

        # 5. Exploration / Scouting phase II (bees unassigned to top best areas are randomly assigned to new areas)
        # Keep single bees at top areas and let the rest of them scout other places.

        # 6. Assessment phase II (assess solutions' quality)
        # Assess scouts performance again.

        # 7. increase iteration and check stop_condition. If stop_condition == True, finish, otherwise continue


        iteration += 1
    # Return top bee_chromosome and cost_value
    return []
