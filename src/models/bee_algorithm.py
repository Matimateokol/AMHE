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

# 1st strategy (aggregation) => all bees assigned to the best food source
# 2nd strategy (disaggregation) => some bees assigned to the best food source and rest randomly to the remaining ones


# demands_count = 3
# admisible_paths_count = 3

# custom flavor for the algorithm
# local_exp_step = 10

# chromosome = tf.zeros((admisible_paths_count, demands_count), dtype='float32') # Food Sources
# print(chromosome)

import sys
from typing import TypeAlias

from structures.ObjectsDB import ObjectsDB
from structures.AdmisiblePaths import AdmisiblePaths
from structures.Demand import Demand
import numpy as np
import math
import random
import pandas as pd

randomness_seed = 77
random.seed(randomness_seed)

### Custom type-aliases ###
DPLMapping: TypeAlias = dict[str, dict[str, list[str]]]  # key: demand_id, value: (key: path_id, value: links)
EncumbrancePerLink: TypeAlias = dict[str, int]  # key: link_id, value: encumbrance
BeeSpecimen: TypeAlias = pd.DataFrame | np.ndarray  # Bee chromosome representing the potential solution
BeePopulation: TypeAlias = list[BeeSpecimen]
ScoutedAreas: TypeAlias = list[EncumbrancePerLink]
ScoutedAreasQuality: TypeAlias = list[int]
BeeSearchArea: TypeAlias = tuple[BeeSpecimen, EncumbrancePerLink, int]
BeeHiveSearchAreas: TypeAlias = list[BeeSearchArea]


def _make_random_value_split(total_value: int, list_size: int, n_splits: int) -> list[int]:
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
    :param total_value: the total integer value to be split
    :param list_size: size of the list
    :param n_splits: number of sub-values to be extracted from total_value
    :return: list of (list_size - n_splits) zeroes and n_splits sub-values from total_value
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


def calculate_link_encumbrance(bee_chromosome: BeeSpecimen,
                               dpl_mapping: DPLMapping) -> EncumbrancePerLink:
    """
    Calculate the total encumbrance on each Link used to deliver Demands in given network design.
    :param bee_chromosome: 2-D numpy array of shape [path_id, demand_id] or pandas dataframe representing solution
    :param dpl_mapping: dictionary mapping demand_id to its path_ids and link_ids on them
    :return: link_id to encumbrance value mapping
    """
    path_ids: list[str] = list(next(iter(dpl_mapping.values())).keys())
    demand_ids: list[str] = list(dpl_mapping.keys())
    edge_encumbrance: EncumbrancePerLink = {}

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


def calculate_cost_function(edge_encumbrance: EncumbrancePerLink, modularity: int) -> int:
    """
    Calculates the total cost of a given solution. It sums up encumbrance on all links and divides it by modularity.
    :param edge_encumbrance: EncumbrancePerLink dictionary
    :param modularity: integer value, one of the hyperparameters of the model
    :return: integer value, the total cost of the solution
    """
    return math.ceil(sum(edge_encumbrance.values()) / modularity)


def initialize_bee(db_context: ObjectsDB, distribution_strategy: int, as_ndarray: bool = True) -> BeeSpecimen:
    """
    Creates a bee specimen with random chromosome values. Genes distribution is based on the distribution_strategy parameter.
    In other words, distribution strategy is related to number of splits of the demand's value gene.
    :param db_context: ObjectsDB instance
    :param distribution_strategy: 0 for deterministic aggregation (always first top path), 1 for random aggregation, 2 and more for random disaggregation
    :param as_ndarray: bool Should the output be given as numpy array?
    :return: DataFrame | np.ndarray, bee chromosome representing the proposed solution
    """
    demands: dict[str, Demand] = db_context.get_demands()
    adm_paths: AdmisiblePaths = db_context.get_admisible_paths()
    max_paths: int = adm_paths.get_max_paths()
    data: pd.DataFrame = pd.DataFrame()
    first_demand: Demand = next(iter(demands.values()))
    path_names: list[str] = [path.get_path_id() for path in
                             adm_paths.get_paths_for_demand(first_demand.get_demand_id())]

    data['path_id'] = path_names

    for dmd_id in demands.keys():
        demand_value: int = int(demands[dmd_id].get_demand_value())
        demand_distribution: list[int] = _make_random_value_split(demand_value, max_paths, distribution_strategy)
        data[dmd_id] = demand_distribution

    # Decide if transform data from Pandas DataFrame to numpy array
    if as_ndarray:
        # Return just the numeric data as a NumPy array (excluding the 'path_id' column)
        # bees_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        return data.drop(columns=['path_id']).to_numpy()

    return data


def create_new_bee_population(db_context: ObjectsDB, distribution_strategy: int, population_size: int) -> BeePopulation:
    """
    Create a new random BeePopulation of size limited by population_size
    :param db_context: ObjectsDB instance
    :param population_size: max size of the BeePopulation
    :param distribution_strategy: 0 for deterministic aggregation (always first top path), 1 for random aggregation, 2 and more for random disaggregation
    :return: BeePopulation, a list of random bee specimens
    """
    bee_colony: BeePopulation = []
    for i in range(population_size):
        bee_colony.append(initialize_bee(db_context, distribution_strategy))

    return bee_colony


def assess_scouted_areas_quality(bee_scouts: BeePopulation, modularity: int, dpl_mapping: DPLMapping) -> tuple[
    BeePopulation, ScoutedAreas, ScoutedAreasQuality]:
    """
    Grades quality of scouted areas by entire bee population in the current phase.
    Higher grade means the potential solution is more expensive, so lesser grade is better.
    :param bee_scouts: list of bee specimens (bee scouts) in the current phase
    :param modularity: integer value, one of the hyperparameters of the model
    :param dpl_mapping: dictionary mapping demand_id to its path_ids and link_ids on them
    :return: tuple trio representing list of bee scouts, list of scouted areas and list of corresponding them grades
    """
    scouted_areas = [calculate_link_encumbrance(bee_scout, dpl_mapping) for bee_scout in bee_scouts]
    scouted_areas_quality_grades = [calculate_cost_function(area, modularity) for area in scouted_areas]

    return bee_scouts, scouted_areas, scouted_areas_quality_grades


def let_bee_search_locally(bee: BeeSpecimen, search_step: int, num_of_attempts: int, distribution_strategy: int,
                           demand_values: list[float], dpl_mapping: DPLMapping, modularity: int) -> list[BeeSearchArea]:
    """
    Performs local search and returns list of searched areas (BeeSpecimen, EncumbrancePerLink dict, solution_cost).

    '''
    Example for performing local search by a bee:
                         D1   D2   D3
                  Ap1 [[ 60,  90,  70 ],
                  Ap2  [ 20,   0,  20 ],
                  Ap3  [ 10,   0,   0 ]]

    BeeSpecimen =       [[ 60,  90,  70 ],
                        [  20,   0,  20 ],
                        [  10,   0,   0 ]]

    NewBeeSpecimen =    [[ 60,  80,  70 ],
                        [  30,  10,  10 ],
                        [   0,   0,  10 ]]
    '''
    :param bee: BeeSpecimen to be mutated (perturbed)
    :param search_step: integer value between 1 and number demands. Represents how many demands to rearrange
    :param num_of_attempts: integer value showing how many local searches to perform
    :param distribution_strategy: integer value between 1 and number of admissible paths per demand.
        Determines how many times to split the demand value
    :param modularity: integer hyperparameter
    :param dpl_mapping: Demand to Path to Links mapping
    :param demand_values: list of demand values
    :return: BeeSpecimen with slightly modified demands distributions obtained as a result of local search parameters
    """

    searched_areas: list[BeeSearchArea] = []

    for i in range(num_of_attempts):
        new_bee: BeeSpecimen = bee.copy()
        num_paths, num_demands = bee.shape
        search_step = min(search_step, num_demands)  # Cap at number of demands

        # Select random demands to perturb
        demands_to_perturb = random.sample(range(num_demands), search_step)

        if distribution_strategy == 1:  # Reassign entire demand to one path
            for col_idx in demands_to_perturb:
                new_bee[:, col_idx] = _make_random_value_split(int(demand_values[col_idx]), num_paths,
                                                               n_splits=distribution_strategy)
        elif distribution_strategy >= 2:  # Randomly split demand across paths
            for col_idx in demands_to_perturb:
                n_splits = random.randint(2, num_paths)  # At least 2 paths
                new_bee[:, col_idx] = _make_random_value_split(int(demand_values[col_idx]), num_paths, n_splits)
        else:
            raise ValueError("Invalid distribution_strategy: Use 1 (single path) or >= 2 (split).")

        edges_encumbrance: EncumbrancePerLink = calculate_link_encumbrance(new_bee, dpl_mapping)
        searched_area: BeeSearchArea = new_bee, edges_encumbrance, calculate_cost_function(edges_encumbrance,
                                                                                           modularity)
        searched_areas.append(searched_area)

    return searched_areas


hyperparameters_dict: dict = {
    "bee_population_size": 20,
    "max_iter": 200,
    "dist_strategy": 1,
    "modularity": 5,
    "pro_search_areas": 5,
    "n_standard_probes": 1,
    "elite_areas": 2,
    "n_elite_probes": 3,
    "local_search_step": 1,
    "k_iter_no_improv": 5
}

BestSearchHistory: TypeAlias = list[BeeSearchArea]


def train_model(db_context: ObjectsDB, hyperparams: dict) -> tuple[BeeSpecimen, int, BestSearchHistory]:
    DPL_MAPPING: DPLMapping = db_context.create_demands_to_paths_to_links_map()
    DEMAND_VALUES: list[float] = db_context.get_demands_values()

    iteration: int = 1
    no_improv_count: int = 0
    best_solution: BeeSpecimen | None = None
    best_solution_cost: int = sys.maxsize
    solution_history: BestSearchHistory = []
    # last_n_best_solutions: list[BeeSpecimen] = []  # n last best bee specimens with the lowest cost function

    # 1. Initial population. Scouting phase I (send scouts).
    # Determine n_bees, pr'_prospect_areas, top_e_areas.
    # Randomly assign search areas for bee population (scouts)
    current_hive: BeePopulation = create_new_bee_population(db_context, hyperparams["dist_strategy"],
                                                            hyperparams["bee_population_size"])

    # 2. Assessment phase I (assess solutions' costs)
    # Assess scouts solutions with assessment function. Choose pr best search areas for next step.
    cur_hive_areas: BeeHiveSearchAreas
    cur_hive_areas = list(zip(*assess_scouted_areas_quality(current_hive, hyperparams["modularity"], DPL_MAPPING)))
    top_scouted_areas: BeeHiveSearchAreas = sorted(cur_hive_areas, key=lambda x: x[2])[
                                            :hyperparams["pro_search_areas"] + 1]  # local search
    elite_areas: BeeHiveSearchAreas = top_scouted_areas[:hyperparams["elite_areas"] + 1]  # intensified local search
    remaining_pro_areas: BeeHiveSearchAreas = top_scouted_areas[hyperparams["elite_areas"] + 1:]  # normal local search

    solution_history.append(elite_areas[0])

    if best_solution_cost > elite_areas[0][2]:
        best_solution = elite_areas[0][0]
        best_solution_cost = elite_areas[0][2]

    while iteration < hyperparams["max_iter"] and no_improv_count < hyperparams["k_iter_no_improv"]:
        iteration += 1
        # 3. (Local) Exploration phase I (assign more bees for top pr areas -> top e from them are explored much stronger)
        # Assign additional bees for pr best areas. Result of this phase is nep * e + nsp * (pr - e) + pr solutions
        local_searched_areas: BeeHiveSearchAreas = elite_areas + remaining_pro_areas

        ### Intensified local search ###
        for start_area in elite_areas:
            local_searched_areas += let_bee_search_locally(
                bee=start_area[0],
                search_step=hyperparams["local_search_step"],
                num_of_attempts=hyperparams["n_elite_probes"],
                distribution_strategy=hyperparams["dist_strategy"],
                demand_values=DEMAND_VALUES,
                dpl_mapping=DPL_MAPPING,
                modularity=hyperparams["modularity"]
            )

        ### Normal local search ###
        for start_area in remaining_pro_areas:
            local_searched_areas += let_bee_search_locally(
                bee=start_area[0],
                search_step=hyperparams["local_search_step"],
                num_of_attempts=hyperparams["n_standard_probes"],
                distribution_strategy=hyperparams["dist_strategy"],
                demand_values=DEMAND_VALUES,
                dpl_mapping=DPL_MAPPING,
                modularity=hyperparams["modularity"]
            )

        # 4. Selection phase I: from local explored areas (nep * e + nsp * (pr - e) + pr),
        # select 'e' the best ones which will be exploited in next iteration
        # Sort and choose e top solutions -> copy them to the next round
        selected_elite_areas: BeeHiveSearchAreas = sorted(local_searched_areas, key=lambda x: x[2])[:hyperparams["elite_areas"]+1]

        # 5. (Global) Exploration / Scouting phase II (bees unassigned to top best areas are randomly assigned to new areas)
        # Keep elite bees from step 4 and generate (n - e) random bees.
        current_hive.clear()
        for area in selected_elite_areas:
            current_hive.append(area[0])
        current_hive += create_new_bee_population(db_context, hyperparams["dist_strategy"], hyperparams["bee_population_size"] - len(selected_elite_areas))

        # 6. Assessment phase II (assess solutions' quality - similar to step 2)
        # Assess scouted areas again sorting them by solution cost ascending.
        cur_hive_areas = list(zip(*assess_scouted_areas_quality(current_hive, hyperparams["modularity"], DPL_MAPPING)))
        top_scouted_areas = sorted(cur_hive_areas, key=lambda x: x[2])[:hyperparams["pro_search_areas"] + 1]

        # 7. increase iteration and check stop_condition. If stop_condition == True, finish, otherwise continue
        elite_areas: BeeHiveSearchAreas = top_scouted_areas[:hyperparams["elite_areas"] + 1]  # intensified local search
        remaining_pro_areas: BeeHiveSearchAreas = top_scouted_areas[hyperparams["elite_areas"] + 1:]

        if best_solution_cost > elite_areas[0][2]:
            no_improv_count = 0
            best_solution = elite_areas[0][0]
            best_solution_cost = elite_areas[0][2]
        else:
            no_improv_count += 1
            if no_improv_count == hyperparams["k_iter_no_improv"]:
                break

        solution_history.append(elite_areas[0])

    # Return top bee_chromosome and cost_value
    return best_solution, best_solution_cost, solution_history
