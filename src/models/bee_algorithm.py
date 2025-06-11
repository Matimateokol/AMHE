import sys
from typing import TypeAlias

from structures.ObjectsDB import ObjectsDB
from structures.AdmisiblePaths import AdmisiblePaths
from structures.Demand import Demand
import numpy as np
import math
import random
import pandas as pd
from collections import defaultdict

# ### Custom type-aliases ###
DPLMapping: TypeAlias = dict[str, dict[str, list[str]]]
BeeSpecimen: TypeAlias = np.ndarray
BeeHive: TypeAlias = list[BeeSpecimen]
BeeSearchArea: TypeAlias = tuple[BeeSpecimen, int, int]
BeeHiveSearchAreas: TypeAlias = list[BeeSearchArea]
BestSearchHistory: TypeAlias = list[tuple[BeeSpecimen, int, int]]

# ### Algorithm's hyperparameters ###
hyperparameters_dict = {
    "bee_population_size": 20,
    "elite_areas": 2,
    "pro_search_areas": 5,
    "elite_area_bees": 4,
    "pro_area_bees": 2,
    "dist_strategy": 2,
    "k_iter_no_improv": 1000,
    "max_iter": 50000,
    "modularity": 5
}


def _make_random_value_split(total_value: int, list_size: int, n_splits: int) -> list[int]:
    """
    Splits a total_value into n_splits parts and distributes them randomly
    across a list of list_size.
    """
    if n_splits <= 0 or total_value < 1:
        return [0] * list_size

    # --- POPRAWKA TUTAJ ---
    # Zapewnia, że liczba części nie jest większa niż wartość do podziału.
    n_splits = min(n_splits, total_value)

    if n_splits == 1:
        parts = [total_value]
    else:
        split_points = sorted(random.sample(range(1, total_value), n_splits - 1))
        parts = []
        last_split = 0
        for split in split_points:
            parts.append(split - last_split)
            last_split = split
        parts.append(total_value - last_split)

    values = [0] * list_size
    indices = random.sample(range(list_size), n_splits)
    for i, value in zip(indices, parts):
        values[i] = value

    return values


def initialize_bee(db: ObjectsDB, distribution_strategy: int, as_ndarray: bool = False) -> pd.DataFrame | np.ndarray:
    """
    Creates an initial solution (bee) by assigning demand values to admissible paths.
    """
    demands: dict[str, Demand] = db.get_demands()
    adm_paths: AdmisiblePaths = db.get_admisible_paths()
    if adm_paths is None or not adm_paths.get_paths():
        raise ValueError(
            "Brak dopuszczalnych ścieżek (AdmissiblePaths) w bazie danych. Uruchom build_features.py, aby je wygenerować.")

    max_paths = adm_paths.get_max_paths()

    # --- Optymalizacja wydajności (unikanie fragmentacji DataFrame) ---
    all_distributions = {}

    for dmd_id, demand_obj in demands.items():
        demand_value = int(demand_obj.get_demand_value())
        num_paths_for_demand = len(adm_paths.get_paths_for_demand(dmd_id))

        if distribution_strategy == 0:
            n_splits = 1
        elif distribution_strategy == 1:
            n_splits = 1
        else:
            max_splits = min(distribution_strategy, num_paths_for_demand)
            if max_splits > 0:
                n_splits = random.randint(1, max_splits)
            else:
                n_splits = 0

        demand_distribution = _make_random_value_split(demand_value, num_paths_for_demand, n_splits)
        all_distributions[dmd_id] = demand_distribution + [0] * (max_paths - num_paths_for_demand)

    path_names = [f"P_{i}" for i in range(max_paths)]
    data = pd.DataFrame(all_distributions, index=path_names)

    return data.to_numpy() if as_ndarray else data


def create_new_bee_population(db: ObjectsDB, dist_strategy: int, population_size: int) -> BeeHive:
    """Creates a new population of bees (solutions)."""
    return [initialize_bee(db, dist_strategy, True) for _ in range(population_size)]


def calculate_link_encumbrance(bee_specimen: BeeSpecimen, dpl_mapping: DPLMapping) -> dict[str, float]:
    """Calculates the total load (encumbrance) for each link in the network."""
    encumbrance_per_link = defaultdict(float)
    demand_ids = list(dpl_mapping.keys())

    for demand_idx, demand_id in enumerate(demand_ids):
        path_distributions = bee_specimen[:, demand_idx]
        paths_for_demand = dpl_mapping[demand_id]

        for path_idx, path_id in enumerate(paths_for_demand.keys()):
            traffic_on_path = path_distributions[path_idx]
            if traffic_on_path > 0:
                for link_id in paths_for_demand[path_id]:
                    encumbrance_per_link[link_id] += traffic_on_path

    return encumbrance_per_link


def calculate_cost_function(encumbrance_per_link: dict[str, float], modularity: int) -> int:
    """Calculates the total network cost based on link encumbrance and modularity."""
    total_cost = 0
    for link_id, total_load in encumbrance_per_link.items():
        if total_load > 0:
            num_modules = math.ceil(total_load / modularity)
            total_cost += num_modules
    return total_cost


def assess_scouted_areas_quality(scouted_areas: BeeHive, modularity: int, dpl_mapping: DPLMapping) -> tuple[
    BeeHive, list[int], list[int]]:
    """Assesses the quality (cost) of each solution in the hive."""
    solutions_costs = []
    encumbrances = []
    for bee in scouted_areas:
        encumbrance = calculate_link_encumbrance(bee, dpl_mapping)
        solutions_costs.append(calculate_cost_function(encumbrance, modularity))
        encumbrances.append(encumbrance)
    return scouted_areas, encumbrances, solutions_costs


def let_bee_search_locally(scouted_areas: BeeHive, bees_per_area: int, db_context: ObjectsDB,
                           dist_strategy: int) -> BeeHive:
    """Performs local search around the best found solutions."""
    return create_new_bee_population(db_context, dist_strategy, len(scouted_areas) * bees_per_area)


def train_model(db_context: ObjectsDB, hyperparams: dict) -> tuple[BeeSpecimen, int, BestSearchHistory]:
    """Main function for the Bee Algorithm."""
    DPL_MAPPING: DPLMapping = db_context.create_demands_to_paths_to_links_map()
    best_solution_history: BestSearchHistory = []

    current_hive: BeeHive = create_new_bee_population(db_context, hyperparams["dist_strategy"],
                                                      hyperparams["bee_population_size"])

    best_solution = None
    best_solution_cost = float('inf')
    no_improv_count = 0

    for iteration in range(hyperparams["max_iter"]):
        cur_hive_areas = list(zip(*assess_scouted_areas_quality(current_hive, hyperparams["modularity"], DPL_MAPPING)))
        top_scouted_areas = sorted(cur_hive_areas, key=lambda x: x[2])[:hyperparams["pro_search_areas"] + 1]

        elite_areas: BeeHiveSearchAreas = top_scouted_areas[:hyperparams["elite_areas"]]
        remaining_pro_areas: BeeHiveSearchAreas = top_scouted_areas[hyperparams["elite_areas"]:]

        elite_bees = [bee for bee, _, _ in elite_areas]
        pro_bees = [bee for bee, _, _ in remaining_pro_areas]

        elite_search_results = let_bee_search_locally(elite_bees, hyperparams["elite_area_bees"], db_context,
                                                      hyperparams["dist_strategy"])
        pro_search_results = let_bee_search_locally(pro_bees, hyperparams["pro_area_bees"], db_context,
                                                    hyperparams["dist_strategy"])

        scout_bees_count = hyperparams["bee_population_size"] - len(elite_search_results) - len(pro_search_results)
        scout_bees = create_new_bee_population(db_context, hyperparams["dist_strategy"],
                                               scout_bees_count) if scout_bees_count > 0 else []

        current_hive = elite_search_results + pro_search_results + scout_bees

        final_assessment = list(
            zip(*assess_scouted_areas_quality(current_hive, hyperparams["modularity"], DPL_MAPPING)))
        if not final_assessment: continue  # Pomiń iterację, jeśli ocena jest pusta

        current_best_solution, _, current_best_cost = min(final_assessment, key=lambda x: x[2])

        if current_best_cost < best_solution_cost:
            no_improv_count = 0
            best_solution = current_best_solution
            best_solution_cost = current_best_cost
            best_solution_history.append((best_solution, iteration, best_solution_cost))
            print(f"Iter {iteration}: New best cost found: {best_solution_cost}")
        else:
            no_improv_count += 1
            if no_improv_count >= hyperparams["k_iter_no_improv"]:
                print(f"Stopping early after {no_improv_count} iterations with no improvement.")
                break

    return best_solution, best_solution_cost, best_solution_history
