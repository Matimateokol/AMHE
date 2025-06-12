# Input is 2-D array, example:
#                          D1  D2  D3
#                   Ap1 [ 60  90  70
#                   Ap2   20   0  20
#                   Ap3   10   0   0 ]
# Potential food sources => Admisible paths (m')
# Number of best food sources => Best admisible paths to exploit (e)
# Quality assesment => Number of nodes (jumps) in the path - the less is better or loss function / target function (less is better).
# Hive => Demand (Di)
# Bees => Demand value (n)

# 1st strategy (aggregation) => all bees assigned to the best food source
# 2nd strategy (disaggregation) => some bees assigned to the best food source and rest randomly to the remaining ones

import sys
import random
import math
from typing import TypeAlias, List
import numpy as np
import pandas as pd
from collections import defaultdict

from structures.ObjectsDB import ObjectsDB
from structures.AdmisiblePaths import AdmisiblePaths
from structures.Demand import Demand

# --- Typy danych ---
DPLMapping: TypeAlias = dict[str, dict[str, list[str]]]
EncumbrancePerLink: TypeAlias = dict[str, int]
BeeSpecimen: TypeAlias = np.ndarray
BeePopulation: TypeAlias = list[BeeSpecimen]
BeeSearchArea: TypeAlias = tuple[BeeSpecimen, EncumbrancePerLink, int]
BestSearchHistory: TypeAlias = list[tuple[BeeSpecimen, int, int]]
IterationCostHistory: TypeAlias = list[int]

# --- Hiperparametry ---
hyperparameters_dict: dict = {
    "bee_population_size": 20,
    "max_iter": 100000,
    "dist_strategy": 3,
    "pro_search_areas": 5,
    "n_standard_probes": 1,
    "elite_areas": 2,
    "n_elite_probes": 3,
    "local_search_step": 1,
    "k_iter_no_improv": 20,
    "modularity": 1,
}


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
    if total_value < 1 or n_splits <= 0: return [0] * list_size
    if n_splits == 1:
        values = [0] * list_size
        if list_size > 0: values[random.randint(0, list_size - 1)] = total_value
        return values
    else:
        n_splits = min(n_splits, total_value)
        if n_splits <= 1:
            values = [0] * list_size
            if list_size > 0: values[random.randint(0, list_size - 1)] = total_value
            return values

        split_points = sorted(random.sample(range(1, total_value), n_splits - 1))
        parts = [split_points[0]] + [split_points[i + 1] - split_points[i] for i in range(len(split_points) - 1)] + [
            total_value - split_points[-1]]
        values = [0] * list_size
        indices = random.sample(range(list_size), n_splits)
        for i, value in zip(indices, parts): values[i] = value
        return values


def calculate_link_encumbrance(bee_chromosome: BeeSpecimen, dpl_mapping: DPLMapping) -> EncumbrancePerLink:
    """
    Calculate the total encumbrance on each Link used to deliver Demands in given network design.
    :param bee_chromosome: 2-D numpy array of shape [path_id, demand_id] or pandas dataframe representing solution
    :param dpl_mapping: dictionary mapping demand_id to its path_ids and link_ids on them
    :return: link_id to encumbrance value mapping
    """
    if not dpl_mapping: return defaultdict(int)
    path_ids_map = next(iter(dpl_mapping.values()), {})
    if not path_ids_map: return defaultdict(int)
    path_ids = list(path_ids_map.keys())

    demand_ids = list(dpl_mapping.keys())
    edge_encumbrance: EncumbrancePerLink = defaultdict(int)

    for col_idx, demand_id in enumerate(demand_ids):
        path_to_links = dpl_mapping[demand_id]
        demand_values = bee_chromosome[:, col_idx]
        for row_idx, path_id in enumerate(path_ids):
            if row_idx < len(demand_values) and path_id in path_to_links and demand_values[row_idx] > 0:
                for link in path_to_links[path_id]:
                    edge_encumbrance[link] += demand_values[row_idx]
    return edge_encumbrance


def calculate_cost_function(edge_encumbrance: EncumbrancePerLink, modularity: int) -> int:
    """
    Calculates the total cost of a given solution. It sums up encumbrance on all links and divides it by modularity.
    :param edge_encumbrance: EncumbrancePerLink dictionary
    :param modularity: integer value, one of the hyperparameters of the model
    :return: integer value, the total cost of the solution
    """
    return sum(math.ceil(load / modularity) for load in edge_encumbrance.values() if load > 0)


def initialize_bee(db_context: ObjectsDB, distribution_strategy: int) -> BeeSpecimen:
    """
    Creates a bee specimen with random chromosome values. Genes distribution is based on the distribution_strategy parameter.
    In other words, distribution strategy is related to number of splits of the demand's value gene.
    :param db_context: ObjectsDB instance
    :param distribution_strategy: 0 for deterministic aggregation (always first top path), 1 for random aggregation, 2 and more for random disaggregation
    :return: DataFrame | np.ndarray, bee chromosome representing the proposed solution
    """
    demands, adm_paths = db_context.get_demands(), db_context.get_admisible_paths()
    max_paths = adm_paths.get_max_paths()
    all_distributions = {}
    for dmd_id, demand_obj in demands.items():
        demand_value = int(demand_obj.get_demand_value())
        num_paths = len(adm_paths.get_paths_for_demand(dmd_id))
        demand_distribution = _make_random_value_split(demand_value, num_paths, distribution_strategy)
        all_distributions[dmd_id] = demand_distribution + [0] * (max_paths - num_paths)
    return pd.DataFrame(all_distributions).to_numpy()


def create_new_bee_population(db_context: ObjectsDB, distribution_strategy: int, population_size: int) -> BeePopulation:
    """
    Create a new random BeePopulation of size limited by population_size
    :param db_context: ObjectsDB instance
    :param population_size: max size of the BeePopulation
    :param distribution_strategy: 0 for deterministic aggregation (always first top path), 1 for random aggregation, 2 and more for random disaggregation
    :return: BeePopulation, a list of random bee specimens
    """
    return [initialize_bee(db_context, distribution_strategy) for _ in range(population_size)]


def assess_scouted_areas_quality(bee_scouts: BeePopulation, modularity: int, dpl_mapping: DPLMapping) -> List[
    BeeSearchArea]:
    """
    Grades quality of scouted areas by entire bee population in the current phase.
    Higher grade means the potential solution is more expensive, so lesser grade is better.
    :param bee_scouts: list of bee specimens (bee scouts) in the current phase
    :param modularity: integer value, one of the hyperparameters of the model
    :param dpl_mapping: dictionary mapping demand_id to its path_ids and link_ids on them
    :return: tuple trio representing list of bee scouts, list of scouted areas and list of corresponding them grades
    """
    results = []
    for bee in bee_scouts:
        encumbrance = calculate_link_encumbrance(bee, dpl_mapping)
        cost = calculate_cost_function(encumbrance, modularity)
        results.append((bee, encumbrance, cost))
    return results


def let_bee_search_locally(bee: BeeSpecimen, search_step: int, num_of_attempts: int, demand_values: list[float],
                           dpl_mapping: DPLMapping, modularity: int) -> List[BeeSearchArea]:
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
    :param demand_values: list of demand values
    :param dpl_mapping: Demand to Path to Links mapping
    :param modularity: integer hyperparameter
    :return: BeeSpecimen with slightly modified demands distributions obtained as a result of local search parameters
    """
    searched_areas: List[BeeSearchArea] = []
    for _ in range(num_of_attempts):
        new_bee = bee.copy()
        num_paths, num_demands = bee.shape
        if num_demands == 0: continue
        demands_to_perturb = random.sample(range(num_demands), min(search_step, num_demands))
        for col_idx in demands_to_perturb:
            n_splits = random.randint(1, num_paths)
            new_bee[:, col_idx] = _make_random_value_split(int(demand_values[col_idx]), num_paths, n_splits)
        encumbrance = calculate_link_encumbrance(new_bee, dpl_mapping)
        cost = calculate_cost_function(encumbrance, modularity)
        searched_areas.append((new_bee, encumbrance, cost))
    return searched_areas


def train_model(db_context: ObjectsDB, hyperparams: dict) -> tuple[
    BeeSpecimen, int, BestSearchHistory, int, IterationCostHistory]:
    """Główna funkcja trenująca model, oparta na logice hybrydowej."""
    # --- PRZYGOTOWANIE ---
    DPL_MAPPING = db_context.create_demands_to_paths_to_links_map()
    DEMAND_VALUES = db_context.get_demands_values()

    best_solution, best_solution_cost = None, sys.maxsize
    best_solution_history: BestSearchHistory = []
    iteration_cost_history: IterationCostHistory = []
    no_improv_count = 0
    iterations = 0

    # --- KROK 1: Inicjalizacja ---
    # Stworzenie pierwszej, losowej populacji pszczół (rozwiązań).
    current_hive = create_new_bee_population(db_context, hyperparams["dist_strategy"],
                                             hyperparams["bee_population_size"])
    # Ocena jakości (kosztu) każdego rozwiązania w początkowej populacji.
    assessed_hive = assess_scouted_areas_quality(current_hive, hyperparams["modularity"], DPL_MAPPING)

    # --- GŁÓWNA PĘTLA ALGORYTMU ---
    for iteration in range(hyperparams["max_iter"]):
        iterations = iteration
        if not assessed_hive: break

        # --- KROK 2: Wybór najlepszych obszarów ---
        # Posortuj ocenione rozwiązania od najlepszego (najniższy koszt) do najgorszego.
        assessed_hive.sort(key=lambda x: x[2])
        # Wybierz 'e' najlepszych rozwiązań do intensywnego przeszukiwania (obszary elitarne).
        elite_areas = assessed_hive[:hyperparams["elite_areas"]]
        # Wybierz kolejne 'p' rozwiązań do standardowego przeszukiwania (obszary obiecujące).
        pro_areas = assessed_hive[hyperparams["elite_areas"]:hyperparams["pro_search_areas"]]

        # --- KROK 3: Faza Eksploatacji (Przeszukiwanie Lokalne) ---
        # Stwórz listę kandydatów do dalszej analizy, składającą się z już znalezionych elitarnych i obiecujących rozwiązań.
        local_searched_areas = elite_areas + pro_areas
        # Dla każdego obszaru elitarnego, wykonaj intensywne przeszukiwanie lokalne (więcej prób mutacji).
        for area in elite_areas:
            local_searched_areas.extend(
                let_bee_search_locally(area[0], hyperparams["local_search_step"], hyperparams["n_elite_probes"],
                                       DEMAND_VALUES, DPL_MAPPING, hyperparams["modularity"]))
        # Dla każdego obszaru obiecującego, wykonaj standardowe przeszukiwanie lokalne (mniej prób mutacji).
        for area in pro_areas:
            local_searched_areas.extend(
                let_bee_search_locally(area[0], hyperparams["local_search_step"], hyperparams["n_standard_probes"],
                                       DEMAND_VALUES, DPL_MAPPING, hyperparams["modularity"]))

        if not local_searched_areas: break

        # --- KROK 4: Wybór nowej elity ---
        # Przesortuj wszystkie dotychczasowe i nowo wygenerowane (zmutowane) rozwiązania.
        local_searched_areas.sort(key=lambda x: x[2])
        # Wybierz nową, najlepszą elitę, która przejdzie do następnej generacji.
        new_elite = local_searched_areas[:hyperparams["elite_areas"]]

        # Zapisz koszt najlepszego rozwiązania z bieżącej iteracji
        if new_elite:
            iteration_cost_history.append(new_elite[0][2])

        # --- KROK 5: Aktualizacja globalnie najlepszego rozwiązania ---
        if new_elite and new_elite[0][2] < best_solution_cost:
            no_improv_count = 0
            best_solution, _, best_solution_cost = new_elite[0]
            best_solution_history.append((best_solution, iteration, best_solution_cost))
            print(f"Iter {iteration}: New best cost found: {best_solution_cost}")
        else:
            no_improv_count += 1
            if no_improv_count >= hyperparams["k_iter_no_improv"]:
                print(f"Stopping early after {no_improv_count} iterations with no improvement.")
                break

        # --- KROK 6: Faza Eksploracji (Tworzenie nowej populacji) ---
        # Stwórz nową populację: zachowaj nową elitę, a resztę zastąp zupełnie nowymi, losowymi pszczołami.
        current_hive = [bee[0] for bee in new_elite]
        current_hive.extend(create_new_bee_population(db_context, hyperparams["dist_strategy"],
                                                      hyperparams["bee_population_size"] - len(new_elite)))

        # --- KROK 7: Ocena nowej populacji ---
        # Oceń nowo stworzoną populację. Wynik tej oceny zostanie użyty w następnej iteracji pętli.
        assessed_hive = assess_scouted_areas_quality(current_hive, hyperparams["modularity"], DPL_MAPPING)

    return best_solution, best_solution_cost, best_solution_history, iterations, iteration_cost_history
