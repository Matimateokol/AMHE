import math
import random
from typing import TypeAlias
import numpy as np
from models.bee_algorithm import (
    initialize_bee, calculate_link_encumbrance,
    calculate_cost_function, _make_random_value_split, BestSearchHistory, IterationCostHistory
)
from structures.ObjectsDB import ObjectsDB

BeeSpecimen: TypeAlias = np.ndarray

hyperparameters_sa: dict = {
    "T_max": 1000.0,
    "T_min": 0.1,
    "cooling_rate": 0.995,
    "max_iter_per_temp": 50,
    "neighbor_intensity": 1,
    "dist_strategy": 7,
    "modularity": 1
}


def _generate_neighbor_solution(current_solution: BeeSpecimen, demand_values: list[float],
                                intensity: int) -> BeeSpecimen:
    """
    Generuje rozwiązanie sąsiednie poprzez losową, niewielką modyfikację (mutację)
    dla określonej liczby zapotrzebowań (genów).
    :param current_solution: Aktualne rozwiązanie (chromosom).
    :param demand_values: Lista wartości dla wszystkich zapotrzebowań.
    :param intensity: Liczba zapotrzebowań do modyfikacji.
    :return: Nowe, sąsiednie rozwiązanie.
    """
    neighbor = current_solution.copy()
    num_paths, num_demands = neighbor.shape
    demands_to_perturb = random.sample(range(num_demands), min(intensity, num_demands))
    for col_idx in demands_to_perturb:
        num_splits = random.randint(1, num_paths)
        new_distribution = _make_random_value_split(
            int(demand_values[col_idx]), num_paths, num_splits
        )
        neighbor[:, col_idx] = new_distribution
    return neighbor

def train_simulated_annealing(db_context: ObjectsDB, hyperparams: dict) -> tuple[
    BeeSpecimen, int, BestSearchHistory, int, IterationCostHistory]:
    """
    Implementuje algorytm Symulowanego Wyżarzania.
    Zwraca ujednolicony format wyników, zgodny z algorytmem pszczelim.
    """

    DPL_MAPPING = db_context.create_demands_to_paths_to_links_map()
    DEMAND_VALUES = db_context.get_demands_values()


    current_solution = initialize_bee(db_context, hyperparams["dist_strategy"])
    current_cost = calculate_cost_function(
        calculate_link_encumbrance(current_solution, DPL_MAPPING), hyperparams["modularity"]
    )


    best_solution, best_cost = current_solution, current_cost

    T = hyperparams["T_max"]

    best_solution_history: BestSearchHistory = [(best_solution, 0, best_cost)]
    iteration_cost_history: IterationCostHistory = [current_cost]
    iterations = 0


    while T > hyperparams["T_min"]:
        for _ in range(hyperparams["max_iter_per_temp"]):
            iterations += 1

            neighbor_solution = _generate_neighbor_solution(current_solution, DEMAND_VALUES,
                                                            hyperparams["neighbor_intensity"])

            neighbor_cost = calculate_cost_function(
                calculate_link_encumbrance(neighbor_solution, DPL_MAPPING), hyperparams["modularity"]
            )

            if neighbor_cost < current_cost or random.random() < math.exp(-(neighbor_cost - current_cost) / T):
                current_solution, current_cost = neighbor_solution, neighbor_cost


                if current_cost < best_cost:
                    best_solution, best_cost = current_solution, current_cost
                    best_solution_history.append((best_solution, iterations, best_cost))
                    print(f"Iter {iterations}: New best cost found: {best_cost}")


            iteration_cost_history.append(current_cost)


        T *= hyperparams["cooling_rate"]


    return best_solution, best_cost, best_solution_history, iterations, iteration_cost_history
