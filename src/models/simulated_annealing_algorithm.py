import math
import random
import sys
from typing import TypeAlias

import numpy as np

from models.bee_algorithm import (
    initialize_bee,
    calculate_link_encumbrance,
    calculate_cost_function,
    _make_random_value_split
)
from structures.ObjectsDB import ObjectsDB

# --- Definicje typów dla spójności z projektem ---
BeeSpecimen: TypeAlias = np.ndarray
DPLMapping: TypeAlias = dict[str, dict[str, list[str]]]
SolutionHistory: TypeAlias = list[int]

# --- Konfiguracja i hiperparametry algorytmu ---

hyperparameters_sa: dict = {
    "T_max": 1000.0,
    "T_min": 0.1,
    "cooling_rate": 0.995,
    "max_iter_per_temp": 50,
    "neighbor_intensity": 1,
    "initial_solution_strategy": 2
}
# hyperparameters_sa: dict = {
#     "T_max": 20000.0,  # <-- Znacznie wyższa temperatura
#     "T_min": 0.1,
#     "cooling_rate": 0.995,
#     "max_iter_per_temp": 50,
#     "neighbor_intensity": 2, # <-- Lekko zwiększona intensywność sąsiada
#     "initial_solution_strategy": 2
# }


def _generate_neighbor_solution(current_solution: BeeSpecimen, demand_values: list[float],
                                intensity: int) -> BeeSpecimen:
    """
    Generuje rozwiązanie sąsiednie poprzez losową zmianę dystrybucji
    dla określonej liczby zapotrzebowań (genów).

    :param current_solution: Aktualne rozwiązanie (chromosom).
    :param demand_values: Lista wartości dla wszystkich zapotrzebowań.
    :param intensity: Liczba zapotrzebowań do modyfikacji.
    :return: Nowe, sąsiednie rozwiązanie.
    """
    neighbor = current_solution.copy()
    num_paths, num_demands = neighbor.shape

    # Wybierz losowe zapotrzebowania (kolumny) do zmiany
    demands_to_perturb = random.sample(range(num_demands), min(intensity, num_demands))

    for col_idx in demands_to_perturb:
        # Stwórz nową, losową dystrybucję wartości dla wybranego zapotrzebowania
        num_splits = random.randint(1, num_paths)
        new_distribution = _make_random_value_split(
            total_value=int(demand_values[col_idx]),
            list_size=num_paths,
            n_splits=num_splits
        )
        neighbor[:, col_idx] = new_distribution

    return neighbor


def train_simulated_annealing(db_context: ObjectsDB, hyperparams: dict) -> tuple[BeeSpecimen, int, SolutionHistory]:
    """
    Główna funkcja implementująca algorytm Symulowanego Wyżarzania
    w celu znalezienia optymalnego projektu sieci.
    """
    # --- Krok 0: Przygotowanie danych ---
    DPL_MAPPING: DPLMapping = db_context.create_demands_to_paths_to_links_map()
    DEMAND_VALUES: list[float] = db_context.get_demands_values()

    # --- Krok 1: Inicjalizacja ---
    # Wybierz losowe rozwiązanie początkowe
    current_solution = initialize_bee(db_context, hyperparams["initial_solution_strategy"])

    # Oblicz koszt bieżącego rozwiązania
    encumbrance = calculate_link_encumbrance(current_solution, DPL_MAPPING)
    current_cost = calculate_cost_function(encumbrance, 5)  # Uwaga: modularność 'm' na stałe

    best_solution = current_solution
    best_cost = current_cost

    # Ustal temperaturę początkową
    T = hyperparams["T_max"]

    solution_history = [current_cost]

    # --- Główna pętla algorytmu ---
    while T > hyperparams["T_min"]:
        for _ in range(hyperparams["max_iter_per_temp"]):
            # --- Kroki 3 i 4: Generowanie i ocena sąsiada ---
            # Wygeneruj nowe rozwiązanie j (sąsiada)
            neighbor_solution = _generate_neighbor_solution(current_solution, DEMAND_VALUES,
                                                            hyperparams["neighbor_intensity"])

            # Oblicz koszt nowego rozwiązania f(j)
            neighbor_encumbrance = calculate_link_encumbrance(neighbor_solution, DPL_MAPPING)
            neighbor_cost = calculate_cost_function(neighbor_encumbrance, 5)  # Uwaga: modularność 'm' na stałe

            # --- Krok 5: Akceptacja nowego rozwiązania ---
            cost_delta = neighbor_cost - current_cost

            # Jeśli nowe rozwiązanie jest lepsze, akceptuj je zawsze
            if cost_delta < 0:
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                # Sprawdź, czy to najlepsze dotychczasowe rozwiązanie globalne
                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost
            # Jeśli jest gorsze, zaakceptuj je z pewnym prawdopodobieństwem
            else:
                acceptance_probability = math.exp(-cost_delta / T)
                if random.random() < acceptance_probability:
                    current_solution = neighbor_solution
                    current_cost = neighbor_cost

            solution_history.append(best_cost)

        # --- Krok 6: Chłodzenie ---
        # Zmniejsz wartość parametru T
        T *= hyperparams["cooling_rate"]

    # --- Krok 7: Zakończenie ---
    # Zwróć najlepsze znalezione rozwiązanie
    return best_solution, best_cost, solution_history