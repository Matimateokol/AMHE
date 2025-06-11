# ==============================================================================
# SEKCJA 1: KOMENTARZE WYJAŚNIAJĄCE KONCEPCJĘ ALGORYTMU
# ==============================================================================

# Algorytm Symulowanego Wyżarzania (Simulated Annealing - SA) to metaheurystyka
# inspirowana procesem wyżarzania w metalurgii. Polega on na podgrzaniu materiału
# do wysokiej temperatury, a następnie powolnym go schładzaniu, co pozwala
# na osiągnięcie stanu o minimalnej energii (stabilnej struktury krystalicznej).

# W kontekście optymalizacji:
# - Stan systemu => Potencjalne rozwiązanie problemu.
# - Energia systemu => Wartość funkcji kosztu dla danego rozwiązania.
# - Temperatura (T) => Parametr kontrolujący prawdopodobieństwo akceptacji gorszych rozwiązań.
# - Chłodzenie => Stopniowe zmniejszanie temperatury.

# Kluczową cechą algorytmu jest akceptowanie z pewnym prawdopodobieństwem rozwiązań
# gorszych od bieżącego. Pozwala to na "ucieczkę" z minimów lokalnych i dokładniejsze
# przeszukanie przestrzeni w poszukiwaniu globalnego optimum.

import math
import random
from typing import TypeAlias
import numpy as np

# Importy z innych modułów
from models.bee_algorithm import (
    initialize_bee, calculate_link_encumbrance,
    calculate_cost_function, _make_random_value_split, BestSearchHistory, IterationCostHistory
)
from structures.ObjectsDB import ObjectsDB

BeeSpecimen: TypeAlias = np.ndarray

# ==============================================================================
# SEKCJA 2: DEFINICJE TYPÓW I FUNKCJI POMOCNICZYCH
# ==============================================================================

hyperparameters_sa: dict = {
    "T_max": 1000.0,
    "T_min": 0.1,
    "cooling_rate": 0.995,
    "max_iter_per_temp": 50,
    "neighbor_intensity": 1,
    "initial_solution_strategy": 2,
    "modularity": 5
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


# ==============================================================================
# SEKCJA 3: GŁÓWNA FUNKCJA ALGORYTMU Z KOMENTARZAMI
# ==============================================================================

def train_simulated_annealing(db_context: ObjectsDB, hyperparams: dict) -> tuple[
    BeeSpecimen, int, BestSearchHistory, int, IterationCostHistory]:
    """
    Implementuje algorytm Symulowanego Wyżarzania.
    Zwraca ujednolicony format wyników, zgodny z algorytmem pszczelim.
    """
    # --- PRZYGOTOWANIE ---
    DPL_MAPPING = db_context.create_demands_to_paths_to_links_map()
    DEMAND_VALUES = db_context.get_demands_values()

    # --- KROK 1: Inicjalizacja ---
    # Stwórz losowe rozwiązanie początkowe 'i'.
    current_solution = initialize_bee(db_context, hyperparams["initial_solution_strategy"])
    # Oblicz jego koszt f(i).
    current_cost = calculate_cost_function(
        calculate_link_encumbrance(current_solution, DPL_MAPPING), hyperparams["modularity"]
    )

    # Ustaw bieżące rozwiązanie jako najlepsze dotychczas znalezione (i*).
    best_solution, best_cost = current_solution, current_cost

    # Ustaw temperaturę początkową 'T' na maksymalną wartość.
    T = hyperparams["T_max"]

    # Przygotuj struktury do zapisu historii działania algorytmu.
    best_solution_history: BestSearchHistory = [(best_solution, 0, best_cost)]
    iteration_cost_history: IterationCostHistory = [current_cost]
    iterations = 0

    # --- KROK 2: Główna pętla wyżarzania ---
    # Pętla wykonuje się, dopóki system nie "ostygnie" do minimalnej temperatury.
    while T > hyperparams["T_min"]:

        # --- KROK 3: Przeszukanie w stałej temperaturze ---
        # Dla każdego poziomu temperatury wykonaj zadaną liczbę prób.
        for _ in range(hyperparams["max_iter_per_temp"]):
            iterations += 1

            # --- KROK 4: Wygenerowanie i ocena sąsiada ---
            # Stwórz nowe rozwiązanie 'j' (sąsiada) przez lekką modyfikację obecnego rozwiązania 'i'.
            neighbor_solution = _generate_neighbor_solution(current_solution, DEMAND_VALUES,
                                                            hyperparams["neighbor_intensity"])
            # Oblicz koszt nowego rozwiązania f(j).
            neighbor_cost = calculate_cost_function(
                calculate_link_encumbrance(neighbor_solution, DPL_MAPPING), hyperparams["modularity"]
            )

            # --- KROK 5: Warunek akceptacji (Kryterium Metropolisa) ---
            # Jeśli nowe rozwiązanie jest lepsze (niższy koszt), zaakceptuj je zawsze jako nowe rozwiązanie bieżące.
            # Jeśli jest gorsze, zaakceptuj je z prawdopodobieństwem P = exp(-delta_f / T).
            # Im wyższa temperatura T, tym większa szansa na akceptację gorszego rozwiązania.
            if neighbor_cost < current_cost or random.random() < math.exp(-(neighbor_cost - current_cost) / T):
                current_solution, current_cost = neighbor_solution, neighbor_cost

                # Sprawdź, czy to nowe rozwiązanie jest najlepszym globalnie (i*).
                if current_cost < best_cost:
                    best_solution, best_cost = current_solution, current_cost
                    best_solution_history.append((best_solution, iterations, best_cost))
                    print(f"Iter {iterations}: New best cost found: {best_cost}")

            # Zapisz koszt bieżącego rozwiązania do historii (na potrzeby wykresu dynamiki).
            iteration_cost_history.append(current_cost)

        # --- KROK 6: Chłodzenie ---
        # Po wykonaniu prób dla danej temperatury, lekko "schłódź" system.
        T *= hyperparams["cooling_rate"]

    # --- KROK 7: Zakończenie ---
    # Zwróć najlepsze znalezione rozwiązanie i historię działania.
    return best_solution, best_cost, best_solution_history, iterations, iteration_cost_history
