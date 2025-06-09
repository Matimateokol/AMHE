import sys
import os
import time
import numpy as np

# --- Konfiguracja Eksperymentu ---
NUM_RUNS = 3  # Ile razy uruchomić algorytm (np. kilkanaście)
DATASET_PATH = 'data/raw/polska.xml'  # Ścieżka do zbioru danych
ALGORITHM = "SA"  # Wybierz algorytm: "SA" dla Symulowanego Wyżarzania lub "BEE" dla Pszczelego

# --- Przygotowanie Środowiska ---

# Dodaj katalog 'src' do ścieżki Pythona, aby znaleźć moduły
sys.path.append('src')

from data.DataParser import parse_data
from models.simulated_annealing_algorithm import train_simulated_annealing, hyperparameters_sa
from models.bee_algorithm import train_model as train_bee_algorithm, hyperparameters_dict as hyperparameters_bee

# Mapowanie wyboru algorytmu na odpowiednie funkcje i parametry
ALGORITHMS = {
    "SA": {
        "function": train_simulated_annealing,
        "params": hyperparameters_sa
    },
    "BEE": {
        "function": train_bee_algorithm,
        "params": hyperparameters_bee
    }
}

if ALGORITHM not in ALGORITHMS:
    raise ValueError(f"Nieznany algorytm: {ALGORITHM}. Wybierz z {list(ALGORITHMS.keys())}")

# --- Uruchomienie Eksperymentu ---

print("=" * 50)
print(f"Rozpoczynanie eksperymentu dla algorytmu: {ALGORITHM}")
print(f"Liczba uruchomień: {NUM_RUNS}")
print(f"Zbiór danych: {DATASET_PATH}")
print("=" * 50)

# Wczytaj dane tylko raz, aby oszczędzić czas
print("Wczytywanie danych...")
DATABASE = parse_data(DATASET_PATH)
print("Dane wczytane pomyślnie.")

cost_results = []
start_time = time.time()

for i in range(NUM_RUNS):
    print(f"\n--- Uruchomienie {i + 1}/{NUM_RUNS} ---")

    # Wybierz funkcję i parametry na podstawie konfiguracji
    selected_algorithm = ALGORITHMS[ALGORITHM]
    train_function = selected_algorithm["function"]
    hyperparams = selected_algorithm["params"]

    # Uruchom algorytm, przechwytując tylko najlepszy koszt
    _, best_cost, _ = train_function(DATABASE, hyperparams)

    print(f"Zakończono. Koszt: {best_cost}")
    cost_results.append(best_cost)

end_time = time.time()
total_duration = end_time - start_time

# --- Podsumowanie Statystyczne ---

print("\n" + "=" * 50)
print("EKSPERYMENT ZAKOŃCZONY - PODSUMOWANIE")
print("=" * 50)
print(f"Całkowity czas: {total_duration:.2f} s")
print(f"Liczba uruchomień: {NUM_RUNS}")
print("-" * 20)

# Obliczenia statystyczne za pomocą numpy
avg_cost = np.mean(cost_results)
std_dev = np.std(cost_results)
min_cost = np.min(cost_results)
max_cost = np.max(cost_results)

print(f"Najlepszy znaleziony koszt (min): {min_cost}")
print(f"Najgorszy znaleziony koszt (max): {max_cost}")
print(f"Średni koszt: {avg_cost:.2f}")
print(f"Odchylenie standardowe: {std_dev:.2f}")
print("-" * 20)
print("Wszystkie uzyskane koszty:")
print(cost_results)
print("=" * 50)