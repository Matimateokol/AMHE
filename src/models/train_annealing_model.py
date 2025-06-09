import random
from matplotlib.ticker import MultipleLocator
from matplotlib import pyplot as plt

# Importy specyficzne dla Symulowanego Wyżarzania
from models.simulated_annealing_algorithm import train_simulated_annealing, hyperparameters_sa
from data.DataParser import parse_data
from structures.ObjectsDB import ObjectsDB

# Ustawienie ziarna losowości dla powtarzalności wyników
# random.seed(77)


# Wczytanie danych z pliku - tak samo jak dla algorytmu pszczelego
DATABASE = parse_data('data/raw/polska.xml')

def train_sa_model() -> tuple:
    """
    Funkcja opakowująca, która uruchamia proces treningowy
    dla algorytmu Symulowanego Wyżarzania.
    """
    print("Starting Simulated Annealing training...")
    # Wywołanie głównej funkcji z zaimportowanymi hiperparametrami
    results = train_simulated_annealing(DATABASE, hyperparameters_sa)
    print("Simulated Annealing training finished.")
    return results


# Uruchomienie algorytmu i pobranie wyników
best_solution, best_cost, learning_history = train_sa_model()

# Wyświetlenie najlepszego znalezionego wyniku
print(f"\nBest solution cost found by Simulated Annealing: {best_cost}")
# Uwaga: Wyświetlenie całej macierzy rozwiązania może być bardzo obszerne
# print(f"Best solution found:\n{best_solution}")


### Wygenerowanie i zapisanie wykresu krzywej uczenia ###

# Oś Y to historia najlepszych kosztów znalezionych w kolejnych iteracjach
Y: list[int] = learning_history
# Oś X to kolejne iteracje
X: list[int] = list(range(len(Y)))

print(f"\nGenerating learning curve plot... ({len(X)} iterations recorded)")

plt.figure(figsize=(12, 6)) # Ustawienie rozmiaru wykresu dla lepszej czytelności
plt.plot(X, Y)
plt.xlabel("Iterations")
plt.ylabel("Cost (number of transmission devices)")
plt.title("Simulated Annealing: Search for minimum-cost network design")

# Ustawienie siatki i znaczników na osiach
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.gca().xaxis.set_major_locator(MultipleLocator(base=int(len(X) / 10))) # Ustawia 10 głównych znaczników na osi X
plt.tight_layout() # Dopasowuje wykres, aby zapobiec ucinaniu etykiet

# Zapisanie wykresu do pliku w katalogu z raportami
output_path = "reports/figures/simulated_annealing_learning_curve.png"
plt.savefig(output_path)

print(f"Plot saved to: {output_path}")