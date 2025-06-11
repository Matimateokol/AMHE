import random
from matplotlib.ticker import MultipleLocator
from matplotlib import pyplot as plt

# Importy specyficzne dla Symulowanego Wyżarzania
from models.simulated_annealing_algorithm import train_simulated_annealing, hyperparameters_sa
from data.DataParser import parse_data
from structures.ObjectsDB import ObjectsDB

# Ustawienie ziarna losowości dla powtarzalności wyników
# random.seed(77)
DATABASE = parse_data('../../data/raw/polska.xml')
# DATABASE = parse_data('../../data/processed/germany50_with_paths.xml')
# DATABASE = parse_data('../../data/processed/janos-us-ca_with_paths.xml')


def train_sa_model() -> tuple:
    """
    Funkcja opakowująca, która uruchamia proces treningowy
    dla algorytmu Symulowanego Wyżarzania.
    """
    print("Starting Simulated Annealing training...")
    results = train_simulated_annealing(DATABASE, hyperparameters_sa)
    print("Simulated Annealing training finished.")
    return results

best_solution, best_cost, learning_history = train_sa_model()

# Wyświetlenie najlepszego znalezionego wyniku
print(f"Best solution found:\n{best_solution}")
print(f"\nBest solution cost found by Simulated Annealing: {best_cost}")


### Wygenerowanie i zapisanie wykresu krzywej uczenia ###
Y: list[int] = learning_history
X: list[int] = list(range(len(Y)))

print(f"\nGenerating learning curve plot... ({len(X)} iterations recorded)")

plt.figure(figsize=(12, 6))
plt.plot(X, Y)
plt.xlabel("Iterations")
plt.ylabel("Cost (number of transmission devices)")
plt.title("Simulated Annealing: Search for minimum-cost network design")

# Ustawienie siatki i znaczników na osiach
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.gca().xaxis.set_major_locator(MultipleLocator(base=int(len(X) / 10)))
plt.tight_layout()

# Zapisanie wykresu do pliku w katalogu z raportami
output_path = "../../reports/figures/simulated_annealing_poland.png"
# output_path = "../../reports/figures/simulated_annealing_germany.png"
# output_path = "../../reports/figures/simulated_annealing_janos-us-ca.png"
plt.savefig(output_path)

print(f"Plot saved to: {output_path}")