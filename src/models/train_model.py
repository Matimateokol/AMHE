
import os
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib import pyplot as plt
from argparse import ArgumentParser

from data.DataParser import parse_data
from models.bee_algorithm import train_model as train_bee, hyperparameters_dict as params_bee, BestSearchHistory, \
    IterationCostHistory
from models.simulated_annealing_algorithm import train_simulated_annealing as train_sa, hyperparameters_sa as params_sa


def plot_convergence_curve(history: BestSearchHistory, total_iterations: int, file_path: str, dataset_name: str,
                           alg_name: str):
    """Rysuje wykres zbieżności (tylko punkty poprawy)."""
    if not history: return
    plt.figure(figsize=(12, 7))
    plt.plot([ele[1] for ele in history], [ele[2] for ele in history], marker='o', linestyle='-',
             label='Najlepszy koszt')
    plt.xlabel("Iteracje z poprawą rozwiązania")
    plt.ylabel("Koszt")
    plt.title(f"Zbieżność ({alg_name}) - Zestaw: {dataset_name}")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)
    print(f"Wykres zbieżności zapisano w: {file_path}")
    plt.close()


def plot_full_history(history: BestSearchHistory, total_iterations: int, file_path: str, dataset_name: str,
                      alg_name: str):
    """Rysuje wykres najlepszego globalnego kosztu w każdej iteracji."""
    if not history: return
    plt.figure(figsize=(12, 7))
    full_history_y = []
    history_pointer, current_best_cost = 0, history[0][2]
    for i in range(total_iterations + 1):
        if history_pointer < len(history) and i == history[history_pointer][1]:
            current_best_cost = history[history_pointer][2]
            history_pointer += 1
        full_history_y.append(current_best_cost)
    plt.plot(range(total_iterations + 1), full_history_y, linestyle='-', label='Najlepszy koszt w czasie',
             color='green')
    plt.xlabel("Całkowita liczba iteracji")
    plt.ylabel("Najlepszy znaleziony koszt (globalnie)")
    plt.title(f"Historia najlepszego kosztu w czasie ({alg_name}) - Zestaw: {dataset_name}")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)
    print(f"Wykres pełnej historii zapisano w: {file_path}")
    plt.close()


def plot_iteration_costs(costs: IterationCostHistory, file_path: str, dataset_name: str, alg_name: str):
    """Rysuje wykres kosztu rozwiązania w każdej iteracji."""
    if not costs: return
    plt.figure(figsize=(12, 7))
    plt.plot(range(len(costs)), costs, linestyle='-', label='Koszt w iteracji', color='red', alpha=0.7)
    plt.xlabel("Całkowita liczba iteracji")
    plt.ylabel("Koszt rozwiązania w iteracji")
    plt.title(f"Koszt w każdej iteracji ({alg_name}) - Zestaw: {dataset_name}")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)
    print(f"Wykres kosztu per iteracja zapisano w: {file_path}")
    plt.close()


def run_single_experiment(dataset_path: str, dataset_name: str, output_dir: str, algorithm_name: str, algorithm_func,
                          algorithm_params):
    """Przeprowadza pełen cykl eksperymentu."""
    print("-" * 60)
    print(f"Rozpoczynanie eksperymentu dla: [Zestaw: {dataset_name}, Algorytm: {algorithm_name}]")
    database = parse_data(dataset_path)
    print("Dane wczytane pomyślnie.")
    print(f"\nRozpoczynanie treningu algorytmu: {algorithm_name}...")

    _, best_cost, learning_history, total_iterations, iteration_costs = algorithm_func(database, algorithm_params)

    print(f"Trening zakończony. Najlepszy znaleziony koszt: {best_cost}")
    if not learning_history:
        print("Nie znaleziono żadnych rozwiązań.")
        return

    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"{dataset_name.lower()}_{algorithm_name.lower()}"

    plot_convergence_curve(learning_history, total_iterations,
                           os.path.join(output_dir, f"{base_filename}_1_convergence.png"), dataset_name, algorithm_name)
    plot_full_history(learning_history, total_iterations,
                      os.path.join(output_dir, f"{base_filename}_2_full_history.png"), dataset_name, algorithm_name)
    plot_iteration_costs(iteration_costs, os.path.join(output_dir, f"{base_filename}_3_iteration_cost.png"),
                         dataset_name, algorithm_name)

    print("Zakończono przetwarzanie.")
    print("-" * 60)


def main():
    """Główna funkcja parsująca argumenty i uruchamiająca eksperyment."""
    parser = ArgumentParser(description="Uruchom eksperyment optymalizacji sieci.")
    parser.add_argument("-a", "--algorithm", type=str, choices=['BEE', 'SA'], default='BEE',
                        help="Wybierz algorytm: BEE lub SA.")
    parser.add_argument("-d", "--dataset", type=str, choices=['PL', 'GE', 'US'], default='PL',
                        help="Wybierz zbiór danych: PL, GE, lub US.")
    args = parser.parse_args()

    # Zaktualizowane ścieżki, aby były bardziej uniwersalne
    base_dir = os.path.dirname(__file__)
    datasets_map = {
        'PL': ('polska', os.path.join(base_dir, '../../data/raw/polska.xml')),
        'GE': ('germany50', os.path.join(base_dir, '../../data/processed/germany50_with_paths.xml')),
        'US': ('janos-us-ca', os.path.join(base_dir, '../../data/processed/janos-us-ca_with_paths.xml'))
    }
    algorithms_map = {'BEE': (train_bee, params_bee), 'SA': (train_sa, params_sa)}

    dataset_name, dataset_path = datasets_map[args.dataset]
    algorithm_name = args.algorithm.upper()
    selected_func, selected_params = algorithms_map[algorithm_name]
    output_directory = os.path.join(base_dir, '../../reports/figures')

    run_single_experiment(
        dataset_path=dataset_path, dataset_name=dataset_name, output_dir=output_directory,
        algorithm_name=algorithm_name, algorithm_func=selected_func, algorithm_params=selected_params
    )
    print("\nEksperyment został zakończony.")


if __name__ == "__main__":
    main()