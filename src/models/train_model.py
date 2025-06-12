# Plik: train_model.py
import os
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib import pyplot as plt
from argparse import ArgumentParser

# Upewnij się, że poniższe importy i ścieżki są zgodne ze strukturą Twojego projektu.
from data.DataParser import parse_data
from models.bee_algorithm import train_model as train_bee, hyperparameters_dict as params_bee, BestSearchHistory, \
    IterationCostHistory
from models.simulated_annealing_algorithm import train_simulated_annealing as train_sa, hyperparameters_sa as params_sa


def plot_convergence_curve(history: BestSearchHistory, file_path: str, dataset_name: str, alg_name: str, params: dict):
    """Rysuje wykres zbieżności (tylko punkty poprawy)."""
    if not history: return
    plt.figure(figsize=(12, 7))
    total_iterations = history[-1][1] if history else 0
    plt.plot([ele[1] for ele in history], [ele[2] for ele in history], marker='o', linestyle='-',
             label='Najlepszy koszt')
    plt.xlabel("Iteracje z poprawą rozwiązania");
    plt.ylabel("Koszt")
    plt.title(f"Zbieżność ({alg_name}, M:{params['modularity']}, S:{params['dist_strategy']}) - Zestaw: {dataset_name}")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5);
    plt.legend();
    plt.tight_layout()
    if len(history) > 15:
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
        plt.xticks(rotation=45)
    plt.savefig(file_path)
    print(f"Wykres zbieżności zapisano w: {file_path}")
    plt.close()


def plot_full_history(history: BestSearchHistory, total_iterations: int, file_path: str, dataset_name: str,
                      alg_name: str, params: dict):
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
    plt.xlabel("Całkowita liczba iteracji");
    plt.ylabel("Najlepszy znaleziony koszt (globalnie)")
    plt.title(
        f"Historia kosztu ({alg_name}, M:{params['modularity']}, S:{params['dist_strategy']}) - Zestaw: {dataset_name}")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5);
    plt.legend();
    plt.tight_layout()
    if total_iterations > 15:
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
        plt.xticks(rotation=45)
    plt.savefig(file_path)
    print(f"Wykres pełnej historii zapisano w: {file_path}")
    plt.close()


def plot_iteration_costs(costs: IterationCostHistory, file_path: str, dataset_name: str, alg_name: str, params: dict):
    """Rysuje wykres kosztu rozwiązania w każdej iteracji."""
    if not costs: return
    plt.figure(figsize=(12, 7))
    plt.plot(range(len(costs)), costs, linestyle='-', label='Koszt w iteracji', color='red', alpha=0.7)
    plt.xlabel("Całkowita liczba iteracji");
    plt.ylabel("Koszt rozwiązania w iteracji")
    plt.title(
        f"Koszt w każdej iteracji ({alg_name}, M:{params['modularity']}, S:{params['dist_strategy']}) - Zestaw: {dataset_name}")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5);
    plt.legend();
    plt.tight_layout()
    if len(costs) > 15:
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
        plt.xticks(rotation=45)
    plt.savefig(file_path)
    print(f"Wykres kosztu per iteracja zapisano w: {file_path}")
    plt.close()


def run_single_experiment(dataset_path: str, dataset_name: str, output_dir: str, algorithm_name: str, algorithm_func,
                          algorithm_params):
    """Przeprowadza pełen cykl eksperymentu."""
    print("-" * 60)
    print(
        f"Rozpoczynanie eksperymentu dla: [Zestaw: {dataset_name}, Algorytm: {algorithm_name}, Modularność: {algorithm_params['modularity']}, Strategia: {algorithm_params.get('dist_strategy') or algorithm_params.get('initial_solution_strategy')}]")
    database = parse_data(dataset_path)
    print("Dane wczytane pomyślnie.")
    print(f"\nRozpoczynanie treningu algorytmu: {algorithm_name}...")

    _, best_cost, learning_history, total_iterations, iteration_costs = algorithm_func(database, algorithm_params)

    print(f"Trening zakończony. Najlepszy znaleziony koszt: {best_cost}")
    if not learning_history:
        print("Nie znaleziono żadnych rozwiązań.");
        return

    os.makedirs(output_dir, exist_ok=True)

    modularity_val = algorithm_params['modularity']
    strategy_val = algorithm_params.get('dist_strategy') or algorithm_params.get('initial_solution_strategy')
    base_filename = f"{dataset_name.lower()}_{algorithm_name.lower()}_M{modularity_val}_S{strategy_val}"

    plot_convergence_curve(learning_history, os.path.join(output_dir, f"{base_filename}_convergence.png"),
                           dataset_name, algorithm_name, algorithm_params)
    plot_full_history(learning_history, total_iterations,
                      os.path.join(output_dir, f"{base_filename}_full_history.png"), dataset_name, algorithm_name,
                      algorithm_params)
    plot_iteration_costs(iteration_costs, os.path.join(output_dir, f"{base_filename}_iteration_cost.png"),
                         dataset_name, algorithm_name, algorithm_params)

    print("Zakończono przetwarzanie.")
    print("-" * 60)


def main():
    """Główna funkcja parsująca argumenty i uruchamiająca eksperyment."""
    parser = ArgumentParser(description="Uruchom eksperyment optymalizacji sieci.")
    parser.add_argument("-a", "--algorithm", type=str, choices=['BEE', 'SA'], default='BEE',
                        help="Wybierz algorytm: BEE lub SA.")
    parser.add_argument("-d", "--dataset", type=str, choices=['PL', 'GE', 'US'], default='PL',
                        help="Wybierz zbiór danych: PL, GE, lub US.")
    parser.add_argument("-mod", "--modularity", type=int, default=5,
                        help="Wartość modularności (np. 1, 5, 10). Domyślnie: 5.")
    # NOWY ARGUMENT: Strategia dystrybucji
    parser.add_argument("-s", "--strategy", type=int, default=1,
                        help="Strategia dystrybucji (np. 1, 3). Domyślnie: 1.")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__) if __file__ else '.'
    datasets_map = {
        'PL': ('polska', os.path.join(base_dir, '../../data/raw/polska.xml')),
        'GE': ('germany50', os.path.join(base_dir, '../../data/processed/germany50_with_paths.xml')),
        'US': ('janos-us-ca', os.path.join(base_dir, '../../data/processed/janos-us-ca_with_paths.xml'))
    }
    algorithms_map = {'BEE': (train_bee, params_bee.copy()), 'SA': (train_sa, params_sa.copy())}

    dataset_name, dataset_path = datasets_map[args.dataset]
    algorithm_name = args.algorithm.upper()
    selected_func, selected_params = algorithms_map[algorithm_name]


    selected_params['modularity'] = args.modularity
    selected_params['dist_strategy'] = args.strategy

    output_directory = os.path.join(base_dir, '../../reports/figures')

    run_single_experiment(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        output_dir=output_directory,
        algorithm_name=algorithm_name,
        algorithm_func=selected_func,
        algorithm_params=selected_params
    )
    print("\nEksperyment został zakończony.")


if __name__ == "__main__":
    main()
