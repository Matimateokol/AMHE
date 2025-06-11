from matplotlib.ticker import MultipleLocator

from models.bee_algorithm import *
# from data import make_dataset
from data.DataParser import parse_data
from matplotlib import pyplot as plt

# random.seed(77)
DATABASE = parse_data('../../data/raw/polska.xml')
DATABASE = parse_data('../../data/processed/germany50_with_paths.xml')
DATABASE = parse_data('../../data/processed/janos-us-ca_with_paths.xml')


def train_bee_algorithm_model() -> tuple[BeeSpecimen, int, BestSearchHistory]:
    return train_model(DATABASE, hyperparameters_dict)


result = train_bee_algorithm_model()
learning_history: BestSearchHistory = result[2]

print(f"Best solution is: {result[0]}")
print(f"Best solution cost is: {result[1]}")

### Plot learning curve """
# y_axis => solution_cost
# x_axis => iterations
Y: list = [ele[2] for ele in learning_history]
X: list = list(range(len(Y)))

print(Y)
print(X)

plt.plot(X, Y)
plt.xlabel("Iterations")
plt.ylabel("Transmission devices used")
plt.title("Searching for network design with minimal transmission devices used")
# plt.gca().xaxis.set_major_locator(MultipleLocator(int(len(X) / 5)))

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if len(X) > 10:
    num_ticks = 10
    base = int(len(X) / num_ticks)
    if base > 0:
        plt.gca().xaxis.set_major_locator(MultipleLocator(base=base))

# Zapisz wykres
output_path = "./../../reports/figures/bee_algorithm_learning_curve.png"
plt.savefig(output_path)
print(f"\nWykres zapisano w: {output_path}")

