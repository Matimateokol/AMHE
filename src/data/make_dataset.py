import random

import numpy as np
import pandas as pd

from structures.ObjectsDB import ObjectsDB
from structures.AdmisiblePaths import AdmisiblePaths
from structures.Demand import Demand

from models import bee_algorithm


def generate_example_demand_to_path_distribution(db: ObjectsDB, distribution_strategy: int, as_ndarray: bool = False) -> pd.DataFrame | np.ndarray:
    """
    Creates an example initial demand values to admisible paths assignment. Behavior is based on the distribution_strategy parameter.
    Distribution strategy is also related to number of splits of the demand's value.

    :param db: ObjectsDB instance
    :param distribution_strategy: 0 for deterministic aggregation (always first path), 1 for random aggregation, 2 and more for random disaggregation
    :return: DataFrame | np.ndarray
    """
    demands: dict[str, Demand] = db.get_demands()
    adm_paths: AdmisiblePaths = db.get_admisible_paths()
    max_paths = adm_paths.get_max_paths()
    data = pd.DataFrame()

    first_demand = next(iter(demands.values()))
    path_names = [path.get_path_id() for path in adm_paths.get_paths_for_demand(first_demand.get_demand_id())]
    data['path_id'] = path_names

    for dmd_id in demands.keys():
        demand_value = int(demands[dmd_id].get_demand_value())
        demand_distribution = make_random_value_split(demand_value, max_paths, distribution_strategy)
        data[dmd_id] = demand_distribution

    if as_ndarray:

        return data.drop(columns=['path_id']).to_numpy()

    return data


def make_random_value_split(total_value: int, list_size: int, n_splits: int) -> list:
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
    """

    if total_value < 0 or list_size < 0 or n_splits < 0:
        raise ValueError("total_value, list_size, and n_splits must be non-negative")
    if n_splits > list_size:
        raise ValueError("n_splits cannot be greater than list_size")

    if list_size == 0:
        return []

    values = [0] * list_size
    if n_splits == 0:
        values[0] = total_value
        return values
    elif n_splits == 1:
        values[random.randint(0, list_size - 1)] = total_value
        return values
    else:

        split_points = sorted(random.sample(range(1, total_value), n_splits - 1))


        first_part = [split_points[0]]  # example [20]
        mid_part = [split_points[i + 1] - split_points[i] for i in
                    range(len(split_points) - 1)]
        last_part = [total_value - split_points[-1]]
        parts = (first_part + mid_part + last_part)


        values = [0] * list_size


        indices = random.sample(range(list_size), n_splits)
        for i, value in zip(indices, parts):
            values[i] = value

        return values



from data.DataParser import parse_data

objects_db = parse_data('./../../data/raw/polska.xml')
random.seed(77)


total_value = 300
list_size = 7
n_splits = 0
result = make_random_value_split(total_value, list_size, n_splits)
print(result)


bee1 = generate_example_demand_to_path_distribution(objects_db, 3, True)
print(bee1)


dmnd_ids = objects_db.get_demand_ids()
d_p_l_ids = objects_db.create_demands_to_paths_to_links_map()
path_ids = list(next(iter(d_p_l_ids.values())).keys())

print(dmnd_ids)
print(d_p_l_ids)
print(path_ids)

encumbrance = bee_algorithm.calculate_link_encumbrance(bee1,  d_p_l_ids)
print(encumbrance)

print(bee_algorithm.calculate_cost_function(encumbrance, 5))

bee_colony = bee_algorithm.create_new_bee_colony(objects_db, 5)

print(bee_colony)

for bee in bee_colony:
    encumbrance = bee_algorithm.calculate_link_encumbrance(bee, d_p_l_ids)
    print(bee_algorithm.calculate_cost_function(encumbrance, 10))