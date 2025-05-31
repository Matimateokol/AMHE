from structures.AdmisiblePaths import AdmisiblePaths, AdmisiblePath
from structures.ObjectsDB import ObjectsDB
from structures.Demand import Demand
from structures.Link import Link
from structures.Node import Node
from data import DataParser
import pandas as pd

def main():
    # list_of_demands = ["Demand_0_1", "Demand_0_2", "Demand_0_3"]
    # adm_paths = AdmisiblePaths(list_of_demands)

    # print(adm_paths.paths)

    # adm_paths.add_path_for_demand("Demand_0_1", AdmisiblePath("P_0", ["Link_0_2", "Link_1_2"]))
    # adm_paths.add_path_for_demand("Demand_0_1", AdmisiblePath("P_1", ["Link_0_10", "Link_1_10"]))

    # print(adm_paths.paths)
    # print(adm_paths)

    demand1 = Demand("1", "Warszawa", "Bytom", 2, 300.00)
    link1 = Link("1", ("Warszawa", "Bytom"), 200.00, 20, 2, 2, (2, 2, 2, 2))
    node1 = Node("1", (45.00, 48.00))
    admisible_path1 = AdmisiblePath("1", ["1", "10"])
    objects_db = ObjectsDB()

    objects_db.persist_object(link1)

    file_location = "data/raw/polska.xml"

    objects_db = DataParser.parse_data(file_location)

    # print(objects_db.demands.items())
    for (k, v) in objects_db.demands.items():
        print(f'{k}: {v.demand_value}')

if __name__ == "__main__":
    main()