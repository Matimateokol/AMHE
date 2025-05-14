from structures.AdmisiblePaths import AdmisiblePaths, AdmisiblePath

def main():
    list_of_demands = ["Demand_0_1", "Demand_0_2", "Demand_0_3"]
    adm_paths = AdmisiblePaths(list_of_demands)

    print(adm_paths.paths)

    adm_paths.add_path_for_demand("Demand_0_1", AdmisiblePath("P_0", ["Link_0_2", "Link_1_2"]))
    adm_paths.add_path_for_demand("Demand_0_1", AdmisiblePath("P_1", ["Link_0_10", "Link_1_10"]))

    print(adm_paths.paths)
    print(adm_paths)

if __name__ == "__main__":
    main()