
class AdmisiblePath:
    def __init__(self, path_id: str, link_ids: list[str] ):
        self.path_id = path_id
        self.link_ids = link_ids

    def __str__(self) -> str:
        return f"{self.path_id} ( {" ".join(str(link_id) for link_id in self.link_ids)} )"
    
    def __repr__(self) -> str:
        return f"{self.path_id} ( {" ".join(str(link_id) for link_id in self.link_ids)} )"

    def get_path_id(self) -> str:
        return self.path_id

    def get_link_ids(self) -> list[str]:
        return self.link_ids

class AdmisiblePaths:
    def __init__(self, demands_list: list):
        paths = {}
        for demand_id in demands_list:
            paths[demand_id] = []
        self.paths = paths
        self.max_paths = 0
    
    def __str__(self) -> str:
        return f"{self.paths}"
    
    def __repr__(self) -> str:
        return f"{self.paths}"
    
    def add_path_for_demand(self, demand_id: str, path: AdmisiblePath):
        self.paths[demand_id].append(path)
        if len(self.paths[demand_id]) > self.max_paths:
            self.max_paths = len(self.paths[demand_id])

    def get_paths_for_demand(self, demand_id: str) -> list:
        return self.paths[demand_id]

    def get_paths(self) -> dict:
        return self.paths

    def get_path_ids_for_demand(self, demand_id: str) -> list[str]:
        return [path.get_path_id() for path in self.paths[demand_id]]

    def get_max_paths(self) -> int:
        return self.max_paths
