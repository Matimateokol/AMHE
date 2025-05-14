
class AdmisiblePath:
    def __init__(self, path_id: str, link_ids: list ):
        self.path_id = path_id
        self.link_ids = link_ids

    def __str__(self) -> str:
        return f"{self.path_id} ( {" ".join(str(link_id) for link_id in self.link_ids)} )"
    
    def __repr__(self) -> str:
        return f"{self.path_id} ( {" ".join(str(link_id) for link_id in self.link_ids)} )"

class AdmisiblePaths:
    def __init__(self, demands_list: list):
        paths = {}
        for demand_id in demands_list:
            paths[demand_id] = []
        self.paths = paths
    
    def __str__(self) -> str:
        return f"{self.paths}"
    
    def __repr__(self) -> str:
        return f"{self.paths}"
    
    def add_path_for_demand(self, demand_id: str, path: AdmisiblePath):
        self.paths[demand_id].append(path)


