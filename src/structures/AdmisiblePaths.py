
class AdmisiblePath:
    def __init__(self, path_id: str, link_ids: list ):
        self.path_id = path_id
        self.link_ids = link_ids

class AdmisiblePaths:
    def __init__(self, demands_count: int):
        return {}
    
    def __str__(self):
        #TODO:implement to_string method
        pass
    
    def add_paths_for_demand(self, demand_id: str, path: AdmisiblePath):
        pass


