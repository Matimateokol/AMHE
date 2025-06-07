from collections import defaultdict
from structures.AdmisiblePaths import AdmisiblePaths


class ObjectsDB:

    def __init__(self):
        self.demands = {}
        self.links = {}
        self.nodes = {}
        self.admisible_paths : AdmisiblePaths | None = None

    def persist_object(self, object, is_debug_mode=False):
        if type(object).__name__ == "Demand":
            self.demands[object.demand_id] = object
            if is_debug_mode: print("Persisted object of type 'Demand'")
        elif type(object).__name__ == "Link":
            self.links[object.link_id] = object
            if is_debug_mode: print("Persisted object of type 'Link'")
        elif type(object).__name__ == "Node":
            self.nodes[object.node_id] = object
            if is_debug_mode: print("Persisted object of type 'Node'")
        elif type(object).__name__ == "AdmisiblePaths":
            self.admisible_paths = object
            if is_debug_mode: print("Persisted object of type 'AdmisiblePaths'")
        else:
            if is_debug_mode: print("Persisted object of some different type")

    """
        Get demands dictionary, where the keys are the demand_ids and the values are the Demand objects.
    """
    def get_demands(self) -> dict:
        return self.demands

    def get_demand_ids(self) -> list[str]:
        return [demand_id for demand_id in self.demands.keys()]

    def get_demands_values(self) -> list[float]:
        return [demand.get_demand_value() for demand in self.demands.values()]
    
    def get_links(self) -> dict:
        return self.links
    
    def get_nodes(self) -> dict:
        return self.nodes
    
    def get_admisible_paths(self) -> AdmisiblePaths | None:
        return self.admisible_paths

    def create_demands_to_paths_to_links_map(self) -> dict[str, dict[str, list[str]]]:
        data = defaultdict(lambda: defaultdict(list))
        for dmd_id, _adm_paths in self.get_admisible_paths().get_paths().items():
            for _path in _adm_paths:
                data[dmd_id][_path.get_path_id()].extend(_path.get_link_ids())

        demands_to_paths_to_links_map = {k: dict(v) for k, v in data.items()}
        return demands_to_paths_to_links_map