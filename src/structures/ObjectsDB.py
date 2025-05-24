class ObjectsDB:

    def __init__(self):
        self.demands = {}
        self.links = {}
        self.nodes = {}
        self.admisible_paths = {}

    def persist_object(self, object):
        if type(object).__name__ == "Demand":
            self.demands[object.demand_id] = object
            print("Persisted object of type 'Demand'")
        elif type(object).__name__ == "Link":
            self.links[object.link_id] = object
            print("Persisted object of type 'Link'")
        elif type(object).__name__ == "Node":
            self.nodes[object.node_id] = object
            print("Persisted object of type 'Node'")
        elif type(object).__name__ == "AdmisiblePath":
            self.admisible_paths[object.path_id] = object
            print("Persisted object of type 'AdmisiblePath'")
        else:
            print("Persisted object of some different type")
        
    
