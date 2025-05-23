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
        else:
            print("Persisted object of different type")
        
    
