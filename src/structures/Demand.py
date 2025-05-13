# <demand_id> ( <source> <target> ) <routing_unit> <demand_value> <max_path_length>
class Demand:
    def __init__(self, demand_id: str, source: str, target: str, routing_unit: int, demand_value: float, max_path_length: str = "UNLIMITED"):
        self.demand_id = demand_id
        self.source = source
        self.target = target
        self.routing_unit = routing_unit
        self.demand_value = demand_value
        self.max_path_length = max_path_length

    def __str__(self):
        return f"{self.demand_id} {( self.source, self.target )} {self.routing_unit} {self.demand_value} {self.max_path_length}"
    
    def _get_demand_id(self):
        return self.demand_id
    
    def _set_demand_id(self, demand_id: str):
        self.demand_id = demand_id

    def _get_connection(self):
        return (self.source, self.target)
    
    def _get_source(self):
        return self.source
    
    def _set_source(self, src: str):
        self.source = src

    def _get_target(self):
        return self.target
    
    def _set_target(self, dst: str):
        self.target = dst
    
    def _get_routing_unit(self):
        return self.routing_unit
    
    def _set_routing_unit(self, routing_unit: int):
        self.routing_unit = routing_unit

    def _get_demand_value(self):
        return self.demand_value
    
    def _set_demand_value(self, demand_value: float):
        self.demand_value = demand_value

    def _get_max_path_length(self):
        return self.max_path_length
    
    def _set_max_path_length(self, max_path_length: str):
        self.max_path_length = max_path_length
