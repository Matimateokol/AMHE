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
    
    def __repr__(self):
        return f"{self.demand_id} {( self.source, self.target )} {self.routing_unit} {self.demand_value} {self.max_path_length}"
    
    def get_demand_id(self) -> str:
        return self.demand_id
    
    def set_demand_id(self, demand_id: str):
        self.demand_id = demand_id

    def get_connection(self) -> tuple[str, str]:
        return (self.source, self.target)
    
    def get_source(self) -> str:
        return self.source
    
    def set_source(self, src: str):
        self.source = src

    def get_target(self) -> str:
        return self.target
    
    def set_target(self, dst: str):
        self.target = dst
    
    def get_routing_unit(self) -> int:
        return self.routing_unit
    
    def set_routing_unit(self, routing_unit: int):
        self.routing_unit = routing_unit

    def get_demand_value(self) -> float:
        return self.demand_value
    
    def set_demand_value(self, demand_value: float):
        self.demand_value = demand_value

    def get_max_path_length(self) -> str:
        return self.max_path_length
    
    def set_max_path_length(self, max_path_length: str):
        self.max_path_length = max_path_length
