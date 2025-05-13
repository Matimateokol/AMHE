## <link_id> ( <source> <target> ) <pre_installed_capacity> <pre_installed_capacity_cost> <routing_cost> <setup_cost> ( {<module_capacity> <module_cost>}* )

class Link:
    def __init__(
            self, link_id: str, 
            connection: tuple, 
            pre_installed_capacity: float, 
            pre_installed_capacity_cost: float, 
            routing_cost: float,
            setup_cost: float,
            module_capacity_cost: tuple
            ):
        self.link_id = link_id
        self.connection = connection
        self.pre_installed_capacity = pre_installed_capacity
        self.pre_installed_capacity_cost = pre_installed_capacity_cost
        self.routing_cost = routing_cost
        self.setup_cost = setup_cost
        self.module_capacity_cost = module_capacity_cost

    def __str__(self):
        return f"{self.link_id} {self.connection} {self.pre_installed_capacity} {self.pre_installed_capacity_cost} {self.routing_cost} {self.setup_cost} {self.module_capacity_cost}"
    
    def _get_link_id(self):
        return self.link_id
    
    def _set_link_id(self, link_id):
        self.link_id = link_id
    
    def _get_connection(self):
        return self.connection
    
    def _set_connection(self, source: str, target: str):
        self.connection = (source, target)

    def _get_pre_installed_capacity(self):
        return self.pre_installed_capacity
    
    def _set_pre_installed_capacity(self, capacity: float):
        self.pre_installed_capacity = capacity

    def _get_pre_installed_capacity_cost(self):
        return self.pre_installed_capacity_cost

    def _set_pre_installed_capacity_cost(self, capacity_cost: float):
        self.pre_installed_capacity_cost = capacity_cost

    def _get_routing_cost(self):
        return self.routing_cost
    
    def _set_routing_cost(self, routing_cost: float):
        self.routing_cost = routing_cost

    def _get_setup_cost(self):
        return self.setup_cost
    
    def _set_setup_cost(self, setup_cost: float):
        self.setup_cost = setup_cost

    def _get_module_capacity_cost(self) -> tuple:
        return self.module_capacity_cost
    
    def _set_module_capacity_cost(self, cap1: float, cap2: float, cost1: float, cost2: float):
        self.module_capacity_cost = (cap1, cap2, cost1, cost2)


    