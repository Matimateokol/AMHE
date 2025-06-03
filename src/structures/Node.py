class Node:
    def __init__(self, node_id: str, longitude_latitude: tuple[float, float]):
        self.node_id = node_id
        self.longitude_latitude = longitude_latitude

    def __str__(self):
        return f"{self.node_id} ( {self.longitude_latitude[0]} {self.longitude_latitude[1]} )"
    
    def __repr__(self):
        return f"{self.node_id} ( {self.longitude_latitude[0]} {self.longitude_latitude[1]} )"
    
    def set_node_id(self, new_node_id):
        self.node_id = new_node_id

    def get_node_id(self) -> str:
        return self.node_id
    
    def set_longitude_latitude(self, longitude: float, latitude: float):
        self.longitude_latitude = (longitude, latitude)
    
    def get_longitude_latitude(self) -> tuple[float, float]:
        return self.longitude_latitude