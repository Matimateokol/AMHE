class Node:
    def __init__(self, node_id: str, longitude_latitude: tuple):
        self.node_id = node_id
        self.longitude_latitude = longitude_latitude

    def __str__(self):
        return f"{self.node_id} ( {self.longitude_latitude[0]} {self.longitude_latitude[1]} )"
    
    def _set_node_id(self, new_node_id):
        self.node_id = new_node_id

    def _get_node_id(self):
        return self.node_id
    
    def _set_longitude_latitude(self, longitude: float, latitude: float):
        self.longitude_latitude = (longitude, latitude)
    
    def _get_longitude_latitude(self):
        return self.longitude_latitude