from structures.ObjectsDB import ObjectsDB
from structures.Demand import Demand
from structures.Link import Link
from structures.Node import Node
from structures.AdmisiblePaths import AdmisiblePaths, AdmisiblePath
import xml.etree.ElementTree as ET

def strip_namespace(tree):
    for elem in tree.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    return tree

def parse_data(file_path, src_file_format = 'xml') -> ObjectsDB:
    data_in_runtime = ObjectsDB()
    xml_tree = ET.parse(file_path)

    root = strip_namespace(xml_tree.getroot())
    
    demands = root.find('demands').findall("demand")
    links = root.find("networkStructure").find("links").findall("link")
    nodes = root.find("networkStructure").find("nodes").findall("node")

    for demand in demands:
        demand_obj = Demand(
            demand.get("id"), 
            demand.find("source").text, 
            demand.find("target").text, 
            1,
            float(demand.find("demandValue").text))
        data_in_runtime.persist_object(demand_obj)
    
    for link in links:
        capacities = [ float(x.find("capacity").text) for x in link.find("additionalModules").findall("addModule") ]
        costs = [ float(y.find("cost").text) for y in link.find("additionalModules").findall("addModule") ]
        module_capacities_costs = []
        for cap, cost in zip(capacities, costs):
            module_capacities_costs.append(cap)
            module_capacities_costs.append(cost)

        link_obj = Link(
            link.get("id"),
            (link.find("source").text, link.find("target").text),
            float(0),
            float(0),
            float(0),
            float(link.find("setupCost").text),
            tuple(module_capacities_costs)
        )
        data_in_runtime.persist_object(link_obj)

    for node in nodes:
        coordinates = [float(node.find("coordinates").find("x").text), float(node.find("coordinates").find("y").text)]
        node_obj = Node(
            node.get("id"),
            tuple(coordinates)
        )
        data_in_runtime.persist_object(node_obj)

    admisible_paths = AdmisiblePaths(data_in_runtime.demands.keys())


    for demand in demands:
        adm_paths = demand.find("admissiblePaths").findall("admissiblePath")
        
        for adm_path in adm_paths:
            path = AdmisiblePath(
                adm_path.get("id"),
                list([link.text for link in adm_path.findall("linkId")])
            )
            admisible_paths.add_path_for_demand(demand.get("id"), path)

    data_in_runtime.persist_object(admisible_paths)

    return data_in_runtime