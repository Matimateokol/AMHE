import sys
import os
import networkx as nx
import xml.etree.ElementTree as ET
from itertools import islice


def strip_namespace(tree_root):
    """Usuwa przestrzeń nazw (namespace) z tagów XML, aby uprościć wyszukiwanie."""
    for elem in tree_root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    return tree_root


def yens_k_shortest_paths(graph, source, target, k):
    """
    Poprawiona i bardziej stabilna implementacja algorytmu Yena
    do znajdowania k-najkrótszych ścieżek.
    """
    try:
        A = [nx.dijkstra_path(graph, source, target, weight='weight')]
        yield A[0]
    except nx.NetworkXNoPath:
        return

    B = []

    for i in range(1, k):
        # Iteruj przez każdy węzeł w ostatniej znalezionej najkrótszej ścieżce, OPRÓCZ OSTATNIEGO.
        # To jest kluczowa poprawka błędu 'IndexError'.
        for spur_node_index in range(len(A[-1]) - 1):
            spur_node = A[-1][spur_node_index]
            root_path = A[-1][:spur_node_index + 1]

            graph_copy = graph.copy()

            for path in A:
                if len(path) > spur_node_index and root_path == path[:spur_node_index + 1]:
                    u, v = path[spur_node_index], path[spur_node_index + 1]
                    if graph_copy.has_edge(u, v):
                        graph_copy.remove_edge(u, v)

            for node in root_path[:-1]:
                if node in graph_copy:
                    graph_copy.remove_node(node)

            try:
                spur_path = nx.dijkstra_path(graph_copy, spur_node, target, weight='weight')
                total_path = root_path[:-1] + spur_path
                if total_path not in B:
                    B.append(total_path)
            except nx.NetworkXNoPath:
                continue

        if not B:
            break

        B.sort(key=lambda p: sum(graph[u][v]['weight'] for u, v in zip(p[:-1], p[1:])))

        if not B:  # Dodatkowe zabezpieczenie
            break

        new_path = B.pop(0)
        A.append(new_path)
        yield new_path


def process_network_file(input_path, output_path, k):
    """
    Główna funkcja, która wczytuje plik XML, dodaje brakujące tagi <setupCost>,
    generuje k ścieżek i zapisuje nowy, uzupełniony plik.
    """
    print(f"Wczytywanie danych z: {input_path}")
    tree = ET.parse(input_path)
    root = strip_namespace(tree.getroot())

    G = nx.Graph()
    links_section = root.find(".//links")
    print("Sprawdzanie i dodawanie brakujących tagów <setupCost>...")
    for link_elem in links_section.findall("link"):
        if link_elem.find("setupCost") is None:
            setup_cost_elem = ET.Element('setupCost')
            setup_cost_elem.text = "0.0"
            target_elem = link_elem.find('target')
            if target_elem is not None:
                children = list(link_elem)
                target_index = children.index(target_elem)
                link_elem.insert(target_index + 1, setup_cost_elem)

        link_id = link_elem.get("id")
        source = link_elem.find("source").text
        target = link_elem.find("target").text
        G.add_edge(source, target, weight=1, link_id=link_id)

    print(f"Graf sieci zbudowany. Liczba węzłów: {G.number_of_nodes()}, Liczba połączeń: {G.number_of_edges()}")

    print(f"Generowanie {k} ścieżek dla każdego zapotrzebowania...")
    demands_section = root.find(".//demands")
    for demand_elem in demands_section.findall("demand"):
        demand_id = demand_elem.get("id")
        source_node = demand_elem.find("source").text
        target_node = demand_elem.find("target").text

        if source_node not in G or target_node not in G:
            print(f"OSTRZEŻENIE: Węzły dla zapotrzebowania {demand_id} nie istnieją w grafie. Pomijam.")
            continue

        paths_as_nodes = list(islice(yens_k_shortest_paths(G, source_node, target_node, k), k))

        if not paths_as_nodes:
            print(f"OSTRZEŻENIE: Nie znaleziono żadnej ścieżki dla zapotrzebowania {demand_id}.")
            continue

        paths_root_element = ET.SubElement(demand_elem, 'admissiblePaths')

        for i, node_path in enumerate(paths_as_nodes):
            link_ids_path = [G[u][v]['link_id'] for u, v in zip(node_path[:-1], node_path[1:])]
            path_element = ET.SubElement(paths_root_element, 'admissiblePath', id=f"P_{i}")
            for link_id in link_ids_path:
                link_element = ET.SubElement(path_element, 'linkId')
                link_element.text = link_id

    print("Zakończono generowanie dopuszczalnych ścieżek.")

    if sys.version_info >= (3, 9):
        ET.indent(tree, space="  ")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.write(output_path, encoding="ISO-8859-1", xml_declaration=True)
    print(f"Pomyślnie zapisano uzupełniony plik w: {output_path}")


if __name__ == "__main__":
    INPUT_FILENAME = 'germany50.xml'
    OUTPUT_FILENAME = 'germany50_with_paths.xml'

    INPUT_FILENAME = 'janos-us-ca.xml'
    OUTPUT_FILENAME = 'janos-us-ca_with_paths.xml'

    NUM_PATHS_TO_GENERATE = 7

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file_path = os.path.join(PROJECT_ROOT, 'data', 'raw', INPUT_FILENAME)
    output_file_path = os.path.join(PROJECT_ROOT, 'data', 'processed', OUTPUT_FILENAME)

    process_network_file(input_file_path, output_file_path, NUM_PATHS_TO_GENERATE)
