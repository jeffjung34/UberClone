import csv
import json
import datetime
from heapq import heappush, heappop
import heapq
import time


current_datetime = datetime.datetime.now()

NEARBY_TIME_THRESHOLD = 15 #  ANYTHING OVER THE DRIVER DOES NOT GO ( IN MINUTES )
SUFFICIENT_TRAVEL_TIME_THRESHOLD = 5 # ANY under 5 MIN WAIT TIME IS GOOD

class BSTNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key, value):
        if not self.root:
            self.root = BSTNode(key, value)
        else:
            self._insert_recursive(self.root, key, value)

    def _insert_recursive(self, node, key, value):
        if key < node.key:
            if node.left is None:
                node.left = BSTNode(key, value)
            else:
                self._insert_recursive(node.left, key, value)
        else:
            if node.right is None:
                node.right = BSTNode(key, value)
            else:
                self._insert_recursive(node.right, key, value)

    def find_closest(self, key):
        return self._find_closest_recursive(self.root, key, float('inf'), None)

    def _find_closest_recursive(self, node, key, min_distance, closest_node):
        if node is None:
            return closest_node

        distance = euclidean_distance(key, node.key)
        if distance < min_distance:
            min_distance = distance
            closest_node = node

        if key < node.key:
            return self._find_closest_recursive(node.left, key, min_distance, closest_node)
        else:
            return self._find_closest_recursive(node.right, key, min_distance, closest_node)
def build_binary_search_tree(node_data):
    bst = BinarySearchTree()
    for node_id, coords in node_data.items():
        key = (coords['lat'], coords['lon'])
        bst.insert(key, node_id)
    return bst

class HeapElement:
    def __init__(self, datetime, id, data):
        self.datetime = datetime
        self.id = id
        self.data = data

    def __lt__(self, other):
        return self.datetime < other.datetime


def read_csv(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def read_json(file_path):
    with open(file_path) as jsonfile:
        return json.load(jsonfile)

def parse_datetime(datetime_str):
    if isinstance(datetime_str, str):
        return datetime.datetime.strptime(datetime_str, '%m/%d/%Y %H:%M:%S')
    return datetime_str


def euclidean_distance(coord1, coord2):
    return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5

def compute_path_length(graph, start_node, end_node):
    priority_queue = [(0, start_node)]

    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node == end_node:
            return distances[end_node]

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return float('inf')


def construct_optimized_graph(edges, current_datetime):
    graph = {}
    weekday = 0 <= current_datetime.weekday() <= 4  
    hour = current_datetime.hour

    for edge in edges:
        start_id = edge['start_id']
        end_id = edge['end_id']
        length = float(edge['length'])

        speed_column = f"{'weekday' if weekday else 'weekend'}_{hour}"
        avg_mph = float(edge[speed_column])

        travel_time = 60 * length / avg_mph

        if start_id not in graph:
            graph[start_id] = {}
        graph[start_id][end_id] = travel_time
    return graph

def closest_node_bst(lat, lon, bst):
    closest_node = bst.find_closest((lat, lon))
    return closest_node.value if closest_node else None



def pair_drivers_passengers(drivers, passengers, graph, node_data):
    bst = build_binary_search_tree(node_data)
    driver_queue = [HeapElement(parse_datetime(driver['Date/Time']), driver['ID'], driver) for driver in drivers]
    heapq.heapify(driver_queue)

    # Sort the passengers based on their waiting time
    sorted_passengers = sorted(passengers, key=lambda p: parse_datetime(p['Date/Time']))

    pairs = []
    passengers_processed = 0
    total_waiting_time = 0
    total_driver_with_passenger_time = 0
    total_driver_to_pickup_time = 0

    while driver_queue and sorted_passengers:
        next_driver_element = heapq.heappop(driver_queue)
        next_driver = next_driver_element.data
        driver_start_time = parse_datetime(next_driver['Date/Time'])
        driver_location_idx = closest_node_bst(next_driver['Source Lat'], next_driver['Source Lon'], bst)

        eligible_passengers = []
        for passenger in sorted_passengers:
            if parse_datetime(passenger['Date/Time']) <= driver_start_time:
                passenger_location_idx = closest_node_bst(passenger['Source Lat'], passenger['Source Lon'], bst)
                travel_time_to_pickup = compute_path_length(graph, driver_location_idx, passenger_location_idx)
                if travel_time_to_pickup <= NEARBY_TIME_THRESHOLD:
                    eligible_passengers.append((passenger, travel_time_to_pickup))

        # find BEST among eligible candidates
        best_passenger, min_travel_time_to_pickup = None, float('inf')
        for passenger, travel_time in eligible_passengers:
            if travel_time <= SUFFICIENT_TRAVEL_TIME_THRESHOLD:
                best_passenger = passenger
                min_travel_time_to_pickup = travel_time
                break  # Found a good enough dude, no need to continue searching
            elif travel_time < min_travel_time_to_pickup:
                best_passenger, min_travel_time_to_pickup = passenger, travel_time
        if best_passenger:
            sorted_passengers.remove(best_passenger)
            dropoff_location_idx = closest_node_bst(best_passenger['Dest Lat'], best_passenger['Dest Lon'], bst)
            travel_time_with_passenger = compute_path_length(graph, passenger_location_idx, dropoff_location_idx)
            waiting_time = max(0, (parse_datetime(best_passenger['Date/Time']) - driver_start_time).total_seconds() / 60)
            total_waiting_time += waiting_time
            total_driver_with_passenger_time += travel_time_with_passenger
            total_driver_to_pickup_time += min_travel_time_to_pickup

            next_driver['Last Location'] = (best_passenger['Dest Lat'], best_passenger['Dest Lon'])
            next_driver['Date/Time'] += datetime.timedelta(minutes=travel_time_with_passenger)
            updated_driver_element = HeapElement(parse_datetime(next_driver['Date/Time']), generate_driver_id(next_driver), next_driver)
            heapq.heappush(driver_queue, updated_driver_element)

            pairs.append((next_driver, best_passenger))
            passengers_processed += 1
            print(f"Processed passenger {passengers_processed}: Paired with Driver at {next_driver['Date/Time']}")

    D1 = total_waiting_time + total_driver_with_passenger_time
    D2 = total_driver_with_passenger_time - total_driver_to_pickup_time

    return pairs, D1, D2

def generate_driver_id(driver):
    return f"{driver['Date/Time']}-{driver['Source Lat']}-{driver['Source Lon']}"

if __name__ == '__main__':
    start_time = time.time()  # Record the start time

    drivers = read_csv('drivers.csv')
    passengers = read_csv('passengers.csv')
    edges = read_csv('edges.csv')
    node_data = read_json('node_data.json')

    for driver in drivers:
        driver['ID'] = generate_driver_id(driver)
        driver['Date/Time'] = parse_datetime(driver['Date/Time'])
        driver['Source Lat'] = float(driver['Source Lat'])
        driver['Source Lon'] = float(driver['Source Lon'])

    for passenger in passengers:
        passenger['Date/Time'] = parse_datetime(passenger['Date/Time'])
        passenger['Source Lat'] = float(passenger['Source Lat'])
        passenger['Source Lon'] = float(passenger['Source Lon'])
        passenger['Dest Lat'] = float(passenger['Dest Lat'])
        passenger['Dest Lon'] = float(passenger['Dest Lon'])

    graph = construct_optimized_graph(edges, current_datetime)
    pairs, D1, D2 = pair_drivers_passengers(drivers, passengers, graph, node_data)

    print(f"D1 (Total Passenger Waiting and Travel Time): {D1} minutes")
    print(f"D2 (Total Driver Ride Profit): {D2} minutes")
    end_time = time.time()  

    elapsed_time = end_time - start_time
    print(f"Empirical Runtime: {elapsed_time} seconds")
