import csv
import json
import datetime
from heapq import heappush, heappop
import heapq
import time


current_datetime = datetime.datetime.now()

NEARBY_TIME_THRESHOLD = 15 #  ANYTHING OVER THE DRIVER DOES NOT GO ( IN MINUTES )
SUFFICIENT_TRAVEL_TIME_THRESHOLD = 5 # ANY under 5 MIN WAIT TIME IS GOOD

class KDNode:
    def __init__(self, point, node_id, depth=0):
        self.point = point
        self.node_id = node_id
        self.depth = depth
        self.left = None
        self.right = None

class KDTree:
    def __init__(self, points):
        self.root = self._build_kd_tree(points)

    def _build_kd_tree(self, points, depth=0, max_depth=10):
        if not points or depth > max_depth:
            return None

        k = 2  # Dimension of space
        axis = depth % k
        points.sort(key=lambda x: x[1][axis])
        median = len(points) // 2

        node = KDNode(points[median][1], points[median][0], depth)
        node.left = self._build_kd_tree(points[:median], depth + 1, max_depth)
        node.right = self._build_kd_tree(points[median + 1:], depth + 1, max_depth)

        return node
    def find_nearest(self, point, best=None, visited=None, depth=0, max_depth=10):
        return self._find_nearest_recursive(self.root, point, best, visited, depth, max_depth)

    def _find_nearest_recursive(self, node, point, best, visited, depth, max_depth):
        if node is None or depth > max_depth:
            return best

        if visited is None:
            visited = set()

        if node.node_id in visited:
            return best

        visited.add(node.node_id)

        if best is None or euclidean_distance(point, node.point) < euclidean_distance(point, best.point):
            best = node

        axis = depth % 2
        next_branch = None

        if point[axis] < node.point[axis]:
            next_branch = node.left
            other_branch = node.right
        else:
            next_branch = node.right
            other_branch = node.left

        best = self._find_nearest_recursive(next_branch, point, best, visited, depth + 1, max_depth)

        if other_branch is not None and abs(point[axis] - node.point[axis]) < euclidean_distance(point, best.point):
            best = self._find_nearest_recursive(other_branch, point, best, visited, depth + 1, max_depth)

        return best
def build_kd_tree(node_data):
    points = [(node_id, (data['lat'], data['lon'])) for node_id, data in node_data.items()]
    return KDTree(points)

def closest_node_kdtree(lat, lon, kd_tree):
    nearest_node = kd_tree.find_nearest((lat, lon))
    return nearest_node.node_id if nearest_node else None


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


import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers

    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1) * math.cos(lat2) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance  # distance in kilometers
def find_lat_range(sorted_nodes, target_lat):
    low, high = 0, len(sorted_nodes) - 1
    while low <= high:
        mid = (low + high) // 2
        mid_lat = sorted_nodes[mid][1][0]

        if mid_lat < target_lat:
            low = mid + 1
        elif mid_lat > target_lat:
            high = mid - 1
        else:
            start = end = mid
            while start > 0 and sorted_nodes[start - 1][1][0] == mid_lat:
                start -= 1
            while end < len(sorted_nodes) - 1 and sorted_nodes[end + 1][1][0] == mid_lat:
                end += 1
            return start, end

    return max(0, low - 1), min(len(sorted_nodes) - 1, high + 1)

def find_nearest_node(sorted_nodes, lat, lon):
    start, end = find_lat_range(sorted_nodes, lat)

    nearest_node, min_dist = None, float('inf')
    for i in range(start, end + 1):
        node_id, (node_lat, node_lon) = sorted_nodes[i]
        dist = haversine_distance(lat, lon, node_lat, node_lon)
        if dist < min_dist:
            nearest_node, min_dist = node_id, dist

    return nearest_node

def build_sorted_node_list(node_data):
    nodes = [(id, (data['lat'], data['lon'])) for id, data in node_data.items()]
    return sorted(nodes, key=lambda x: (x[1][0], x[1][1]))


def precompute_travel_times(edges):
    travel_times = {}
    for edge in edges:
        start_id, end_id = edge['start_id'], edge['end_id']
        length = float(edge['length'])
        travel_times[(start_id, end_id)] = {}
        
        for hour in range(24):
            for day_type in ['weekday', 'weekend']:
                speed = float(edge[f'{day_type}_{hour}'])
                travel_times[(start_id, end_id)][f'{day_type}_{hour}'] = 60 * length / speed
    return travel_times


def euclidean_heuristic(node1, node2, node_data):
    lat1, lon1 = node_data[node1]['lat'], node_data[node1]['lon']
    lat2, lon2 = node_data[node2]['lat'], node_data[node2]['lon']
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5


def a_star_path_length(graph, start_node, end_node, heuristic):
    priority_queue = [(heuristic(start_node, end_node), 0, start_node)]

    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0

    visited = set()

    while priority_queue:
        estimated_total_cost, current_d, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == end_node:
            return distances[end_node]

        for neighbor, weight in graph[current_node].items():
            distance = current_d + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                estimated_cost = distance + heuristic(neighbor, end_node)
                heapq.heappush(priority_queue, (estimated_cost, distance, neighbor))

    return float('inf')




def construct_optimized_graph(travel_times, current_datetime):
    graph = {}
    weekday = 0 <= current_datetime.weekday() <= 4  
    hour = current_datetime.hour
    day_type = 'weekday' if weekday else 'weekend'

    for (start_id, end_id), times in travel_times.items():
        travel_time = times[f'{day_type}_{hour}']
        if start_id not in graph:
            graph[start_id] = {}
        graph[start_id][end_id] = travel_time
    return graph




def pair_drivers_passengers(drivers, passengers, graph, node_data):
    kd_tree = build_kd_tree(node_data)
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
        driver_location_idx = closest_node_kdtree(next_driver['Source Lat'], next_driver['Source Lon'], kd_tree)

        eligible_passengers = []
        for passenger in sorted_passengers:
            if parse_datetime(passenger['Date/Time']) <= driver_start_time:
                passenger_location_idx = closest_node_kdtree(passenger['Source Lat'], passenger['Source Lon'], kd_tree)
                heuristic_cost = lambda x, y: euclidean_heuristic(x, y, node_data)
                travel_time_to_pickup = a_star_path_length(graph, driver_location_idx, passenger_location_idx, heuristic_cost)
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
            dropoff_location_idx = closest_node_kdtree(best_passenger['Dest Lat'], best_passenger['Dest Lon'], kd_tree)
            heuristic_cost = lambda x, y: euclidean_heuristic(x, y, node_data)
            travel_time_with_passenger = a_star_path_length(graph, passenger_location_idx, dropoff_location_idx, heuristic_cost)
            
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
    travel_times = precompute_travel_times(edges)
    sorted_nodes = build_sorted_node_list(node_data)


    graph = construct_optimized_graph(travel_times, current_datetime)
    pairs, D1, D2 = pair_drivers_passengers(drivers, passengers, graph, node_data)

    print(f"D1 (Total Passenger Waiting and Travel Time): {D1} minutes")
    print(f"D2 (Total Driver Ride Profit): {D2} minutes")
    end_time = time.time()  

    elapsed_time = end_time - start_time
    print(f"Empirical Runtime: {elapsed_time} seconds")
