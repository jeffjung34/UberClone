import csv
import json
import datetime
from heapq import heappush, heappop
import heapq
import time


current_datetime = datetime.datetime.now()

NEARBY_TIME_THRESHOLD = 15 #  ANYTHING OVER THE DRIVER DOES NOT GO ( IN MINUTES )
SUFFICIENT_TRAVEL_TIME_THRESHOLD = 5 # ANY under 5 MIN WAIT TIME IS GOOD


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

def coordinates_to_geohash(coords, precision=9):
  
    latitude = coords[0]
    longitude = coords[1]
    BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"

    lat_range = (-90.0, 90.0)
    lon_range = (-180.0, 180.0)

    geohash_result = 0

    for i in range(precision * 5):
        if i % 2 == 0:
            mid = (lon_range[0] + lon_range[1]) / 2.0
            geohash_result = (geohash_result << 1) + (longitude > mid)
            lon_range = (lon_range[0], mid) if longitude <= mid else (mid, lon_range[1])
        else:
            mid = (lat_range[0] + lat_range[1]) / 2.0
            geohash_result = (geohash_result << 1) + (latitude > mid)
            lat_range = (lat_range[0], mid) if latitude <= mid else (mid, lat_range[1])

    geohash_str = ''.join(BASE32[(geohash_result >> (4 * (precision - i - 1))) & 0x1F] for i in range(precision))

    return geohash_str

def similar_geo(geohash1, geohash2):
    min_length = min(len(geohash1), len(geohash2))

    previous_percent = 0
    for i in range(min_length):
        if geohash1[i] == geohash2[i]:
            previous_percent += 1
        else:
            break

    find_similar = (previous_percent / min_length) * 100

    return find_similar

def build_geohash_index(node_data, precision=9):
    geohash_index = {}
    for node_id, coords in node_data.items():
        ghash = coordinates_to_geohash((coords['lat'], coords['lon']), precision=precision)
        if ghash not in geohash_index:
            geohash_index[ghash] = []
        geohash_index[ghash].append(node_id)
    return geohash_index


def find_nearby_nodes(lat, lon, geohash_index, precision=9, similarity_threshold=80):
    ghash = coordinates_to_geohash((lat, lon), precision=precision)
    nearby_nodes = set()

    for hash_key, nodes in geohash_index.items():
        if similar_geo(ghash, hash_key) >= similarity_threshold:
            nearby_nodes.update(nodes)

    return list(nearby_nodes)


def compute_path_length(graph, start_node, end_node):
    if start_node is None or end_node is None or start_node not in graph or end_node not in graph:
        return float('inf')

    priority_queue = [(0, start_node)]
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_node == end_node:
            return distances[end_node]

        for neighbor, weight in graph.get(current_node, {}).items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return float('inf') if distances[end_node] == float('inf') else distances[end_node]



def construct_optimized_graph(edges, current_datetime):
    graph = {}
    weekday = 0 <= current_datetime.weekday() <= 4  
    hour = current_datetime.hour

    for edge in edges:
        start_id = str(edge['start_id'])
        end_id = str(edge['end_id'])
        length = float(edge['length'])

        speed_column = f"{'weekday' if weekday else 'weekend'}_{hour}"
        avg_mph = float(edge[speed_column])

        travel_time = 60 * length / avg_mph

        if start_id not in graph:
            graph[start_id] = {}
        graph[start_id][end_id] = travel_time
    return graph

def closest_node_geohash(lat, lon, node_data, geohash_index, precision=9, similarity_threshold=80):
    nearby_nodes = find_nearby_nodes(lat, lon, geohash_index, precision, similarity_threshold)
    
    if nearby_nodes:
        return nearby_nodes[0]  # assume first as closest
    

    # fallback strategy 
    return find_closest_by_distance(lat, lon, node_data)

def find_closest_by_distance(lat, lon, node_data):
    closest_node = None
    min_distance = float('inf')

    for node_id, coords in node_data.items():
        distance = euclidean_distance((lat, lon), (coords['lat'], coords['lon']))
        if distance < min_distance:
            closest_node = node_id
            min_distance = distance

    return closest_node





def pair_drivers_passengers(drivers, passengers, graph, node_data, precision=9, similarity_threshold=80):
    # Build geohash index for all nodes
    geohash_index = build_geohash_index(node_data, precision)
    

    driver_queue = [HeapElement(parse_datetime(driver['Date/Time']), driver['ID'], driver) for driver in drivers]
    heapq.heapify(driver_queue)

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
        driver_location_idx = closest_node_geohash(next_driver['Source Lat'], next_driver['Source Lon'], node_data, geohash_index, precision, similarity_threshold)
        print(f"Processing driver {next_driver['ID']}, remaining passengers: {len(sorted_passengers)}")

        eligible_passengers = []
        for passenger in sorted_passengers:
            if parse_datetime(passenger['Date/Time']) <= driver_start_time:
                passenger_location_idx = closest_node_geohash(passenger['Source Lat'], passenger['Source Lon'], node_data, geohash_index, precision, similarity_threshold)
                travel_time_to_pickup = compute_path_length(graph, driver_location_idx, passenger_location_idx)
                if travel_time_to_pickup <= NEARBY_TIME_THRESHOLD:
                    eligible_passengers.append((passenger, travel_time_to_pickup))

        best_passenger, min_travel_time_to_pickup = None, float('inf')
        for passenger, travel_time in eligible_passengers:
            if travel_time < min_travel_time_to_pickup:
                best_passenger, min_travel_time_to_pickup = passenger, travel_time

        if best_passenger:
            sorted_passengers.remove(best_passenger)
            dropoff_location_idx = closest_node_geohash(best_passenger['Dest Lat'], best_passenger['Dest Lon'], node_data, geohash_index, precision, similarity_threshold)
            travel_time_with_passenger = compute_path_length(graph, passenger_location_idx, dropoff_location_idx)
            waiting_time = max(0, (parse_datetime(best_passenger['Date/Time']) - driver_start_time).total_seconds() / 60)
            total_waiting_time += waiting_time
            total_driver_with_passenger_time += travel_time_with_passenger
            total_driver_to_pickup_time += min_travel_time_to_pickup

            next_driver['Last Location'] = (best_passenger['Dest Lat'], best_passenger['Dest Lon'])
            next_driver['Date/Time'] += datetime.timedelta(minutes=travel_time_with_passenger)
            updated_driver_element = HeapElement(parse_datetime(next_driver['Date/Time']), next_driver['ID'], next_driver)
            heapq.heappush(driver_queue, updated_driver_element)

            pairs.append((next_driver, best_passenger))
            passengers_processed += 1

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

    edges_ids = set()
    for edge in edges:
        edges_ids.add(edge['start_id'])
        edges_ids.add(edge['end_id'])

    node_data_ids = set(node_data.keys())



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
