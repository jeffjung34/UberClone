import csv
from datetime import datetime
from collections import deque
import math
import time  

def convert_to_datetime(s):
    return datetime.strptime(s, '%m/%d/%Y %H:%M:%S')

def read_and_sort_csv(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        data = sorted(reader, key=lambda row: convert_to_datetime(row['Date/Time']))
        return data

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3959  # Radius of the Earth in miles
    lat_dif = math.radians(lat2 - lat1)
    lon_dif = math.radians(lon2 - lon1)
    a = math.sin(lat_dif/2) * math.sin(lat_dif/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(lon_dif/2) * math.sin(lon_dif/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

start_time = time.time()  # Record the start time
drivers = read_and_sort_csv('drivers75.csv')
passengers = read_and_sort_csv('passengers75.csv')
end_time = time.time()  # Record the end time

elapsed_time = end_time - start_time
print(f"Empirical Runtime: {elapsed_time} seconds")

drivers_q = deque(drivers)
passengers_q = deque(passengers)

total_wait_time = 0
total_drive_time = 0
average_speed = 30  

while drivers_q and passengers_q:
    driver = drivers_q.popleft()
    passenger = passengers_q.popleft()

    driver_time = convert_to_datetime(driver['Date/Time'])
    passenger_time = convert_to_datetime(passenger['Date/Time'])

    wait_time = (driver_time - passenger_time).total_seconds() / 60
    total_wait_time += max(0, wait_time)

    distance = haversine_distance(float(passenger['Source Lat']), float(passenger['Source Lon']), float(passenger['Dest Lat']), float(passenger['Dest Lon']))
    drive_time = distance / average_speed * 60  
    total_drive_time += drive_time

average_wait_time = total_wait_time / len(passengers)
average_drive_time = total_drive_time / len(drivers)
print(f"Average Passenger Wait Time (D1): {average_wait_time} minutes")
print(f"Average Driver Drive Time (D2): {average_drive_time} minutes")
