import pandas as pd
from queue import Queue
from datetime import datetime
import time
def parse_datetime(dt_str):
    return datetime.strptime(dt_str, '%m/%d/%Y %H:%M:%S')

drivers_df = pd.read_csv('drivers.csv')
passengers_df = pd.read_csv('passengers.csv')

drivers_df['Date/Time'] = drivers_df['Date/Time'].apply(parse_datetime)
passengers_df['Date/Time'] = passengers_df['Date/Time'].apply(parse_datetime)

drivers_df = drivers_df.sort_values(by='Date/Time')
passengers_df = passengers_df.sort_values(by='Date/Time')

drivers_q = Queue()
passengers_q = Queue()


total_waiting_time = 0  # D1
total_driver_profit = 0  # D2

for _, passenger in passengers_df.iterrows():
    passengers_q.put(passenger)

start_time = time.time()


for _, driver in drivers_df.iterrows():
    if not passengers_q.empty():
        passenger = passengers_q.get()
        waiting_time = (driver['Date/Time'] - passenger['Date/Time']).total_seconds() / 60  # Convert to minutes
        total_waiting_time += waiting_time
        total_driver_profit += waiting_time  

end_time = time.time()

average_waiting_time = total_waiting_time / len(passengers_df)
average_driver_profit = total_driver_profit / len(drivers_df)

# Print results
print(f"Total Passenger Waiting Time (D1): {total_waiting_time} minutes")
print(f"Average Waiting Time (D1): {average_waiting_time} minutes")
print(f"Average Driver Profit (D2): {average_driver_profit} minutes")

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")