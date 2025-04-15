import pandas as pd
import googlemaps
# Load data from a CSV file
distances = []
df = pd.read_csv('台北大巨蛋_Tourist.csv')
gmaps = googlemaps.Client(key='AIzaSyA65CeniB1QhUdJEUJtJ9IJl4mojffBw5w')
for i in range(10):
    row = []
    for j in range(10):
        if i == j:
            row.append(0)
        else:
            origin = df.loc[i, 'name']
            destination = df.loc[j, 'name']
            result = gmaps.distance_matrix(origin, destination)
            if 'distance' in result['rows'][0]['elements'][0]:
                distance = result['rows'][0]['elements'][0]['distance']['value']
                row.append(distance)
            else:
                row.append(None)
    distances.append(row)

for i in range (10):
    print(distances[i], end='\n')