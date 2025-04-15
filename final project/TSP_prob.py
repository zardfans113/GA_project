import requests

def get_coordinates(address):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key=AIzaSyA65CeniB1QhUdJEUJtJ9IJl4mojffBw5w"
    response = requests.get(url)
    data = response.json()
    
    if data['status'] == 'OK':
        location = data['results'][0]['geometry']['location']
        latitude = location['lat']
        longitude = location['lng']
        return latitude, longitude
    else:
        return None

address = "猴硐貓村"
coordinates = get_coordinates(address)

if coordinates:
    latitude, longitude = coordinates
    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")
else:
    print("Unable to find coordinates for the given address.")