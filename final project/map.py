#coding=utf-8
import googlemaps
import time
import csv  

gmaps = googlemaps.Client(key='AIzaSyA65CeniB1QhUdJEUJtJ9IJl4mojffBw5w')

# 定義查詢範圍的中心點和半徑
location = (24.9585212,121.647399)
radius = 4000  # meter

# 查詢範圍內的旅遊景點
places_result = gmaps.places_nearby(location=location, keyword='tourist', radius=radius)
results = places_result['results']

while 'next_page_token' in places_result:
    time.sleep(2)
    places_result = gmaps.places_nearby(page_token=places_result['next_page_token'])
    results.extend(places_result['results'])

with open('_Tourist.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
    fieldnames = ['name', 'latitude', 'longitude', 'rating']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quotechar='"', quoting=csv.QUOTE_ALL)

    writer.writeheader()

    # 查詢每個景點的詳細資訊並列出經緯度和評分
    for i, place in enumerate(results, 1):
        place_name = place['name']
        
        # 使用地理編碼 API 取得景點的 Place ID
        geocode_result = gmaps.geocode(place_name)
        if not geocode_result:
            print(f"Could not find location: {place_name}")
            continue
        
        place_id = geocode_result[0]['place_id']
        
        place_details = gmaps.place(place_id=place_id, fields=['name', 'rating', 'geometry'])
        
        if 'result' in place_details:
            result = place_details['result']
            place_name = result.get('name')
            rating = result.get('rating', 'N/A')  # 有些景點可能沒有評分
            location = result['geometry']['location']
            
            # 寫入資料到 CSV
            writer.writerow({
                'name': place_name,
                'latitude': location['lat'],
                'longitude': location['lng'],
                'rating': rating
            })
            
            print(f"{i}: {place_name} - Rating: {rating}, Location: ({location['lat']}, {location['lng']})")
        else:
            print(f"Could not retrieve place details for {place_name}")