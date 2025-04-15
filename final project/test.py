import googlemaps
from geopy import distance

# 設置 Google Maps API 金鑰
gmaps = googlemaps.Client(key='AIzaSyA65CeniB1QhUdJEUJtJ9IJl4mojffBw5w')

# 城市陣列
city_array = ["新竹", "苗栗", "台中", "彰化", "南投", "雲林", "嘉義", "台南", "高雄", "屏東", "宜蘭", "花蓮", "台東", "台北", "新北", "桃園", "基隆"]

# 定義每個城市的知名景點類型關鍵字
city_types = {city: "tourist_attraction|night_market|university|park|shopping_mall" for city in city_array}

# 起點座標:陽明交通大學光復校區
origin = (24.9576, 121.1917)

# 儲存所有景點座標及其所屬城市
all_places = []

# 遍歷每個城市,找出前 10 名最高評分的景點中評分最高的景點
for city in city_array:
    types = city_types.get(city, "tourist_attraction|night_market|university|park|shopping_mall")
    top_places = []

    # 在城市附近 100 公里範圍內搜尋前 10 名最高評分的景點
    for radius in range(100, 10001, 100):
        places_result = gmaps.places_nearby(location=origin, radius=radius, type=types)

        # 將搜尋結果中的前 10 名最高評分的景點添加到 top_places 列表中
        for place in places_result['results']:
            rating = place.get('rating', 0)
            if len(top_places) < 10:
                top_places.append((rating, place))
            else:
                top_places.sort(key=lambda x: x[0], reverse=True)
                if rating > top_places[-1][0]:
                    top_places[-1] = (rating, place)

    # 從前 10 名最高評分的景點中選擇評分最高的作為該城市的代表景點
    if top_places:
        best_place = max(top_places, key=lambda x: x[0])[1]
        place_location = (best_place['geometry']['location']['lat'], best_place['geometry']['location']['lng'])
        all_places.append((place_location, city))
        origin = place_location

# 檢查 all_places 是否為空
if not all_places:
    print("No places found. Exiting.")
    exit()

# 計算最短路徑
path = [all_places[0][0]]  # 起點為陽明交通大學光復校區
unvisited = set(all_places)
unvisited.remove(all_places[0])
while unvisited:
    nearest = min(unvisited, key=lambda x: distance.distance(path[-1], x[0]).km)
    path.append(nearest[0])
    unvisited.remove(nearest)

# 輸出路徑
total_distance = 0
for i in range(len(path) - 1):
    place1, place2 = path[i], path[i+1]
    dist = distance.distance(place1, place2).km
    total_distance += dist
    city1 = next(x[1] for x in all_places if x[0] == place1)
    city2 = next(x[1] for x in all_places if x[0] == place2)
    print(f"從 {city1} 到 {city2}, 距離: {dist:.2f} 公里")

print(f"總路徑長度: {total_distance:.2f} 公里")