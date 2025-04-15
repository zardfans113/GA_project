import googlemaps   ##導入googlemaps模組

gmaps = googlemaps.Client(key='YOUR_API_KEY') ##利用API建立客戶端

# Geocoding an address
geocode_result = gmaps.geocode('Taiwan')[0] ##利用Geocode函數進行定位
location = geocode_result['geometry']['location'] #取得定位後經緯度


#location回傳格式如下圖，可依照格式自訂義變數，就不用調用Geodcode API

# Search for places
#(keyword參數="輸入你想查詢的物件",radius="公尺單位")

places_result = gmaps.places_nearby(location, keyword='滷肉飯', radius=5000000)

##列印出地點的名稱與地點
for place in places_result['results']:
    print(place['name'], place['formatted_address'])