import requests
import json
data = 'data=[out:json][timeout:25];(node["man_made"="mast"]["communication:mobile_phone"="yes"](around:5000,17.385,78.486););out body;'
r = requests.post('https://overpass-api.de/api/interpreter', data=data)
print(json.dumps(r.json(), indent=2))
