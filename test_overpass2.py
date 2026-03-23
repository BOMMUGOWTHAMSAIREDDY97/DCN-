import requests
import json
data = 'data=[out:json][timeout:25];(node["man_made"="mast"]["communication:mobile_phone"="yes"](around:50000,19.0760,72.8777);node["man_made"="telecommunications_tower"](around:50000,19.0760,72.8777););out body;'
r = requests.post('https://overpass-api.de/api/interpreter', data=data)
print(json.dumps(r.json(), indent=2))
