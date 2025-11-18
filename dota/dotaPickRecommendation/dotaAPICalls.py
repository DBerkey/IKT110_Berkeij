import requests
import json

url = "https://api.opendota.com/api/heroes"
heroes = requests.get(url, timeout=60).json()
hero_id_to_name = {hero['id']: hero['localized_name'] for hero in heroes}
print(hero_id_to_name)

url = "https://api.opendota.com/api/heroStats"
heroes_stats = requests.get(url, timeout=60).json()
hero_id_to_roles = {hero['id']: hero['roles'] for hero in heroes_stats}
print(hero_id_to_roles)

# save data to json files
with open("hero_id_to_name.json", "w") as f:
    json.dump(hero_id_to_name, f)

with open("hero_id_to_roles.json", "w") as f:
    json.dump(hero_id_to_roles, f)
