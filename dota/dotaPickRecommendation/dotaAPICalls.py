"""
Author: Urs Pfrommer & Douwe Berkeij
Date: 18-11-2025
Description: Dota 2 API Calls to fetch hero data and roles and save them as JSON files.
"""

import json
import time
import requests


def fetch_hero_data():
    url = "https://api.opendota.com/api/heroes"
    heroes = requests.get(url, timeout=100).json()
    hero_id_to_name = {hero['id']: hero['localized_name'] for hero in heroes}
    return hero_id_to_name


def fetch_hero_roles():
    url = "https://api.opendota.com/api/heroStats"
    heroes_stats = requests.get(url, timeout=100).json()
    hero_id_to_roles = {hero['id']: hero['roles'] for hero in heroes_stats}
    return hero_id_to_roles


def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    hero_id_to_name = fetch_hero_data()
    save_json(hero_id_to_name, "hero_id_to_name.json")
    time.sleep(1)
    hero_id_to_roles = fetch_hero_roles()
    save_json(hero_id_to_roles, "hero_id_to_roles.json")
