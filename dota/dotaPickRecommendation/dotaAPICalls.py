"""
Author: Urs Pfrommer & Douwe Berkeij
Date: 18-11-2025
Description: Dota 2 API Calls to fetch hero data and roles and save them as JSON files.
"""

import json
import time
import requests


BASE_URL = "https://api.opendota.com/api"

def get_json(url, timeout=100):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()  # raise for HTTP errors (4xx/5xx)
        try:
            return resp.json()
        except ValueError:
            print(f"Failed to decode JSON from {url}")
            print("Response text (truncated):")
            print(resp.text[:500])
            return None
    except requests.RequestException as e:
        print(f"Request error for {url}: {e}")
        return None

def fetch_hero_data():
    url = f"{BASE_URL}/heroes"
    heroes = get_json(url)
    if not heroes:
        return {}
    return {hero['id']: [hero['localized_name'], hero['roles']] for hero in heroes}


def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    hero_id_to_name = fetch_hero_data()
    if not hero_id_to_name:
        print("Could not fetch hero names; exiting.")
    else:
        save_json(hero_id_to_name, "hero_id_to_name.json")
