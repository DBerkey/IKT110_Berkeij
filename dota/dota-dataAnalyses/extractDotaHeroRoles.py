import json
import csv
from collections import defaultdict
from typing import Dict, List

import requests
from bs4 import BeautifulSoup


ROLE_URL = "https://dota2.fandom.com/wiki/Role"

# The eight "Official Roles" used on the Role page
OFFICIAL_ROLES = [
	"Carry",
	"Support",
	"Nuker",
	"Disabler",
	"Durable",
	"Escape",
	"Pusher",
	"Initiator",
]


def fetch_page(url: str) -> str:
	response = requests.get(url)
	response.raise_for_status()
	return response.text


def normalize_hero_name(name: str) -> str:
	return name.strip().replace("\u00a0", " ")


def extract_official_roles(html: str) -> Dict[str, List[str]]:
	soup = BeautifulSoup(html, "html.parser")

	# Find the heading for "Official Roles" first
	official_header = soup.find("span", {"id": "Official_Roles"})
	if not official_header:
		raise RuntimeError("Could not find Official Roles section on the page")

	hero_to_roles: Dict[str, List[str]] = defaultdict(list)

	# Iterate over the headings (h3) following the Official Roles section
	for header in official_header.parent.find_all_next("h3"):
		span = header.find("span", class_="mw-headline")
		if not span or not span.get("id"):
			continue

		role_name = span.get_text(strip=True)

		# Stop once we reach the next major section ("Unofficial roles")
		if role_name == "Unofficial roles":
			break

		# Only care about the eight official role blocks
		if role_name not in OFFICIAL_ROLES:
			continue

		# The hero table for this role is the first collapsible wikitable after the heading
		table = header.find_next("table", class_="wikitable")
		if not table:
			continue

		# Heroes are listed inside div.heroentrytext in that table
		for div in table.find_all("div", class_="heroentrytext"):
			hero_name = normalize_hero_name(div.get_text())
			if not hero_name:
				continue
			if role_name not in hero_to_roles[hero_name]:
				hero_to_roles[hero_name].append(role_name)

	return dict(hero_to_roles)


def save_as_json(mapping: Dict[str, List[str]], path: str) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(mapping, f, ensure_ascii=False, indent=2)


def main() -> None:
	html = fetch_page(ROLE_URL)
	hero_to_roles = extract_official_roles(html)

	json_path = "official_hero_roles.json"

	save_as_json(hero_to_roles, json_path)

	print(f"Extracted {len(hero_to_roles)} heroes with official roles.")
	for i, (hero, roles) in enumerate(sorted(hero_to_roles.items())):
		if i >= 10:
			break
		print(f"{hero}: {roles}")
	print(f"Saved JSON to {json_path}")


if __name__ == "__main__":
	main()

