"""
Author: Douwe Berkeij
Date: 18-11-2025
"""

from typing import List, Tuple
import numpy as np
import csv

def load_data(filepathMatchCSV: str, filepathPlayerCSV: str) -> Tuple[List[Tuple[List[int], List[int], int]], int]:
    """
    Loads match data from two CSV files:
    - matches CSV: match_id, radiant_win, ...
    - players CSV: match_id, hero_id, ..., is_radiant (1 for radiant, 0 for dire)
    Returns:
    - List of tuples, each containing (radiant_heroes, dire_heroes, winner)
    """
    # First load winners per match into a dict
    match_winners = {}
    n_heroes = 0
    with open(filepathMatchCSV, newline='') as f_match:
        reader = csv.reader(f_match)
        for row in reader:
            if not row:
                continue
            match_id = row[0]
            radiant_win = row[1]
            match_winners[match_id] = radiant_win

    matches: List[Tuple[List[int], List[int], int]] = []

    # Now collect heroes per match
    with open(filepathPlayerCSV, newline='') as f_players:
        reader = csv.reader(f_players)
        current_match_id = None
        radiant_heroes: List[int] = []
        dire_heroes: List[int] = []

        for row in reader:
            if not row:
                continue
            if row[0] == 'match_id':
                continue
            match_id_player = row[0]
            hero_id = int(row[1])
            is_radiant = row[3]

            # New match: flush previous one if complete
            if current_match_id is not None and match_id_player != current_match_id:
                if len(radiant_heroes) == 5 and len(dire_heroes) == 5 and current_match_id in match_winners:
                    winner = match_winners[current_match_id]
                    matches.append((radiant_heroes, dire_heroes, winner))
                # Reset for new match
                radiant_heroes = []
                dire_heroes = []

            current_match_id = match_id_player

            if hero_id + 1 > n_heroes:
                n_heroes = hero_id + 1

            if is_radiant == "True":
                radiant_heroes.append(hero_id)
            else:
                dire_heroes.append(hero_id)

        # Flush last match
        if current_match_id is not None:
            if len(radiant_heroes) == 5 and len(dire_heroes) == 5 and current_match_id in match_winners:
                winner = match_winners[current_match_id]
                matches.append((radiant_heroes, dire_heroes, winner))

    return matches, n_heroes

