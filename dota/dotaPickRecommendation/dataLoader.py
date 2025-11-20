"""
Author: Urs Pfrommer
Date: 18-11-2025
Description: Data loader for Dota 2 match data.
"""

import csv
from typing import List, Tuple
import numpy as np


def load_match_data(match_csv_path: str, player_csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads match and player data from CSV files.

    Args:
        match_csv_path (str): Path to the matches CSV file.
        player_csv_path (str): Path to the players CSV file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - features: Array of shape (num_matches, num_features)
            - labels: Array of shape (num_matches,)
            - hero_count: Total number of unique heroes
    """
    match_data = {}
    hero_set = set()
    # Load match data
    with open(match_csv_path, newline='') as match_file:
        reader = csv.DictReader(match_file)
        for row in reader:
            match_id = row['match_id']
            radiant_win = str(row['radiant_win'])
            match_data[match_id] = {
                'radiant_win': radiant_win,
                'radiant_heroes': [],
                'dire_heroes': []
            }

    # Load player data
    with open(player_csv_path, newline='') as player_file:
        reader = csv.DictReader(player_file)
        for row in reader:
            match_id = row['match_id']
            hero_id = int(row['hero_id'])
            is_radiant = row['is_radiant'].lower() == 'true'
            hero_set.add(hero_id)

            if match_id in match_data:
                if is_radiant:
                    match_data[match_id]['radiant_heroes'].append(hero_id)
                else:
                    match_data[match_id]['dire_heroes'].append(hero_id)
    features = []
    labels = []
    print(f"Total unique heroes: {len(match_data)}")
    for match_id, data in match_data.items():
        if len(data['radiant_heroes']) == 5 and len(data['dire_heroes']) == 5:
            feature_vector = data['radiant_heroes'] + data['dire_heroes']
            features.append(feature_vector)
            labels.append(1 if data['radiant_win'].lower() == 'true' else 0)
    features_array = np.array(features)
    labels_array = np.array(labels)
    hero_count = len(hero_set)
    return features_array, labels_array, hero_count


def _read_lookup_matrix(path: str) -> np.ndarray:
    with open(path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return np.empty((0, 0), dtype=float)

    data_rows = rows[1:]  # drop header row
    matrix = [
        [float(value) for value in row[1:]]  # drop leading hero-id column
        for row in data_rows if row
    ]
    return np.array(matrix, dtype=float)


def load_counter_synergy_data(counter_csv_path: str, synergy_csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads hero counter and synergy data from CSV files.

    Args:
        counter_csv_path (str): Path to the hero counter CSV file.
        synergy_csv_path (str): Path to the hero synergy CSV file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - counter_matrix: 2D array where entry (i, j) indicates how well hero i counters hero j
            - synergy_matrix: 2D array where entry (i, j) indicates how well hero i synergizes with hero j
    """
    counter_matrix = _read_lookup_matrix(counter_csv_path)
    synergy_matrix = _read_lookup_matrix(synergy_csv_path)
    return counter_matrix, synergy_matrix


def load_safe_first_picks(json_path: str) -> List[int]:
    """
    Loads safe first pick hero IDs from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing safe first pick hero IDs.
    Returns:
        List[int]: List of hero IDs considered safe for first pick.
    """
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    safe_picks = list(map(int, data.keys()))
    safe_picks = sorted(
        safe_picks, key=lambda x: data[str(x)]['score'], reverse=True)[:5]
    return safe_picks


if __name__ == "__main__":
    match_csv = "C:\\Users\\User\\Desktop\\new_ranked_matches.csv"
    player_csv = "C:\\Users\\User\\Desktop\\new_ranked_players.csv"
    counter_csv = "C:\\Users\\User\\Desktop\\hero_counter_lookup.csv"
    synergy_csv = "C:\\Users\\User\\Desktop\\hero_synergy_lookup.csv"
    first_pick_json = "dota\\dotaPickRecommendation\\safe_first_picks.json"
    # features, labels, hero_count = load_match_data(match_csv, player_csv)
    # print(
    #    f"Loaded {features.shape[0]} matches with {hero_count} unique heroes.")
    counter_matrix, synergy_matrix = load_counter_synergy_data(
        counter_csv, synergy_csv)
    print(f"Counter matrix shape: {counter_matrix.shape}")
    print(f"Synergy matrix shape: {synergy_matrix.shape}")
    safe_first_picks = load_safe_first_picks(first_pick_json)
    print(f"Safe first pick hero IDs: {safe_first_picks}")
