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


if __name__ == "__main__":
    match_csv = "C:\\Users\\User\\Desktop\\new_ranked_matches.csv"
    player_csv = "C:\\Users\\User\\Desktop\\new_ranked_players.csv"
    features, labels, hero_count = load_match_data(match_csv, player_csv)
    print(
        f"Loaded {features.shape[0]} matches with {hero_count} unique heroes.")
