"""
Author: Urs Pfrommer
Date: 18-11-2025
Descrition: Pick recommendation for drafting tool
"""

from dataLoader import load_match_data
import numpy as np


if __name__ == "__main__":
    match_csv = "C:\\Users\\User\\Desktop\\new_ranked_matches.csv"
    player_csv = "C:\\Users\\User\\Desktop\\new_ranked_players.csv"
    # features, labels, hero_count = load_match_data(match_csv, player_csv)
    # print(
    #    f"Loaded {features.shape[0]} matches with {hero_count} unique heroes.")
