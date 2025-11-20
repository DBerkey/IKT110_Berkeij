"""
Author: Urs Pfrommer
Date: 18-11-2025
Descrition: Pick recommendation for drafting tool
"""

from __future__ import annotations

from .dataLoader import load_match_data, load_counter_synergy_data, load_safe_first_picks, load_list_of_valid_heroes
import numpy as np


def _average(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _collect_scores(row: np.ndarray, indices: list[int]) -> list[float]:
    n_cols = row.shape[0]
    scores = []
    for idx in indices:
        if 0 <= idx < n_cols:
            scores.append(float(row[idx]))
    return scores


def recommend_picks(current_bans: list, current_side: str, current_radiant_picks: list, current_dire_picks: list, counter_matrix: np.ndarray, synergy_matrix: np.ndarray,
                    safe_first_picks: list, top_k: int = 5, valid_heroes: list = None) -> list:
    """
    Recommends hero picks based on current picks using counter and synergy matrices.

    Args:
        current_picks (list): List of currently picked hero IDs.
        counter_matrix (np.ndarray): Hero counter matrix.
        synergy_matrix (np.ndarray): Hero synergy matrix.
        safe_first_picks (list): List of safe first pick hero IDs.
        top_k (int): Number of top recommendations to return.

    Returns:
        list: List of recommended hero IDs.
    """
    num_heroes = counter_matrix.shape[0]
    scores = np.zeros(num_heroes)

    if not current_radiant_picks and not current_dire_picks:
        print("No current picks, recommending safe first picks.")
        return safe_first_picks[:top_k]
    if current_side.lower() == "radiant":
        for hero_id in current_radiant_picks:
            scores += 10 * counter_matrix[hero_id]
            scores -= 10 * synergy_matrix[hero_id]

        for hero_id in current_dire_picks:
            scores -= 10 * counter_matrix[hero_id]
            scores += 10 * synergy_matrix[hero_id]
    else:
        for hero_id in current_dire_picks:
            scores += 10 * counter_matrix[hero_id]
            scores -= 10 * synergy_matrix[hero_id]

        for hero_id in current_radiant_picks:
            scores -= 10 * counter_matrix[hero_id]
            scores += 10 * synergy_matrix[hero_id]

    for picked_hero in current_radiant_picks + current_dire_picks + current_bans:
        scores[picked_hero] = -np.inf

    recommended_indices = np.argsort(scores)[-top_k:][::-1]

    for idx in recommended_indices:
        if idx not in valid_heroes:
            scores[idx] = -np.inf

    recommended_heroes = [
        idx for idx in recommended_indices]

    return recommended_heroes[:top_k]


def recommend_bans(current_bans: list, current_side: str, current_radiant_picks: list, current_dire_picks: list,
                   counter_matrix: np.ndarray, synergy_matrix: np.ndarray, safe_first_picks: list[int], top_k: int = 5) -> list:
    """Suggest heroes to ban for the specified side."""
    num_heroes = counter_matrix.shape[0]
    available = set(range(num_heroes)) - set(current_bans +
                                             current_radiant_picks + current_dire_picks)
    scores = np.zeros(num_heroes)

    side = current_side.lower()
    if side == "radiant":
        threaten_team = current_dire_picks
        target_opponents = current_radiant_picks
    else:
        threaten_team = current_radiant_picks
        target_opponents = current_dire_picks

    if not threaten_team and not target_opponents:
        fallback = [
            hero_id for hero_id in safe_first_picks if hero_id in available]
        return fallback[:top_k]

    for hero_id in available:
        # Favor heroes that synergize with the opposing team and counter our current picks.
        synergy_scores = _collect_scores(
            synergy_matrix[hero_id], threaten_team) if synergy_matrix.size else []
        counter_scores = _collect_scores(
            counter_matrix[hero_id], target_opponents) if counter_matrix.size else []
        synergy_avg = _average(synergy_scores)
        counter_avg = _average(counter_scores)
        # Heavier weight on counter pressure because bans typically remove hard counters.
        scores[hero_id] = 0.6 * counter_avg + 0.4 * synergy_avg

    ranked = [idx for idx in np.argsort(scores)[::-1] if idx in available]
    return ranked[:top_k]


if __name__ == "__main__":
    match_csv = "C:\\Users\\User\\Desktop\\new_ranked_matches.csv"
    heroes_csv = "C:\\Users\\User\\Desktop\\new_ranked_players.csv"
    counter_csv = "C:\\Users\\User\\Desktop\\hero_counter_lookup.csv"
    synergy_csv = "C:\\Users\\User\\Desktop\\hero_synergy_lookup.csv"
    first_pick_json = "dota\\dotaPickRecommendation\\safe_first_picks.json"
    # features, labels, hero_count = load_match_data(match_csv, heroes_csv)
    # print(
    #    f"Loaded {features.shape[0]} matches with {hero_count} unique heroes.")
    demo_counter_matrix, demo_synergy_matrix = load_counter_synergy_data(
        counter_csv, synergy_csv)
    demo_safe_first_picks = load_safe_first_picks(first_pick_json)

    recommendations = recommend_picks([75, 47, 67, 11, 23], "Radiant",
                                      [1, 5, 10], [3, 2], demo_counter_matrix, demo_synergy_matrix, demo_safe_first_picks, top_k=5)
    print(f"Recommended hero picks: {recommendations}")
