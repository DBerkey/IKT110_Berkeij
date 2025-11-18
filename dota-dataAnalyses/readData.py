"""
Author: Douwe Berkeij
Date: 17-11-2025
"""

import zipfile
import json
import pandas as pd
from pathlib import Path

ZIP_PATH = Path("c:/Users/berke/Documents/UiA/IKT110_Data/dota_games.zip")

def iter_matches(max_matches=None):
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        json_files = [n for n in zf.namelist() if n.startswith("dota_games/") and n.endswith(".json")]
        if max_matches is not None:
            json_files = json_files[:max_matches]

        for path in json_files:
            with zf.open(path) as fp:
                raw = json.load(fp)
            game = raw.get("result", raw)
            yield game

def build_dataframes(max_matches=20000, newRankedMachesCSV="ranked_matches.csv"):
    match_rows = []
    player_rows = []

    for game in iter_matches(max_matches=max_matches):
        match_id = game.get("match_id")
        radiant_win = game.get("radiant_win")
        duration = game.get("duration")
        game_mode = game.get("game_mode")
        lobby_type = game.get("lobby_type")
        human_players = game.get("human_players", 0)

        # Keep only Ranked All Pick, ranked lobby, full-human matches,
        # and exclude matches where any player abandoned.
        if game_mode != 22:
            continue
        if lobby_type != 7:
            continue
        if human_players != 10:
            continue

        players = game.get("players", [])
        # leaver_status = 2 3 or 4 indicates abandonment
        if any(p.get("leaver_status", 0) in {2, 3, 4} for p in players):
            continue

        match_rows.append({
            "match_id": match_id,
            "radiant_win": radiant_win,
            "duration": duration,
            "game_mode": game_mode,
            "lobby_type": lobby_type,
            "human_players": human_players,
        })

        for p in players:
            slot = p.get("player_slot", 0)
            is_radiant = slot < 128
            win = (radiant_win and is_radiant) or ((radiant_win is False) and not is_radiant)
            player_rows.append({
                "match_id": match_id,
                "hero_id": p.get("hero_id"),
                "player_slot": slot,
                "is_radiant": is_radiant,
                "win": win,
                "kills": p.get("kills", 0),
                "deaths": p.get("deaths", 0),
                "assists": p.get("assists", 0),
                "gpm": p.get("gold_per_min", 0),
                "xpm": p.get("xp_per_min", 0),
                "last_hits": p.get("last_hits", 0),
                "denies": p.get("denies", 0),
                "hero_damage": p.get("hero_damage", 0),
                "tower_damage": p.get("tower_damage", 0),
            })

    matches_df = pd.DataFrame(match_rows)
    players_df = pd.DataFrame(player_rows)

    matches_df.to_csv(newRankedMachesCSV, index=False)
    players_df.to_csv(newRankedMachesCSV.replace("matches", "players"), index=False)
    print(f"Data saved to {newRankedMachesCSV} and {newRankedMachesCSV.replace('matches', 'players')}")

    return matches_df, players_df

if __name__ == "__main__":
    newRankedMachesCSV = "c:/Users/berke/Documents/UiA/IKT110_Data/new_ranked_matches.csv"

    matches_df, players_df = build_dataframes(max_matches=2338043, newRankedMachesCSV=newRankedMachesCSV)
    print("Matches DataFrame:")
    print(matches_df.head())
    print("\nPlayers DataFrame:")
    print(players_df.head())
    print(f"\nTotal matches processed: {len(matches_df)}")
