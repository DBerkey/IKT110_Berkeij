"""
Author: Douwe Berkeij
Date: 17-11-2025

Performs offline analysis on the cleaned ranked match data.
Data is loaded from the CSV files produced by `readData.py`.
"""

import json
import pandas as pd
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path


MATCHES_CSV = "C:\\Users\\User\\Desktop\\new_ranked_matches.csv"
PLAYERS_CSV = "C:\\Users\\User\\Desktop\\new_ranked_players.csv"
HERO_METADATA_JSON = "C:\\Users\\User\\Desktop\\UiA\\IKT110\\IKT110_Berkeij\\dota\\hero_id_to_name.json"
COUNTER_LOOKUP_CSV = "C:\\Users\\User\\Desktop\\hero_counter_lookup.csv"
SYNERGY_LOOKUP_CSV = "C:\\Users\\User\\Desktop\\hero_synergy_lookup.csv"


def load_data():
    matches_df = pd.read_csv(MATCHES_CSV)
    players_df = pd.read_csv(PLAYERS_CSV)
    return matches_df, players_df


def load_hero_roles(metadata_path: str) -> tuple[dict[int, set[str]], list[str]]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hero_roles: dict[int, set[str]] = {}
    role_universe: set[str] = set()
    for hero_id_str, payload in data.items():
        hero_id = int(hero_id_str)
        roles = payload[1] if isinstance(
            payload, list) and len(payload) > 1 else []
        role_set = set(roles)
        hero_roles[hero_id] = role_set
        role_universe.update(role_set)

    return hero_roles, sorted(role_universe)


def most_picked_hero(hero_counts: pd.Series):
    hero_id = hero_counts.idxmax()
    return int(hero_id), int(hero_counts.loc[hero_id])


def hero_highest_winrate(hero_winrate: pd.Series, hero_counts: pd.Series, min_games: int = 100):
    eligible = hero_counts[hero_counts >= min_games].index
    wr_slice = hero_winrate.loc[eligible]
    best_hero = wr_slice.idxmax()
    return int(best_hero), float(wr_slice.loc[best_hero]), int(hero_counts.loc[best_hero])


def hero_impact(impact_by_hero: pd.Series, hero_counts: pd.Series, min_games: int = 100):
    eligible = hero_counts[hero_counts >= min_games].index
    impact_slice = impact_by_hero.loc[eligible]
    best_hero = impact_slice.idxmax()
    return int(best_hero), float(impact_slice.loc[best_hero]), int(hero_counts.loc[best_hero])


def side_advantage(matches_df: pd.DataFrame, players_df: pd.DataFrame, min_games_side: int = 50):
    radiant_win_rate = float(matches_df["radiant_win"].mean())

    radiant_players = players_df[players_df["is_radiant"]]
    dire_players = players_df[~players_df["is_radiant"]]

    radiant_wr_by_hero = radiant_players.groupby("hero_id")["win"].mean()
    dire_wr_by_hero = dire_players.groupby("hero_id")["win"].mean()

    rad_counts = radiant_players["hero_id"].value_counts()
    dir_counts = dire_players["hero_id"].value_counts()

    common_heroes = radiant_wr_by_hero.index.intersection(
        dire_wr_by_hero.index)
    eligible = common_heroes[
        (rad_counts[common_heroes] >= min_games_side)
        & (dir_counts[common_heroes] >= min_games_side)
    ]

    side_diff = radiant_wr_by_hero[eligible] - dire_wr_by_hero[eligible]
    most_side_dependent_hero = side_diff.abs().idxmax()

    return radiant_win_rate, most_side_dependent_hero, float(radiant_wr_by_hero[most_side_dependent_hero]), float(dire_wr_by_hero[most_side_dependent_hero])


def hero_game_lengths(players_with_duration: pd.DataFrame, hero_counts: pd.Series, min_games: int = 100):
    avg_duration = players_with_duration.groupby(
        "hero_id", observed=True, sort=False)["duration"].mean()
    eligible = hero_counts[hero_counts >= min_games].index
    avg_duration = avg_duration.loc[eligible]
    hero_longest = avg_duration.idxmax()
    hero_shortest = avg_duration.idxmin()
    return (
        int(hero_longest),
        float(avg_duration.loc[hero_longest]),
        int(hero_shortest),
        float(avg_duration.loc[hero_shortest]),
    )


def hero_pair_stats(players_df: pd.DataFrame) -> pd.DataFrame:
    pair_stats: dict[tuple[int, int], dict[str, int]] = {}

    for _, group in players_df.groupby("match_id", sort=False):
        for _, team_group in group.groupby("is_radiant", sort=False):
            heroes = list(team_group["hero_id"])
            win = bool(team_group["win"].iloc[0])

            for h1, h2 in combinations(sorted(heroes), 2):
                key = (h1, h2)
                stats = pair_stats.setdefault(key, {"games": 0, "wins": 0})
                stats["games"] += 1
                if win:
                    stats["wins"] += 1

    return pd.DataFrame(
        [
            {
                "hero1": k[0],
                "hero2": k[1],
                "games": v["games"],
                "winrate": v["wins"] / v["games"],
            }
            for k, v in pair_stats.items()
        ]
    )


def best_hero_pairs(players_df: pd.DataFrame, min_pair_games: int = 100):
    pairs_df = hero_pair_stats(players_df)
    eligible_pairs = pairs_df[pairs_df["games"] >= min_pair_games]
    return eligible_pairs.sort_values("winrate", ascending=False)


def counter_stats_df(players_df: pd.DataFrame, min_matchups: int = 50):
    counter_stats = {}

    for match_id, group in players_df.groupby("match_id", sort=False):
        rad = group[group["is_radiant"]]
        dire = group[~group["is_radiant"]]
        radiant_win = bool(rad["win"].iloc[0])

        rad_heroes = list(rad["hero_id"])
        dire_heroes = list(dire["hero_id"])

        for h_r in rad_heroes:
            for h_d in dire_heroes:
                key_rd = (h_r, h_d)
                if key_rd not in counter_stats:
                    counter_stats[key_rd] = {"games": 0, "wins": 0}
                counter_stats[key_rd]["games"] += 1
                if radiant_win:
                    counter_stats[key_rd]["wins"] += 1

                key_dr = (h_d, h_r)
                if key_dr not in counter_stats:
                    counter_stats[key_dr] = {"games": 0, "wins": 0}
                counter_stats[key_dr]["games"] += 1
                if not radiant_win:
                    counter_stats[key_dr]["wins"] += 1

    counters_df = pd.DataFrame(
        [
            {
                "hero": k[0],
                "opponent": k[1],
                "games": v["games"],
                "winrate_vs_opponent": v["wins"] / v["games"],
            }
            for k, v in counter_stats.items()
        ]
    )

    counters_df = counters_df[counters_df["games"] >= min_matchups]
    counters_df["counter_strength"] = 1.0 - counters_df["winrate_vs_opponent"]
    return counters_df


def hardest_counter(counters_df: pd.DataFrame):
    return counters_df.sort_values("counter_strength", ascending=False).head(1)


def best_when_not_countered(hero_winrate: pd.Series,
                            hero_counts: pd.Series,
                            counters_df: pd.DataFrame,
                            top_k: int = 5,
                            min_games: int = 200) -> pd.DataFrame:
    top_counters = (
        counters_df
        .sort_values(["hero", "counter_strength"], ascending=[True, False])
        .groupby("hero", sort=False)
        .head(top_k)
    )
    avg_wr_vs_top_counters = (
        top_counters
        .groupby("hero", sort=False)["winrate_vs_opponent"]
        .mean()
    )
    summary = pd.DataFrame(
        {
            "overall_wr": hero_winrate,
            "games": hero_counts,
            "wr_vs_top_counters": avg_wr_vs_top_counters,
        }
    ).dropna()
    summary = summary[summary["games"] >= min_games]
    summary["approx_wr_without_top_counters"] = (
        summary["overall_wr"]
        + (summary["overall_wr"] - summary["wr_vs_top_counters"])
    )
    return summary.sort_values("approx_wr_without_top_counters", ascending=False)


def safe_first_picks(hero_winrate: pd.Series,
                     hero_counts: pd.Series,
                     counters_df: pd.DataFrame,
                     min_games: int = 200):

    avg_counter_strength = counters_df.groupby("hero", sort=False)[
        "counter_strength"].mean()
    summary = pd.DataFrame(
        {
            "winrate": hero_winrate,
            "games": hero_counts,
            "avg_counter_strength": avg_counter_strength,
        }
    ).dropna()
    summary = summary[summary["games"] >= min_games]
    summary["score"] = summary["winrate"] - \
        0.3 * summary["avg_counter_strength"]
    ranked = summary.sort_values("score", ascending=False)

    output_dir = Path("./dotaPickRecommendation")
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked.reset_index().rename(columns={"index": "hero_id"}).to_json(
        output_dir / "safe_first_picks.json",
        orient="records",
        indent=2,
    )

    return ranked


def counter_lookup_table(counters_df: pd.DataFrame) -> pd.DataFrame:
    """Build hero-vs-hero lookup scored in the continuous range [-1, 1]."""
    scored = counters_df.copy()
    scored["score"] = 2.0 * (scored["winrate_vs_opponent"] - 0.5)
    scored["score"] = scored["score"].clip(-1.0, 1.0)

    lookup = (
        scored
        .pivot(index="hero", columns="opponent", values="score")
        .fillna(0.0)
    )
    return lookup


def synergy_lookup_table(
        players_df: pd.DataFrame,
        counters_df: pd.DataFrame,
        hero_roles: dict[int, set[str]],
        top_counter_k: int = 3,
        min_pair_games: int = 50,
        win_weight: float = 0.6,
        role_weight: float = 0.2,
        counter_weight: float = 0.2,
) -> pd.DataFrame:
    """Create hero synergy matrix combining win, role diversity, and counter overlap."""
    pairs_df = hero_pair_stats(players_df)
    pairs_df = pairs_df[pairs_df["games"] >= min_pair_games].copy()
    if pairs_df.empty:
        return pd.DataFrame()

    pairs_df["win_score"] = (2.0 * (pairs_df["winrate"] - 0.5)).clip(-1.0, 1.0)

    top_counters = (
        counters_df
        .sort_values(["hero", "counter_strength"], ascending=[True, False])
        .groupby("hero")
        .head(top_counter_k)
        .groupby("hero")["opponent"]
        .apply(set)
        .to_dict()
    )

    def role_score(hero_a: int, hero_b: int) -> float:
        roles_a = hero_roles.get(hero_a, set())
        roles_b = hero_roles.get(hero_b, set())
        if not roles_a and not roles_b:
            return 0.0
        union = roles_a | roles_b
        if not union:
            return 0.0
        overlap = roles_a & roles_b
        diversity = 1.0 - (len(overlap) / len(union))
        return (diversity * 2.0) - 1.0

    def counter_score(hero_a: int, hero_b: int) -> float:
        counters_a = top_counters.get(hero_a, set())
        counters_b = top_counters.get(hero_b, set())
        if not counters_a and not counters_b:
            return 0.0
        shared = len(counters_a & counters_b)
        score = 1.0 - (shared / max(1, top_counter_k))
        return (score * 2.0) - 1.0

    rows: list[dict[str, float]] = []
    for _, row in pairs_df.iterrows():
        h1 = int(row["hero1"])
        h2 = int(row["hero2"])
        win_score = float(row["win_score"])
        roles_component = role_score(h1, h2)
        counter_component = counter_score(h1, h2)
        synergy = (
            win_weight * win_score
            + role_weight * roles_component
            + counter_weight * counter_component
        )
        rows.append({"hero": h1, "ally": h2, "synergy": synergy})
        rows.append({"hero": h2, "ally": h1, "synergy": synergy})

    synergy_df = pd.DataFrame(rows)
    lookup = (
        synergy_df
        .pivot(index="hero", columns="ally", values="synergy")
        .fillna(0.0)
    )
    return lookup


@dataclass(slots=True)
class HeroAggregates:
    counts: pd.Series
    winrate: pd.Series
    impact: pd.Series


def optimize_frames(matches_df: pd.DataFrame, players_df: pd.DataFrame):
    matches_df = matches_df.copy()
    if "duration" in matches_df:
        matches_df["duration"] = matches_df["duration"].astype("int32")
    if "match_id" in matches_df:
        matches_df["match_id"] = matches_df["match_id"].astype("int64")

    players_df = players_df.copy()
    players_df["match_id"] = players_df["match_id"].astype("int64")
    players_df["hero_id"] = players_df["hero_id"].astype("int16")
    players_df["is_radiant"] = players_df["is_radiant"].astype(bool)
    players_df["win"] = players_df["win"].astype("int8")
    for col in ("kills", "assists", "deaths"):
        if col in players_df:
            players_df[col] = players_df[col].astype("int16")
    return matches_df, players_df


def build_hero_aggregates(players_df: pd.DataFrame) -> HeroAggregates:
    hero_group = players_df.groupby("hero_id", observed=True, sort=False)
    counts = hero_group.size()
    winrate = hero_group["win"].mean()
    impact = (
        (players_df["kills"] + 0.5 *
         players_df["assists"] - players_df["deaths"])
        .groupby(players_df["hero_id"])
        .mean()
    )
    return HeroAggregates(counts=counts, winrate=winrate, impact=impact)


def attach_match_durations(players_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    return players_df.merge(
        matches_df[["match_id", "duration"]],
        on="match_id",
        how="left",
    )


if __name__ == "__main__":
    matches_df, players_df = load_data()
    matches_df, players_df = optimize_frames(matches_df, players_df)
    hero_roles, role_labels = load_hero_roles(HERO_METADATA_JSON)

    hero_aggregates = build_hero_aggregates(players_df)
    players_with_duration = attach_match_durations(players_df, matches_df)

    hero, picks = most_picked_hero(hero_aggregates.counts)
    print(f"Most picked hero_id: {hero} (picks: {picks})")

    best_hero_wr, wr, games = hero_highest_winrate(
        hero_aggregates.winrate, hero_aggregates.counts)
    print(
        f"Hero with highest win rate: {best_hero_wr} (WR: {wr:.3f}, games: {games})")

    overall_rad_wr, side_hero, rad_wr, dire_wr = side_advantage(
        matches_df, players_df)
    print(f"Overall Radiant win rate: {overall_rad_wr:.3f}")
    print(
        f"Hero most affected by side: {side_hero} (Radiant WR: {rad_wr:.3f}, Dire WR: {dire_wr:.3f})"
    )

    impact_hero, impact_score, impact_games = hero_impact(
        hero_aggregates.impact, hero_aggregates.counts
    )
    print(
        f"Hero with highest impact: {impact_hero} (impact: {impact_score:.3f}, games: {impact_games})"
    )

    h_long, dur_long, h_short, dur_short = hero_game_lengths(
        players_with_duration, hero_aggregates.counts)
    print(f"Hero with longest avg games: {h_long} (duration: {dur_long:.1f}s)")
    print(
        f"Hero with shortest avg games: {h_short} (duration: {dur_short:.1f}s)")

    best_pairs = best_hero_pairs(players_df)
    print("Best hero pairs (top 5):")
    print(best_pairs.head(5))

    counters_df = counter_stats_df(players_df)
    hardest = hardest_counter(counters_df)
    print("\nHardest counter (hero, counter):")
    print(hardest)

    best_no_counter = best_when_not_countered(
        hero_aggregates.winrate, hero_aggregates.counts, counters_df)
    print("\nBest heroes when not facing their top 5 counters (top 5, approx):")
    print(best_no_counter.head(5))

    safe_picks = safe_first_picks(
        hero_aggregates.winrate, hero_aggregates.counts, counters_df)
    print("\nSafe first-pick candidates (top 5):")
    print(safe_picks.head(5))

    counter_lookup = counter_lookup_table(counters_df)
    counter_lookup.to_csv(COUNTER_LOOKUP_CSV)
    print("\nHero counter lookup table preview:")
    print(counter_lookup.head(5))
    print(f"Saved full lookup table to {COUNTER_LOOKUP_CSV}")

    synergy_lookup = synergy_lookup_table(players_df, counters_df, hero_roles)
    synergy_lookup.to_csv(SYNERGY_LOOKUP_CSV)
    print("\nHero synergy lookup table preview:")
    print(synergy_lookup.head(5))
    print(f"Saved full lookup table to {SYNERGY_LOOKUP_CSV}")
