"""
Author: Douwe Berkeij
Date: 17-11-2025

Performs offline analysis on the cleaned ranked match data.
Data is loaded from the CSV files produced by `readData.py`.
"""

import pandas as pd
from itertools import combinations


MATCHES_CSV = "C:/Users/berke/Documents/UiA/IKT110_Berkeij/dota/new_ranked_matches.csv"
PLAYERS_CSV = "C:/Users/berke/Documents/UiA/IKT110_Berkeij/dota/new_ranked_players.csv"
COUNTER_LOOKUP_CSV = "C:/Users/berke/Documents/UiA/IKT110_Berkeij/dota/hero_counter_lookup.csv"


def load_data():
	matches_df = pd.read_csv(MATCHES_CSV)
	players_df = pd.read_csv(PLAYERS_CSV)
	return matches_df, players_df


def most_picked_hero(players_df: pd.DataFrame):
	counts = players_df["hero_id"].value_counts()
	hero_id = counts.idxmax()
	return hero_id, int(counts.max())


def hero_highest_winrate(players_df: pd.DataFrame, min_games: int = 100):
	hero_counts = players_df["hero_id"].value_counts()
	hero_wr = players_df.groupby("hero_id")["win"].mean()
	eligible = hero_counts[hero_counts >= min_games].index
	best_hero = hero_wr[eligible].idxmax()
	return best_hero, float(hero_wr[eligible].max()), int(hero_counts[best_hero])


def side_advantage(matches_df: pd.DataFrame, players_df: pd.DataFrame, min_games_side: int = 50):
	radiant_win_rate = float(matches_df["radiant_win"].mean())

	radiant_players = players_df[players_df["is_radiant"]]
	dire_players = players_df[~players_df["is_radiant"]]

	radiant_wr_by_hero = radiant_players.groupby("hero_id")["win"].mean()
	dire_wr_by_hero = dire_players.groupby("hero_id")["win"].mean()

	rad_counts = radiant_players["hero_id"].value_counts()
	dir_counts = dire_players["hero_id"].value_counts()

	common_heroes = radiant_wr_by_hero.index.intersection(dire_wr_by_hero.index)
	eligible = common_heroes[
		(rad_counts[common_heroes] >= min_games_side)
		& (dir_counts[common_heroes] >= min_games_side)
	]

	side_diff = radiant_wr_by_hero[eligible] - dire_wr_by_hero[eligible]
	most_side_dependent_hero = side_diff.abs().idxmax()

	return radiant_win_rate, most_side_dependent_hero, float(radiant_wr_by_hero[most_side_dependent_hero]), float(dire_wr_by_hero[most_side_dependent_hero])


def hero_impact(players_df: pd.DataFrame, min_games: int = 100):
	players_df = players_df.copy()
	players_df["impact"] = (
		players_df["kills"] + 0.5 * players_df["assists"] - players_df["deaths"]
	)
	impact_by_hero = players_df.groupby("hero_id")["impact"].mean()
	counts = players_df["hero_id"].value_counts()
	eligible = counts[counts >= min_games].index
	best_hero = impact_by_hero[eligible].idxmax()
	return best_hero, float(impact_by_hero[best_hero]), int(counts[best_hero])


def hero_game_lengths(matches_df: pd.DataFrame, players_df: pd.DataFrame, min_games: int = 100):
	players = players_df.merge(
		matches_df[["match_id", "duration"]], on="match_id", how="left"
	)
	avg_duration = players.groupby("hero_id")["duration"].mean()
	counts = players["hero_id"].value_counts()
	eligible = counts[counts >= min_games].index

	hero_longest = avg_duration[eligible].idxmax()
	hero_shortest = avg_duration[eligible].idxmin()

	return (
		hero_longest,
		float(avg_duration[hero_longest]),
		hero_shortest,
		float(avg_duration[hero_shortest]),
	)


def best_hero_pairs(players_df: pd.DataFrame, min_pair_games: int = 100):
	pair_stats = {}

	for match_id, group in players_df.groupby("match_id"):
		for is_radiant, team_group in group.groupby("is_radiant"):
			heroes = list(team_group["hero_id"])
			win = bool(team_group["win"].iloc[0])

			for h1, h2 in combinations(sorted(heroes), 2):
				key = (h1, h2)
				if key not in pair_stats:
					pair_stats[key] = {"games": 0, "wins": 0}
				pair_stats[key]["games"] += 1
				if win:
					pair_stats[key]["wins"] += 1

	pairs_df = pd.DataFrame(
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

	eligible_pairs = pairs_df[pairs_df["games"] >= min_pair_games]
	return eligible_pairs.sort_values("winrate", ascending=False)


def counter_stats_df(players_df: pd.DataFrame, min_matchups: int = 50):
	counter_stats = {}

	for match_id, group in players_df.groupby("match_id"):
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


def best_when_not_countered(players_df: pd.DataFrame,
							 counters_df: pd.DataFrame,
							 top_k: int = 5,
							 min_games: int = 200) -> pd.DataFrame:
	"""Approximate 'best hero when not countered by its top K counters'.

	Uses only the hero-vs-hero lookup table (no per-match loops).
	"""

	# Overall winrate and games per hero
	hero_winrate = players_df.groupby("hero_id")["win"].mean()
	hero_games = players_df["hero_id"].value_counts()

	# Top K counters (by highest counter_strength) for each hero
	top_counters = (
		counters_df
		.sort_values(["hero", "counter_strength"], ascending=[True, False])
		.groupby("hero")
		.head(top_k)
	)

	# Average winrate vs top K counters for each hero
	avg_wr_vs_top_counters = (
		top_counters
		.groupby("hero")["winrate_vs_opponent"]
		.mean()
	)

	# Combine into a summary table
	summary = pd.DataFrame(
		{
			"overall_wr": hero_winrate,
			"games": hero_games,
			"wr_vs_top_counters": avg_wr_vs_top_counters,
		}
	).dropna()

	# Approximate 'winrate when not countered'
	summary["approx_wr_without_top_counters"] = (
		summary["overall_wr"]
		+ (summary["overall_wr"] - summary["wr_vs_top_counters"])
	)

	# Require a minimum sample size for stability
	summary = summary[summary["games"] >= min_games]

	# Sort best heroes first
	return summary.sort_values("approx_wr_without_top_counters", ascending=False)


def safe_first_picks(players_df: pd.DataFrame, counters_df: pd.DataFrame, min_games: int = 200):
	hero_winrate = players_df.groupby("hero_id")["win"].mean()
	hero_games = players_df["hero_id"].value_counts()

	avg_counter_strength = counters_df.groupby("hero")["counter_strength"].mean()

	summary = pd.DataFrame(
		{
			"winrate": hero_winrate,
			"games": hero_games,
			"avg_counter_strength": avg_counter_strength,
		}
	).dropna()

	summary = summary[summary["games"] >= min_games]
	summary["score"] = summary["winrate"] - 0.3 * summary["avg_counter_strength"]

	return summary.sort_values("score", ascending=False)


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


if __name__ == "__main__":
	matches_df, players_df = load_data()

	# 2. Most picked hero
	hero, picks = most_picked_hero(players_df)
	print(f"Most picked hero_id: {hero} (picks: {picks})")

	# 3. Hero with highest win rate
	best_hero_wr, wr, games = hero_highest_winrate(players_df)
	print(f"Hero with highest win rate: {best_hero_wr} (WR: {wr:.3f}, games: {games})")

	# 4. Dire vs Radiant advantage & hero most affected by side
	overall_rad_wr, side_hero, rad_wr, dire_wr = side_advantage(matches_df, players_df)
	print(f"Overall Radiant win rate: {overall_rad_wr:.3f}")
	print(
		f"Hero most affected by side: {side_hero} (Radiant WR: {rad_wr:.3f}, Dire WR: {dire_wr:.3f})"
	)

	# 5. Hero with highest impact
	impact_hero, impact_score, impact_games = hero_impact(players_df)
	print(
		f"Hero with highest impact: {impact_hero} (impact: {impact_score:.3f}, games: {impact_games})"
	)

	# 6 & 7. Longest and shortest games per hero
	h_long, dur_long, h_short, dur_short = hero_game_lengths(matches_df, players_df)
	print(f"Hero with longest avg games: {h_long} (duration: {dur_long:.1f}s)")
	print(f"Hero with shortest avg games: {h_short} (duration: {dur_short:.1f}s)")

	# 8. Best hero pair
	best_pairs = best_hero_pairs(players_df)
	print("Best hero pairs (top 5):")
	print(best_pairs.head(5))

	# 9. Hardest counter
	counters_df = counter_stats_df(players_df)
	hardest = hardest_counter(counters_df)
	print("\nHardest counter (hero, counter):")
	print(hardest)

	# 10. Best hero when not countered by top 5 counters
	best_no_counter = best_when_not_countered(players_df, counters_df)
	print("\nBest heroes when not facing their top 5 counters (top 5, approx):")
	print(best_no_counter.head(5))

	# 11. Safe first picks
	safe_picks = safe_first_picks(players_df, counters_df)
	print("\nSafe first-pick candidates (top 5):")
	print(safe_picks.head(5))

	# 12. Hero counter lookup table (continuous -1..1)
	counter_lookup = counter_lookup_table(counters_df)
	counter_lookup.to_csv(COUNTER_LOOKUP_CSV)
	print("\nHero counter lookup table preview:")
	print(counter_lookup.head(5))
	print(f"Saved full lookup table to {COUNTER_LOOKUP_CSV}")


