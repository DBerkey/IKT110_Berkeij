# ----------------------------------------------------------------------------#
# Imports
# ----------------------------------------------------------------------------#

import csv
import getpass
import json
import os
import sys
import logging
import numpy as np
from threading import Lock
import requests as req
from flask import Flask, render_template, request, jsonify, g, redirect, url_for, send_from_directory
from functools import lru_cache

# The ML model files
# Make sure the project root (which contains the training code) is importable.
package_directory = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(package_directory, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from dota.dotaWinPredition.winPredictor import DotaWinPredictor, load_lookup_matrix
from dota.dotaPickRecommendation.pickRecommender import (
    recommend_picks,
    recommend_bans,
    load_counter_synergy_data,
    load_safe_first_picks,
)

# ----------------------------------------------------------------------------#
# Configs
# ----------------------------------------------------------------------------#

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

frontend_port = 5000
app = Flask(__name__, static_folder='static', template_folder='templates')

# TODO: Remove before prod
running_user = getpass.getuser()

app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
debug_mode = True

model_path = os.path.join(repo_root, "dota", "dota_win_predictor_model.npz")
counter_lookup_path = os.path.join(repo_root, "dota", "hero_counter_lookup.csv")
synergy_lookup_path = os.path.join(repo_root, "dota", "hero_synergy_lookup.csv")
safe_first_picks_path = os.path.join(repo_root, "dota", "dotaPickRecommendation", "safe_first_picks.json")

_pick_assets = None
_pick_assets_lock = Lock()

_win_predictor = None
_predictor_lock = Lock()

# ----------------------------------------------------------------------------#
# Renders
# ----------------------------------------------------------------------------#


@app.route("/")
def index():
    return render_template("index.html", heroes=get_heroes())


@app.route("/explore")
def explore():
    return render_template("explore.html", heroes=get_heroes())


# ----------------------------------------------------------------------------#
# Oracle
# ----------------------------------------------------------------------------#

# <button 1>
@app.route("/suggest1", methods=["POST"])
def suggest1():
    payload = request.get_json(silent=True) or {}
    recommendations = _get_recommendations(payload, "radiant")
    return _html_or_error(recommendations, heading="Suggested Picks")


# <button 2>
@app.route("/suggest2", methods=["POST"])
def suggest2():
    payload = request.get_json(silent=True) or {}
    recommendations = _get_recommendations(payload, "dire")
    return _html_or_error(recommendations, heading="Suggested Picks")


@app.route("/suggestBanRadiant", methods=["POST"])
def suggest_ban_radiant():
    payload = request.get_json(silent=True) or {}
    recommendations = _get_ban_recommendations(payload, "radiant")
    return _html_or_error(recommendations, heading="Suggested Bans")


@app.route("/suggestBanDire", methods=["POST"])
def suggest_ban_dire():
    payload = request.get_json(silent=True) or {}
    recommendations = _get_ban_recommendations(payload, "dire")
    return _html_or_error(recommendations, heading="Suggested Bans")


# <button 3>
@app.route("/suggest3", methods=["POST"])
def suggest3():
    payload = request.get_json(silent=True) or {}
    try:
        radiant = _coerce_ids(payload.get("radiant", []))
        dire = _coerce_ids(payload.get("dire", []))
        bans = _coerce_ids(payload.get("bans", []))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if len(radiant) != 5 or len(dire) != 5:
        return jsonify({"error": "Please provide 5 heroes for Radiant and 5 for Dire."}), 400

    try:
        predictor = _get_win_predictor()
        radiant_prob = predictor.predict(radiant, dire)
    except FileNotFoundError as exc:
        return jsonify({"error": f"Missing model files: {exc}"}), 500
    except Exception as exc:  # pragma: no cover - guard rail for unexpected issues
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    dire_prob = 1.0 - radiant_prob
    response_html = _format_prediction_html(radiant_prob, dire_prob)
    return response_html, 200, {"Content-Type": "text/html; charset=utf-8"}

# ----------------------------------------------------------------------------#
# Explore
# ----------------------------------------------------------------------------#


@app.route("/stats/<int:heroid>", methods=["GET"])
def get_hero_stats(heroid):
    hero_lookup = get_hero_lookup()
    hero = hero_lookup.get(heroid)
    if hero is None:
        return jsonify({"error": f"Unknown hero id {heroid}"}), 404

    safe_pick_stats = _get_safe_pick_stats()
    safe_entry = safe_pick_stats.get(heroid, {})

    logging.debug(f"Safe pick stats for hero {heroid}: {safe_entry}")
    logging.debug(f"Hero data: {hero}")
    logging.debug(f"Hero name: {hero.get('name')}")
    logging.debug(_top_synergy_partners(heroid, limit=3))

    top_pairs_payload = []
    for partner_id, synergy_score in _top_synergy_partners(heroid, limit=3):
        partner_entry = _hero_entry(partner_id)
        partner_entry["synergy_score"] = round(float(synergy_score), 3)
        partner_entry["image"] = url_for("static", filename=f"img/avatar-sb/{partner_id}.png")
        logging.debug(f"Partner entry: {partner_entry}")
        top_pairs_payload.append(partner_entry)

    win_rate_value = safe_entry.get("winrate")
    hero_stats = {
        "hero_id": heroid,
        "hero": hero.get("name", f"Hero {heroid}"),
        "win_rate": win_rate_value,
        "win_rate_percent": f"{win_rate_value * 100:.2f}%" if win_rate_value is not None else None,
        "games_sampled": safe_entry.get("games"),
        "safe_pick_score": safe_entry.get("score"),
        "avg_counter_strength": safe_entry.get("avg_counter_strength"),
        "best_pairs": top_pairs_payload,
    }

    return jsonify(hero_stats)

# ----------------------------------------------------------------------------#
# Helpers
# ----------------------------------------------------------------------------#
@lru_cache(maxsize=1)
def get_heroes():
    print(os.getcwd())
    with open(os.path.join(package_directory, "data/heroes.json"), "r") as fp:
        heroes = json.load(fp)
    
    return heroes


@lru_cache(maxsize=1)
def get_hero_lookup():
    lookup = {}
    for hero in get_heroes():
        lookup[int(hero["id"])] = hero
    return lookup


@lru_cache(maxsize=1)
def _valid_hero_ids() -> list[int]:
    return sorted(get_hero_lookup().keys())


def _get_pick_assets():
    global _pick_assets
    with _pick_assets_lock:
        if _pick_assets is None:
            counter_matrix, synergy_matrix = load_counter_synergy_data(counter_lookup_path, synergy_lookup_path)
            counter_matrix = _reindex_lookup_matrix(counter_matrix, counter_lookup_path)
            synergy_matrix = _reindex_lookup_matrix(synergy_matrix, synergy_lookup_path)
            safe_first_picks = load_safe_first_picks(safe_first_picks_path)
            _pick_assets = (counter_matrix, synergy_matrix, safe_first_picks)
    return _pick_assets


@lru_cache(maxsize=1)
def _get_safe_pick_stats() -> dict[int, dict[str, float | int | None]]:
    try:
        with open(safe_first_picks_path, "r", encoding="utf-8") as fp:
            raw_stats = json.load(fp)
    except FileNotFoundError:
        return {}

    parsed: dict[int, dict[str, float | int | None]] = {}
    for hero_id_str, payload in raw_stats.items():
        try:
            hero_id = int(hero_id_str)
        except (TypeError, ValueError):
            continue
        parsed[hero_id] = {
            "winrate": float(payload.get("winrate")) if payload.get("winrate") is not None else None,
            "avg_counter_strength": float(payload.get("avg_counter_strength")) if payload.get("avg_counter_strength") is not None else None,
            "score": float(payload.get("score")) if payload.get("score") is not None else None,
        }
    return parsed


def _reindex_lookup_matrix(matrix: np.ndarray, csv_path: str) -> np.ndarray:
    """Aligns a lookup matrix so indices match actual hero IDs from the CSV."""
    if matrix.size == 0:
        return matrix

    try:
        with open(csv_path, newline='', encoding='utf-8') as fp:
            reader = csv.reader(fp)
            header = next(reader, None)
            if not header:
                return matrix

            col_ids: list[int | None] = []
            for value in header[1:]:
                try:
                    col_ids.append(int(value))
                except (TypeError, ValueError):
                    col_ids.append(None)

            row_ids: list[int | None] = []
            for row in reader:
                if not row:
                    continue
                try:
                    row_ids.append(int(row[0]))
                except (TypeError, ValueError):
                    row_ids.append(None)
    except FileNotFoundError:
        return matrix

    usable_rows = min(len(row_ids), matrix.shape[0])
    usable_cols = min(len(col_ids), matrix.shape[1])

    max_row_id = max((rid for rid in row_ids if rid is not None), default=-1)
    max_col_id = max((cid for cid in col_ids if cid is not None), default=-1)
    max_id = max(max_row_id, max_col_id)
    if max_id < 0:
        return matrix

    reindexed = np.zeros((max_id + 1, max_id + 1), dtype=matrix.dtype)
    for row_idx in range(usable_rows):
        hero_i = row_ids[row_idx]
        if hero_i is None:
            continue
        for col_idx in range(usable_cols):
            hero_j = col_ids[col_idx]
            if hero_j is None:
                continue
            reindexed[hero_i, hero_j] = matrix[row_idx, col_idx]
    return reindexed


def _top_synergy_partners(hero_id: int, limit: int = 3) -> list[tuple[int, float]]:
    if limit <= 0:
        return []
    _, synergy_matrix, _ = _get_pick_assets()
    if synergy_matrix.size == 0:
        return []

    row_ids, col_ids, hero_to_row = _get_synergy_index_maps()
    row_idx = hero_to_row.get(hero_id)
    if row_idx is None or row_idx >= synergy_matrix.shape[0]:
        return []

    row = synergy_matrix[row_idx]
    candidates = []
    for col_idx, score in enumerate(row):
        partner_id = col_ids[col_idx] if col_idx < len(col_ids) else None
        if partner_id is None or partner_id == hero_id:
            continue
        candidates.append((partner_id, float(score)))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:limit]


@lru_cache(maxsize=1)
def _get_synergy_index_maps() -> tuple[list[int], list[int | None], dict[int, int]]:
    """Map hero ids to their row/column indices using CSV headers."""
    row_ids: list[int] = []
    col_ids: list[int | None] = []
    hero_to_row: dict[int, int] = {}

    try:
        with open(synergy_lookup_path, newline='', encoding='utf-8') as fp:
            reader = csv.reader(fp)
            header = next(reader, None)
            if header:
                for value in header[1:]:
                    try:
                        col_ids.append(int(value))
                    except (TypeError, ValueError):
                        col_ids.append(None)
            for row in reader:
                if not row:
                    continue
                try:
                    hero_id = int(row[0])
                except (TypeError, ValueError):
                    continue
                hero_to_row[hero_id] = len(row_ids)
                row_ids.append(hero_id)
    except FileNotFoundError:
        return row_ids, col_ids, hero_to_row

    return row_ids, col_ids, hero_to_row


def _coerce_ids(values):
    coerced = []
    for value in values:
        try:
            coerced.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid hero id: {value}") from exc
    return coerced


def _hero_entry(hero_id: int) -> dict:
    lookup = get_hero_lookup()
    hero = lookup.get(hero_id)
    entry = {"id": hero_id}
    if hero:
        entry["name"] = hero.get("name", f"Hero {hero_id}")
        if hero.get("api_name"):
            entry["api_name"] = hero["api_name"]
    else:
        entry["name"] = f"Hero {hero_id}"
    return entry


def _format_hero_list(ids):
    return [_hero_entry(hero_id) for hero_id in ids]


def _format_team_payload(ids, win_prob: float) -> dict:
    return {
        "ids": ids,
        "heroes": _format_hero_list(ids),
        "win_probability": win_prob,
        "win_probability_percent": f"{win_prob * 100:.2f}%"
    }


def _format_summary(radiant_prob: float, dire_prob: float) -> str:
    return (
        f"Radiant win chance {radiant_prob * 100:.1f}% vs Dire {dire_prob * 100:.1f}%."
    )


def _format_prediction_html(radiant_prob: float, dire_prob: float) -> str:
    radiant_percent = radiant_prob * 100
    dire_percent = dire_prob * 100
    highlight_class = "radiant" if radiant_percent >= dire_percent else "dire"
    return f"""
    <div style='font-family: "Segoe UI", Arial, sans-serif; max-width: 420px; margin: 0 auto; padding: 1rem; border-radius: 8px; background: #0d1117; color: #ffffff; box-shadow: 0 10px 30px rgba(0,0,0,0.35);'>
        <h3 style='margin-top: 0; text-align: center;'>Win Probability</h3>
        <div style='display: flex; gap: 1rem;'>
            <div style='flex: 1; text-align: center; padding: 0.75rem; border-radius: 6px; background: {"#1f6feb" if highlight_class == "radiant" else "#161b22"};'>
                <div style='font-size: 0.9rem; letter-spacing: 0.05em;'>Radiant</div>
                <div style='font-size: 2rem; font-weight: 600;'>{radiant_percent:.1f}%</div>
            </div>
            <div style='flex: 1; text-align: center; padding: 0.75rem; border-radius: 6px; background: {"#1f6feb" if highlight_class == "dire" else "#161b22"};'>
                <div style='font-size: 0.9rem; letter-spacing: 0.05em;'>Dire</div>
                <div style='font-size: 2rem; font-weight: 600;'>{dire_percent:.1f}%</div>
            </div>
        </div>
        <p style='margin-top: 1rem; font-size: 0.95rem; text-align: center;'>Radiant win chance {radiant_percent:.1f}% vs Dire {dire_percent:.1f}%</p>
    </div>
    """


def _format_recommendations_html(side: str, heroes: list[dict], heading: str = "Suggested Picks") -> str:
    side_title = "Radiant" if side.lower() == "radiant" else "Dire"
    accent_color = "#1f6feb" if side_title == "Radiant" else "#bb3a3a"
    hero_rows = []
    for rank, hero in enumerate(heroes, start=1):
        hero_id = hero.get("id", 0)
        hero_name = hero.get("name", f"Hero {hero_id}")
        hero_img = url_for("static", filename=f"img/avatar-sb/{hero_id}.png")
        hero_rows.append(
            f"""
            <div style='display:flex;align-items:center;gap:0.75rem;padding:0.65rem;border-radius:6px;background:#161b22;border:1px solid #1f2329;'>
                <span style='width:2rem;text-align:center;font-weight:600;color:{accent_color};'>#{rank}</span>
                <img src='{hero_img}' alt='{hero_name}' width='60' height='34' style='border-radius:4px;object-fit:cover;'>
                <div>
                    <div style='font-weight:600;font-size:1rem;'>{hero_name}</div>
                    <div style='font-size:0.8rem;color:#8b949e;'>Hero ID {hero_id}</div>
                </div>
            </div>
            """
        )

    heroes_html = "".join(hero_rows) if hero_rows else "<p>No recommendations available.</p>"
    return f"""
    <div style='font-family:"Segoe UI", Arial, sans-serif;max-width:420px;margin:0 auto;padding:1rem;border-radius:10px;background:#0d1117;color:#ffffff;box-shadow:0 12px 32px rgba(0,0,0,0.4);'>
        <div style='text-align:center;margin-bottom:0.75rem;'>
            <div style='font-size:0.85rem;text-transform:uppercase;letter-spacing:0.08em;color:#8b949e;'>{heading}</div>
            <div style='font-size:1.5rem;font-weight:600;color:{accent_color};'>{side_title}</div>
        </div>
        <div style='display:flex;flex-direction:column;gap:0.6rem;'>
            {heroes_html}
        </div>
    </div>
    """


def _html_or_error(payload: dict, heading: str) -> tuple:
    if "error" in payload:
        return jsonify(payload), 400
    response_html = _format_recommendations_html(
        payload["side"], payload["recommendations"], heading=heading
    )
    return response_html, 200, {"Content-Type": "text/html; charset=utf-8"}


def _get_win_predictor() -> DotaWinPredictor:
    global _win_predictor
    with _predictor_lock:
        if _win_predictor is None:
            heroes = get_heroes()
            n_heroes = max(hero["id"] for hero in heroes) + 1
            counter_lookup = load_lookup_matrix(counter_lookup_path, n_heroes)
            synergy_lookup = load_lookup_matrix(synergy_lookup_path, n_heroes)

            def build_predictor(enable_extra_features: bool) -> DotaWinPredictor:
                return DotaWinPredictor(
                    n_heroes=n_heroes,
                    counter_lookup=counter_lookup if enable_extra_features else None,
                    synergy_lookup=synergy_lookup if enable_extra_features else None,
                )

            predictor = build_predictor(enable_extra_features=True)
            predictor.load_model(model_path)

            if predictor.model.w.shape[0] != predictor.n_features:
                app.logger.warning(
                    "Loaded model has %s weights but predictor expects %s features. Falling back to legacy feature set.",
                    predictor.model.w.shape[0], predictor.n_features,
                )
                predictor = build_predictor(enable_extra_features=False)
                predictor.load_model(model_path)

            _win_predictor = predictor
    return _win_predictor


def _get_recommendations(payload: dict, side: str) -> dict:
    try:
        radiant = _coerce_ids(payload.get("radiant", []))
        dire = _coerce_ids(payload.get("dire", []))
        bans = _coerce_ids(payload.get("bans", []))
    except ValueError as exc:
        return {"error": str(exc)}

    counter_matrix, synergy_matrix, safe_first = _get_pick_assets()
    recommended_ids = recommend_picks(
        current_bans=bans,
        current_side=side,
        current_radiant_picks=radiant,
        current_dire_picks=dire,
        counter_matrix=counter_matrix,
        synergy_matrix=synergy_matrix,
        safe_first_picks=safe_first,
        top_k=5,
        valid_heroes=_valid_hero_ids(),
    )
    recommended_ids = [int(hero_id) for hero_id in recommended_ids]

    return {
        "side": side,
        "recommendations": _format_hero_list(recommended_ids),
        "raw_ids": recommended_ids,
    }


def _get_ban_recommendations(payload: dict, side: str) -> dict:
    try:
        radiant = _coerce_ids(payload.get("radiant", []))
        dire = _coerce_ids(payload.get("dire", []))
        bans = _coerce_ids(payload.get("bans", []))
    except ValueError as exc:
        return {"error": str(exc)}

    counter_matrix, synergy_matrix, safe_first = _get_pick_assets()
    recommended_ids = recommend_bans(
        current_bans=bans,
        current_side=side,
        current_radiant_picks=radiant,
        current_dire_picks=dire,
        counter_matrix=counter_matrix,
        synergy_matrix=synergy_matrix,
        safe_first_picks=safe_first,
        top_k=5,
        valid_heroes=_valid_hero_ids(),
    )
    recommended_ids = [int(hero_id) for hero_id in recommended_ids]

    return {
        "side": side,
        "recommendations": _format_hero_list(recommended_ids),
        "raw_ids": recommended_ids,
    }


if __name__ == '__main__':
    app.jinja_env.cache = {}
    app.run(debug=debug_mode, host='127.0.0.1', port=frontend_port)
