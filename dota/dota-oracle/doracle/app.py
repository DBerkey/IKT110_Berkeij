# ----------------------------------------------------------------------------#
# Imports
# ----------------------------------------------------------------------------#

import getpass
import json
import os
import sys
from threading import Lock
import requests as req
from flask import Flask, render_template, request, jsonify, g, redirect, url_for, send_from_directory
from functools import lru_cache

# The ML model files
from doracle.model import HeroStats

# Make sure the project root (which contains the training code) is importable.
package_directory = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(package_directory, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from dota.dotaWinPredition.winPredictor import DotaWinPredictor, load_lookup_matrix

# ----------------------------------------------------------------------------#
# Configs
# ----------------------------------------------------------------------------#

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
    r = request.get_json()
    r["todo"] = "Please rename the button from Suggest1 and implement some functionality here.",

    return jsonify(r)

# <button 2>
@app.route("/suggest2", methods=["POST"])
def suggest2():
    r = request.get_json()
    r["todo"] = "Please rename the button from Suggest2 and implement some functionality here.",

    return jsonify(r)


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

    hero_model = get_hero_stat_model()
    win_rate = hero_model.get_winrate(heroid)
    pick_rate = hero_model.get_pickrate(heroid)
    best_paired_with = hero_model.get_best_paired_with_hero(heroid)

    hero_lookup = get_hero_lookup()
    hero_name = hero_lookup.get(heroid, {}).get("name", f"Hero {heroid}")
    best_paired_with_name = hero_lookup.get(best_paired_with, {}).get("name", f"Hero {best_paired_with}")

    hero_stats = {
        "todo": "please implement this feature instead of the random stats that are not used.",
        "hero": hero_name,
        "win_rate": win_rate,
        "pick_rate": pick_rate,
        "best_paired_with": best_paired_with_name
    }

    print("hero_stats:", hero_stats)

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
def get_hero_stat_model():
    path_to_model = "/home/dota_oracle_user/models/herostat.pkl"

    hero_stat_model = HeroStats.load(path_to_model)
    return hero_stat_model


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


if __name__ == '__main__':
    app.jinja_env.cache = {}
    app.run(debug=debug_mode, host='127.0.0.1', port=frontend_port)
