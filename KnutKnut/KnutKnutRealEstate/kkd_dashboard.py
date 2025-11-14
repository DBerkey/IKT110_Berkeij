import pickle
from pathlib import Path

import numpy as np
import dash
from dash import dcc
from dash import html as html
from dash.dependencies import Input, Output

from data_loader import load_jsonl, DATA_DIR


# Load trained price model artifacts
ARTIFACTS_PATH = Path(__file__).with_name("price_model.pkl")
with ARTIFACTS_PATH.open("rb") as f:
    _artifacts = pickle.load(f)

W = _artifacts["weights"]
B = _artifacts["bias"]
MEAN = _artifacts["mean"]
STD = _artifacts["std"]
FEATURE_NAMES = _artifacts["feature_names"]

# Optional classification model for sale probability
CLS_W = _artifacts.get("cls_weights")
CLS_B = _artifacts.get("cls_bias")
CLS_THRESHOLD_DAYS = _artifacts.get("cls_threshold_days", 30.0)


def load_raw_data_for_analysis():
    houses = load_jsonl(f"{DATA_DIR}/houses.jsonl")
    agents = load_jsonl(f"{DATA_DIR}/agents.jsonl")
    agents_dict = {a["agent_id"]: a["name"] for a in agents}
    return houses, agents_dict


def compute_best_agents(houses, agents_dict, top_n=5):
    sold = [h for h in houses if str(h.get("sold", "")).lower() == "yes"]
    per_agent = {}
    for h in sold:
        aid = h.get("agent_id")
        if not aid:
            continue
        rec = per_agent.setdefault(aid, {"sum_days": 0.0, "sum_price": 0.0, "count": 0})
        rec["sum_days"] += float(h.get("days_on_marked", 0.0))
        rec["sum_price"] += float(h.get("price", 0.0))
        rec["count"] += 1

    rows = []
    for aid, rec in per_agent.items():
        if rec["count"] == 0:
            continue
        rows.append({
            "agent_name": agents_dict.get(aid, aid),
            "avg_days_on_marked": rec["sum_days"] / rec["count"],
            "avg_price": rec["sum_price"] / rec["count"],
            "n_sales": rec["count"],
        })

    rows.sort(key=lambda r: (r["avg_days_on_marked"], -r["avg_price"]))
    return rows[:top_n]


def compute_ad_package_stats(houses):
    buckets = {}
    for h in houses:
        adv = h.get("advertisement", "unknown")
        price = h.get("price")
        if price is None:
            continue
        rec = buckets.setdefault(adv, {"sum_price": 0.0, "count": 0})
        rec["sum_price"] += float(price)
        rec["count"] += 1

    rows = []
    for adv, rec in buckets.items():
        if rec["count"] == 0:
            continue
        rows.append({
            "package": adv,
            "avg_price": rec["sum_price"] / rec["count"],
            "n": rec["count"],
        })

    rows.sort(key=lambda r: r["avg_price"])
    return rows


def compute_data_quality(houses):
    total = len(houses)
    missing_rooms = sum(1 for h in houses if not h.get("rooms"))
    remodeled_neg1 = sum(1 for h in houses if h.get("remodeled", 0) == -1)
    color_unknown = sum(1 for h in houses if h.get("color") == "unknown")
    missing_school = sum(1 for h in houses if not h.get("school_id"))
    missing_district = sum(1 for h in houses if not h.get("district_id"))

    return {
        "total": total,
        "missing_rooms": missing_rooms,
        "remodeled_eq_minus1": remodeled_neg1,
        "color_unknown": color_unknown,
        "missing_school_id": missing_school,
        "missing_district_id": missing_district,
    }

def _build_feature_vector(year: int, remodeled: int, color: str, month_str: str) -> np.ndarray:
    """Construct a single feature vector matching `build_dataset` layout.

    For simplicity this uses sensible defaults for fields not exposed in the UI
    (e.g., size, bathrooms). The important part is that indices line up with
    FEATURE_NAMES used during training.
    """

    x = np.zeros(len(FEATURE_NAMES), dtype=float)

    # Helper to set if feature exists
    def set_feat(name: str, value: float):
        if name in FEATURE_NAMES:
            idx = FEATURE_NAMES.index(name)
            x[idx] = value

    # Basic numeric defaults
    set_feat("size", 80.0)
    set_feat("bathrooms", 1.0)
    set_feat("kitchens", 1.0)
    set_feat("external_storage_m2", 5.0)
    set_feat("lot_w", 20.0)
    set_feat("storage_rating", 5.0)
    set_feat("sun_factor", 0.6)
    set_feat("condition_rating", 6.0)
    set_feat("days_on_marked", 10.0)
    set_feat("year", float(year))

    # Remodel features
    years_since_remodel = float(year) - float(remodeled) if remodeled > 0 else 0.0
    set_feat("years_since_remodel", years_since_remodel)
    set_feat("remodeled_flag", 1.0 if remodeled > 0 else 0.0)

    # District and school defaults
    set_feat("district_crime_rating", 3.0)
    set_feat("district_public_transport_rating", 3.0)
    set_feat("school_built_year", 1980.0)
    set_feat("school_rating", 2.5)
    set_feat("school_capacity", 50.0)

    # Rooms
    set_feat("rooms_count", 3.0)

    # Sold flag: we assume we want price for a house that will sell
    set_feat("sold_flag", 1.0)

    # Month sin/cos: map short month to full name
    month_map = {
        "jan": "January",
        "feb": "February",
        "march": "March",
        "april": "April",
        "november": "November",
    }
    full_month = month_map.get(month_str, "January")
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    if full_month in months:
        idx_month = months.index(full_month)
        angle = 2.0 * np.pi * idx_month / 12.0
        set_feat("month_sin", float(np.sin(angle)))
        set_feat("month_cos", float(np.cos(angle)))

    # Color one-hot
    color_key = f"color_{color}"
    set_feat(color_key, 1.0)

    # Reasonable defaults for categorical advertisement / fireplace / parking / sold_cat
    set_feat("adv_regular", 1.0)
    set_feat("fireplace_no", 1.0)
    set_feat("parking_yes", 1.0)
    set_feat("sold_cat_yes", 1.0)

    return x


def _predict_price(year: int, remodeled: int, color: str, month_str: str) -> float:
    x = _build_feature_vector(year, remodeled, color, month_str)
    # Scale
    x_scaled = (x - MEAN) / STD
    y_log = float(x_scaled @ W + B)
    return float(np.exp(y_log))


def _predict_sale_probability(year: int, remodeled: int, color: str, month_str: str):
    """Predict probability that the house is sold within CLS_THRESHOLD_DAYS.

    Returns None if the classification model is not available in artifacts.
    """

    if CLS_W is None or CLS_B is None:
        return None

    x = _build_feature_vector(year, remodeled, color, month_str)
    x_scaled = (x - MEAN) / STD
    logit = float(x_scaled @ CLS_W + CLS_B)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return prob


app = dash.Dash(__name__)
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3("KKD - Real Estate Dashboard"),

    dcc.Tabs(id="tabs", value="tab-price", children=[
        dcc.Tab(label="Price Prediction", value="tab-price", children=[
            html.Div([
                html.H4("Year built:"),
                dcc.Input(id="year", value="1990", type="number"),
            ]),
            html.Div([
                html.H4("Remodeled year (0 if never):"),
                dcc.Input(id="remodeled", value="2015", type="number"),
            ]),
            html.Div([
                html.H4("Color:"),
                dcc.Dropdown(
                    ["blue", "red", "white", "gray", "green", "black"],
                    "blue",
                    id="color",
                ),
            ]),
            html.Div([
                html.H4("Put to market in:"),
                dcc.Dropdown(
                    ["jan", "feb", "march", "april", "november"],
                    "november",
                    id="month-to-marked",
                ),
            ]),
            html.H4("Predicted Price:"),
            html.Div([html.Pre(id="output-price")]),
        ]),

        dcc.Tab(label="Data Analysis", value="tab-analysis", children=[
            html.H4("Best Agents (by days on market)"),
            html.Pre(id="best-agents"),

            html.H4("Ad Package Pricing (average sale price)"),
            html.Pre(id="ad-packages"),

            html.H4("Data Quality / Missing Data"),
            html.Pre(id="data-quality"),
        ]),
    ]),
])

@app.callback(
    Output("output-price", "children"),
    Input("year", "value"),
    Input("remodeled", "value"),
    Input("color", "value"),
    Input("month-to-marked", "value"),
)
def predict_price(year, remodeled, house_color, month_to_marked):
    try:
        y = int(year)
        ry = int(remodeled)
    except (TypeError, ValueError):
        return "Please enter valid numeric years."

    price = _predict_price(y, ry, house_color, month_to_marked)
    prob = _predict_sale_probability(y, ry, house_color, month_to_marked)

    if prob is None:
        return f"{price:,.0f} NOK"

    return (
        f"{price:,.0f} NOK\n"
        f"Probability sold within {int(CLS_THRESHOLD_DAYS)} days: {prob * 100:.1f}%"
    )

@app.callback(
    Output("best-agents", "children"),
    Output("ad-packages", "children"),
    Output("data-quality", "children"),
    Input("tabs", "value"),
)
def update_analysis_tab(active_tab):
    if active_tab != "tab-analysis":
        return "", "", ""

    houses, agents_dict = load_raw_data_for_analysis()

    best_agents = compute_best_agents(houses, agents_dict)
    ad_stats = compute_ad_package_stats(houses)
    quality = compute_data_quality(houses)

    best_agents_lines = [
        f"- {r['agent_name']}: "
        f"{r['avg_days_on_marked']:.1f} days on market, "
        f"{r['n_sales']} sales, "
        f"avg price {r['avg_price']:,.0f} NOK"
        for r in best_agents
    ]
    best_agents_text = "\n".join(best_agents_lines) or "No data."

    ad_lines = [
        f"- {r['package']}: avg price {r['avg_price']:,.0f} NOK over {r['n']} houses"
        for r in ad_stats
    ]
    ad_text = "\n".join(ad_lines) or "No data."

    q = quality
    quality_text = (
        f"Total houses: {q['total']}\n"
        f"Missing rooms: {q['missing_rooms']}\n"
        f"Remodeled == -1: {q['remodeled_eq_minus1']}\n"
        f'Color == \"unknown\": {q["color_unknown"]}'
    )

    return best_agents_text, ad_text, quality_text

if __name__ == "__main__":
    app.run(debug=True)






    





