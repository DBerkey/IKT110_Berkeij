import pickle
from pathlib import Path

import numpy as np
import dash
from dash import dcc
from dash import html as html
from dash.dependencies import Input, Output


# Load trained price model artifacts
ARTIFACTS_PATH = Path(__file__).with_name("price_model.pkl")
with ARTIFACTS_PATH.open("rb") as f:
    _artifacts = pickle.load(f)

W = _artifacts["weights"]
B = _artifacts["bias"]
MEAN = _artifacts["mean"]
STD = _artifacts["std"]
FEATURE_NAMES = _artifacts["feature_names"]


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


app = dash.Dash(__name__)
app.layout = html.Div([
    html.H3(children="KKD - Real Estate Dashboard"),

    html.Div([
        html.H4(children="Year built:"),
        dcc.Input(id="year", value="1990", type="number"),
    ]),

    html.Div([
        html.H4(children="Remodeled year (0 if never):"),
        dcc.Input(id="remodeled", value="2015", type="number"),
    ]),

    html.Div([
        html.H4(children="Color:"),
        dcc.Dropdown(["blue", "red", "white", "gray", "green", "black"], "blue", id="color"),
    ]),

    html.Div([
        html.H4(children="Put to market in:"),
        dcc.Dropdown(["jan", "feb", "march", "april", "november"], "november", id="month-to-marked"),
    ]),

    html.H4(children="Predicted Price:"),
    html.Div([
        html.Pre(id="output-price")
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
    return f"{price:,.0f} NOK"


if __name__ == "__main__":
    app.run(debug=True)






    





