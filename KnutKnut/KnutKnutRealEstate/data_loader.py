import json
import math
from typing import Dict, List, Tuple

import numpy as np


DATA_DIR = "KnutKnut/KnutKnutRealEstate/data"


def load_jsonl(filepath: str) -> List[dict]:
    """Load a .jsonl file as a list of dicts."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _parse_rooms(value) -> float:
    """Convert values like "3 rooms" or "" to a numeric room count."""
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    if not value:
        return math.nan
    # Expect formats like "3 rooms" or "1 rooms"
    parts = value.split()
    try:
        return float(parts[0])
    except (ValueError, IndexError):
        return math.nan


def _month_to_angle(month_name: str) -> float:
    """Map month name to angle in [0, 2*pi)."""
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
    if month_name not in months:
        return 0.0
    idx = months.index(month_name)
    return 2.0 * math.pi * idx / 12.0


def build_dataset(
    agents_path: str = f"{DATA_DIR}/agents.jsonl",
    districts_path: str = f"{DATA_DIR}/districts.jsonl",
    houses_path: str = f"{DATA_DIR}/houses.jsonl",
    schools_path: str = f"{DATA_DIR}/schools.jsonl",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load raw jsonl data, join tables, clean and encode features.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Target log-price.
    feature_names : List[str]
        Names for each column in X.
    """

    agents = load_jsonl(agents_path)
    districts = load_jsonl(districts_path)
    houses = load_jsonl(houses_path)
    schools = load_jsonl(schools_path)

    agents_dict: Dict[str, dict] = {a["agent_id"]: a for a in agents}
    districts_dict: Dict[str, dict] = {d["id"]: d for d in districts}
    schools_dict: Dict[str, dict] = {s["id"]: s for s in schools}

    # First pass: build vocabularies for categorical variables
    advertisement_vals = set()
    color_vals = set()
    fireplace_vals = set()
    parking_vals = set()
    sold_vals = set()
    month_vals = set()

    for h in houses:
        advertisement_vals.add(h.get("advertisement", "unknown"))
        color_vals.add(h.get("color", "unknown"))
        fireplace_vals.add(h.get("fireplace", "unknown"))
        parking_vals.add(h.get("parking", "unknown"))
        sold_vals.add(h.get("sold", "unknown"))
        month_vals.add(h.get("sold_in_month", "unknown"))

    # Sort to get deterministic order
    advertisement_vals = sorted(advertisement_vals)
    color_vals = sorted(color_vals)
    fireplace_vals = sorted(fireplace_vals)
    parking_vals = sorted(parking_vals)
    sold_vals = sorted(sold_vals)
    month_vals = sorted(month_vals)

    # Build index maps
    adv_index = {v: i for i, v in enumerate(advertisement_vals)}
    color_index = {v: i for i, v in enumerate(color_vals)}
    fireplace_index = {v: i for i, v in enumerate(fireplace_vals)}
    parking_index = {v: i for i, v in enumerate(parking_vals)}
    sold_index = {v: i for i, v in enumerate(sold_vals)}

    feature_names: List[str] = []

    # Numeric features (per house)
    numeric_keys = [
        "size",
        "bathrooms",
        "kitchens",
        "external_storage_m2",
        "lot_w",
        "storage_rating",
        "sun_factor",
        "condition_rating",
        "days_on_marked",
        "year",
    ]

    feature_names.extend(numeric_keys)

    # Engineered numeric: years_since_remodel, remodeled_flag
    feature_names.extend(["years_since_remodel", "remodeled_flag"])

    # District-level numeric features
    feature_names.extend(["district_crime_rating", "district_public_transport_rating"])

    # School-level numeric features
    feature_names.extend(["school_built_year", "school_rating", "school_capacity"])

    # Parsed rooms
    feature_names.append("rooms_count")

    # Sold flag as numeric
    feature_names.append("sold_flag")

    # Cyclical month encoding
    feature_names.extend(["month_sin", "month_cos"])

    # Categorical one-hot parts (except month, modeled as sin/cos)
    feature_names.extend([f"adv_{v}" for v in advertisement_vals])
    feature_names.extend([f"color_{v}" for v in color_vals])
    feature_names.extend([f"fireplace_{v}" for v in fireplace_vals])
    feature_names.extend([f"parking_{v}" for v in parking_vals])
    feature_names.extend([f"sold_cat_{v}" for v in sold_vals])

    n_features = len(feature_names)

    X_rows: List[np.ndarray] = []
    y_vals: List[float] = []

    current_year = max(h.get("year", 0) for h in houses if isinstance(h.get("year", 0), (int, float)))

    for h in houses:
        price = h.get("price")
        if price is None or price <= 0:
            # Drop clearly invalid price rows
            continue

        row = np.zeros(n_features, dtype=float)

        # Numeric basic features
        for idx, key in enumerate(numeric_keys):
            value = h.get(key)
            if isinstance(value, (int, float)):
                row[idx] = float(value)
            else:
                row[idx] = math.nan

        offset = len(numeric_keys)

        # years_since_remodel and remodeled_flag
        remodeled = h.get("remodeled", -1)
        years_since_remodel = math.nan
        remodeled_flag = 0.0
        year_val = h.get("year")
        if isinstance(remodeled, (int, float)) and remodeled > 0 and isinstance(year_val, (int, float)):
            years_since_remodel = year_val - remodeled
            remodeled_flag = 1.0
        row[offset] = years_since_remodel
        row[offset + 1] = remodeled_flag
        offset += 2

        # District features
        district = districts_dict.get(h.get("district_id"))
        crime_rating = district.get("crime_rating") if district is not None else math.nan
        transport_rating = district.get("public_transport_rating") if district is not None else math.nan
        row[offset] = float(crime_rating) if isinstance(crime_rating, (int, float)) else math.nan
        row[offset + 1] = float(transport_rating) if isinstance(transport_rating, (int, float)) else math.nan
        offset += 2

        # School features
        school = schools_dict.get(h.get("school_id"))
        built_year = school.get("built_year") if school is not None else math.nan
        school_rating = school.get("rating") if school is not None else math.nan
        capacity = school.get("capacity") if school is not None else math.nan
        row[offset] = float(built_year) if isinstance(built_year, (int, float)) else math.nan
        row[offset + 1] = float(school_rating) if isinstance(school_rating, (int, float)) else math.nan
        row[offset + 2] = float(capacity) if isinstance(capacity, (int, float)) else math.nan
        offset += 3

        # Rooms
        rooms_count = _parse_rooms(h.get("rooms"))
        row[offset] = rooms_count
        offset += 1

        # Sold flag numeric
        sold_str = h.get("sold", "unknown")
        sold_flag = 1.0 if str(sold_str).lower() == "yes" else 0.0
        row[offset] = sold_flag
        offset += 1

        # Month cyclic encoding
        month_name = h.get("sold_in_month", "January")
        angle = _month_to_angle(month_name)
        row[offset] = math.sin(angle)
        row[offset + 1] = math.cos(angle)
        offset += 2

        # Categorical one-hots
        adv_val = h.get("advertisement", "unknown")
        adv_idx = adv_index.get(adv_val)
        if adv_idx is not None:
            row[offset + adv_idx] = 1.0
        offset += len(advertisement_vals)

        color_val = h.get("color", "unknown")
        color_idx = color_index.get(color_val)
        if color_idx is not None:
            row[offset + color_idx] = 1.0
        offset += len(color_vals)

        fire_val = h.get("fireplace", "unknown")
        fire_idx = fireplace_index.get(fire_val)
        if fire_idx is not None:
            row[offset + fire_idx] = 1.0
        offset += len(fireplace_vals)

        park_val = h.get("parking", "unknown")
        park_idx = parking_index.get(park_val)
        if park_idx is not None:
            row[offset + park_idx] = 1.0
        offset += len(parking_vals)

        sold_cat_val = h.get("sold", "unknown")
        sold_cat_idx = sold_index.get(sold_cat_val)
        if sold_cat_idx is not None:
            row[offset + sold_cat_idx] = 1.0
        offset += len(sold_vals)

        X_rows.append(row)
        y_vals.append(math.log(float(price)))

    X = np.vstack(X_rows)
    y = np.asarray(y_vals, dtype=float)
    return X, y, feature_names


if __name__ == "__main__":
    X, y, feature_names = build_dataset()
    print("Shapes:", X.shape, y.shape)
    print("First 5 features:", feature_names)
