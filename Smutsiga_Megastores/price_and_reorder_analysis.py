"""
price_and_reorder_analysis.py

Reads a JSONL file with sales/inventory records and produces per-product metrics,
price-change suggestions, and reorder quantity recommendations.

Flexible input: the script will look for common keys. If timestamps exist, it will
compute daily rates; otherwise it will use totals.

Outputs:
 - products_summary.csv  : per-product computed metrics and suggested price changes & reorder qty
 - suggestions.txt       : human-readable summary with top candidates for price change and reorders

Usage (example):
 python price_and_reorder_analysis.py --input sales.jsonl --output-dir analysis_out --lead-time 14 --z 1.65 --desired-margin 0.30

Assumptions (if missing fields):
 - `price` and `quantity` are assumed names for sale price and quantity sold; also tries `unit_price`, `qty`, `quantity_sold`.
 - `cost` or `unit_cost` may be present; if absent, margin-based suggestions are skipped.
 - `inventory` or `stock` used for current stock; otherwise current stock is unknown and reorder will be based on no stock.

"""
from __future__ import annotations
import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, List, Optional

try:
    import numpy as np
    import pandas as pd
except Exception:
    raise SystemExit("This script requires pandas and numpy. Install them (pip install pandas numpy).")

COMMON_PRICE_KEYS = ["price", "unit_price", "sale_price"]
COMMON_QTY_KEYS = ["quantity", "qty", "quantity_sold", "units_sold"]
COMMON_COST_KEYS = ["cost", "unit_cost", "unit_cost_price"]
COMMON_INV_KEYS = ["inventory", "stock", "on_hand", "quantity_on_hand"]
COMMON_TIMESTAMP_KEYS = ["timestamp", "time", "date", "sold_at"]
COMMON_PRODUCT_ID_KEYS = ["product_id", "id", "sku"]
COMMON_PRODUCT_NAME_KEYS = ["product_name", "name", "title"]


def find_key(d: dict, candidates: List[str]):
    for k in candidates:
        if k in d:
            return k
    return None


def parse_timestamp(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        # epoch
        try:
            return datetime.fromtimestamp(v)
        except Exception:
            return None
    if isinstance(v, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(v, fmt)
            except Exception:
                continue
    return None


class ProductStats:
    def __init__(self, product_id: str):
        self.product_id = product_id
        self.names = set()
        self.records = []  # each record: dict with price, qty, cost, timestamp, inventory

    def add(self, rec: dict):
        self.records.append(rec)
        if rec.get("name"):
            self.names.add(rec.get("name"))

    def compute(self):
        # build dataframe
        df = pd.DataFrame(self.records)
        # ensure numeric
        for col in ["price", "qty", "cost", "inventory"]:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        metrics = {}
        metrics["product_id"] = self.product_id
        metrics["product_name"] = ", ".join(sorted(self.names)) if self.names else ""
        metrics["total_qty_sold"] = int(df["qty"].sum()) if "qty" in df else 0
        metrics["total_revenue"] = float((df.get("price", 0) * df.get("qty", 0)).sum()) if ("price" in df and "qty" in df) else 0.0
        metrics["avg_price"] = float(df["price"].mean()) if "price" in df and not df["price"].dropna().empty else float("nan")
        metrics["std_price"] = float(df["price"].std(ddof=0)) if "price" in df and len(df["price"].dropna()) > 1 else 0.0
        metrics["avg_cost"] = float(df["cost"].mean()) if "cost" in df and not df["cost"].dropna().empty else float("nan")
        metrics["last_inventory"] = int(df["inventory"].dropna().iloc[-1]) if "inventory" in df and not df["inventory"].dropna().empty else None

        # time-based daily sales
        if "timestamp" in df:
            df_ts = df.dropna(subset=["timestamp"]).copy()
            if not df_ts.empty:
                first = df_ts["timestamp"].min()
                last = df_ts["timestamp"].max()
                days = max(1, (last - first).days + 1)
                metrics["days_of_data"] = days
                total_qty = int(df_ts.get("qty", 0).sum()) if "qty" in df_ts else 0
                metrics["avg_daily_sales"] = total_qty / days
                # compute daily totals to get std
                df_ts["date"] = df_ts["timestamp"].dt.normalize()
                daily = df_ts.groupby("date").agg({"qty": "sum"}).reindex(pd.date_range(first.normalize(), last.normalize(), freq="D"), fill_value=0)
                metrics["std_daily_sales"] = float(daily["qty"].std(ddof=0)) if len(daily) > 1 else 0.0
            else:
                metrics["days_of_data"] = 0
                metrics["avg_daily_sales"] = 0.0
                metrics["std_daily_sales"] = 0.0
        else:
            metrics["days_of_data"] = 0
            # If qty exists, report total as "avg_daily_sales" (no time dimension available)
            if "qty" in df:
                # total qty (since we don't have days), keep as-is but cast to float
                try:
                    total_qty = float(df["qty"].sum())
                except Exception:
                    total_qty = float(pd.to_numeric(df["qty"], errors="coerce").fillna(0).sum())
                metrics["avg_daily_sales"] = total_qty
                metrics["std_daily_sales"] = float(df["qty"].std(ddof=0)) if len(df["qty"].dropna()) > 1 else 0.0
            else:
                metrics["avg_daily_sales"] = 0.0
                metrics["std_daily_sales"] = 0.0

        # elasticity estimate if price varies
        elasticity = None
        if "price" in df and "qty" in df and len(df.dropna(subset=["price", "qty"])) > 3:
            tmp = df.dropna(subset=["price", "qty"]).copy()
            # require variation
            if tmp["price"].nunique() > 1 and tmp["qty"].nunique() > 1 and (tmp["price"] > 0).all() and (tmp["qty"] > 0).all():
                # fit log-log linear regression for elasticity
                x = np.log(tmp["price"].values)
                y = np.log(tmp["qty"].values)
                A = np.vstack([x, np.ones_like(x)]).T
                try:
                    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                    elasticity = float(m)
                except Exception:
                    elasticity = None
        metrics["price_elasticity_estimate"] = elasticity

        # margin
        if not math.isnan(metrics.get("avg_price", float("nan"))) and not math.isnan(metrics.get("avg_cost", float("nan"))):
            if metrics["avg_price"]:
                metrics["avg_margin"] = (metrics["avg_price"] - metrics["avg_cost"]) / metrics["avg_price"]
            else:
                metrics["avg_margin"] = float("nan")
        else:
            metrics["avg_margin"] = float("nan")

        return metrics


def suggest_price_and_reorder(metrics: dict, lead_time_days: int, z: float, desired_margin: float, price_change_limit: float = 0.20):
    """Return suggestion dict with keys: price_action, price_change_pct, new_price, reorder_qty, reorder_reason."""
    s = {
        "product_id": metrics.get("product_id"),
        "price_action": "no_change",
        "price_change_pct": 0.0,
        "price_reason": "",
        "new_price": None,
        "reorder_qty": 0,
        "reorder_reason": "",
    }

    avg_price = metrics.get("avg_price")
    avg_cost = metrics.get("avg_cost")
    avg_daily = metrics.get("avg_daily_sales", 0.0)
    std_daily = metrics.get("std_daily_sales", 0.0)
    last_inv = metrics.get("last_inventory")
    elasticity = metrics.get("price_elasticity_estimate")

    # REORDER calculation
    # safety_stock = z * std_daily * sqrt(lead_time)
    safety_stock = z * std_daily * math.sqrt(max(1, lead_time_days))
    target_stock = (lead_time_days * avg_daily) + safety_stock
    if last_inv is None:
        # unknown inventory; recommend ordering for 30 days
        reorder_qty = int(max(0, math.ceil((lead_time_days + 30) * avg_daily)))
        s["reorder_reason"] = "no_inventory_info: order to cover lead_time + 30 days"
    else:
        reorder_qty = int(max(0, math.ceil(target_stock - last_inv)))
        if reorder_qty <= 0:
            s["reorder_reason"] = "stock_ok"
        else:
            s["reorder_reason"] = f"target_stock={target_stock:.1f}, last_inventory={last_inv}"
    s["reorder_qty"] = reorder_qty

    # PRICE suggestion rules (simple heuristics):
    # - If avg_margin available and < desired_margin and demand is decent (avg_daily > 0.5), try to increase price up to price_change_limit
    # - If inventory too high (days_of_stock large) and low demand, reduce price up to price_change_limit
    # - If elasticity known, use it to estimate price change needed to reach a sales uplift or margin
    days = metrics.get("days_of_data", 0)

    # compute days_of_stock if last_inv available
    days_of_stock = None
    if last_inv is not None and avg_daily > 0:
        days_of_stock = last_inv / avg_daily

    # Case A: low margin and reasonable demand -> increase price
    if not math.isnan(metrics.get("avg_margin", float("nan"))) and avg_cost and not math.isnan(float(avg_cost)):
        margin = metrics.get("avg_margin")
        if not math.isnan(margin) and margin < desired_margin and avg_daily >= 0.5:
            # increase price to reach desired margin if possible
            if avg_cost > 0:
                target_price = avg_cost / (1 - desired_margin)
                pct = (target_price - avg_price) / avg_price if avg_price and not math.isnan(avg_price) else 0.0
                # cap to price_change_limit
                pct_capped = max(-price_change_limit, min(price_change_limit, pct))
                new_price = avg_price * (1 + pct_capped)
                s.update({"price_action": "increase", "price_change_pct": float(pct_capped), "new_price": float(new_price), "price_reason": f"increase_to_target_margin {desired_margin:.2f}"})

    # Case B: too much stock relative to sales -> discount
    if days_of_stock is not None and days_of_stock > max(60, lead_time_days * 2) and avg_daily < 5:
        # reduce price to move stock
        pct = -min(price_change_limit, 0.20)
        new_price = avg_price * (1 + pct) if not math.isnan(avg_price) else None
        if s["price_action"] == "no_change":
            s.update({"price_action": "decrease", "price_change_pct": float(pct), "new_price": float(new_price), "price_reason": f"slow_turnover_days_of_stock={days_of_stock:.1f}"})

    # Case C: use elasticity if present to recommend price to increase revenue when demand is inelastic (|e| < 1)
    if elasticity is not None and avg_price and avg_daily:
        # revenue = price * qty; elasticities negative typically
        e = elasticity
        # If inelastic (abs(e) < 1), raising price increases revenue
        if abs(e) < 0.8 and s["price_action"] == "no_change":
            pct = min(price_change_limit, 0.05 + (0.05 * (0.8 - abs(e))))
            new_price = avg_price * (1 + pct)
            s.update({"price_action": "increase", "price_change_pct": float(pct), "new_price": float(new_price), "price_reason": f"inelastic_e={e:.2f}"})
        elif e > 0 and s["price_action"] == "no_change":
            # weird positive elasticity (increase in price increases qty) => small increase
            pct = min(price_change_limit, 0.05)
            new_price = avg_price * (1 + pct)
            s.update({"price_action": "increase", "price_change_pct": float(pct), "new_price": float(new_price), "price_reason": "positive_elasticity"})

    # final fallback: no change
    if s["new_price"] is None and not math.isnan(avg_price):
        s["new_price"] = float(avg_price)

    return s


def main():
    p = argparse.ArgumentParser(description="Analyze JSONL sales/inventory file and recommend price and reorder quantities.")
    p.add_argument("--input", "-i", required=True, help="Path to JSONL file containing records (one JSON object per line).")
    p.add_argument("--output-dir", "-o", required=False, default="analysis_output/price_analysis", help="Directory to write results.")
    p.add_argument("--lead-time", type=int, default=14, help="Lead time in days for reorders.")
    p.add_argument("--z", type=float, default=1.65, help="Safety stock z-score multiplier (e.g. 1.65 for 95% service level).")
    p.add_argument("--desired-margin", type=float, default=0.30, help="Desired target gross margin (fraction, e.g. 0.30).")
    p.add_argument("--price-change-limit", type=float, default=0.20, help="Max absolute fraction to adjust price in one step.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read JSONL
    products: Dict[str, ProductStats] = {}
    line_no = 0
    with open(args.input, "r", encoding="utf-8") as fh:
        for line in fh:
            line_no += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"warning: skipping invalid json on line {line_no}: {e}")
                continue
            # pick product id
            pid_key = find_key(obj, COMMON_PRODUCT_ID_KEYS)
            if pid_key:
                pid = str(obj.get(pid_key))
            else:
                # fallback to product_name if present
                name_key = find_key(obj, COMMON_PRODUCT_NAME_KEYS)
                pid = obj.get(name_key) or f"unknown_{line_no}"
                pid = str(pid)

            if pid not in products:
                products[pid] = ProductStats(pid)

            # build normalized record
            rec = {}
            name_key = find_key(obj, COMMON_PRODUCT_NAME_KEYS)
            if name_key:
                rec["name"] = obj.get(name_key)

            price_key = find_key(obj, COMMON_PRICE_KEYS)
            if price_key:
                rec["price"] = obj.get(price_key)

            qty_key = find_key(obj, COMMON_QTY_KEYS)
            if qty_key:
                rec["qty"] = obj.get(qty_key)

            cost_key = find_key(obj, COMMON_COST_KEYS)
            if cost_key:
                rec["cost"] = obj.get(cost_key)

            inv_key = find_key(obj, COMMON_INV_KEYS)
            if inv_key:
                rec["inventory"] = obj.get(inv_key)

            ts_key = find_key(obj, COMMON_TIMESTAMP_KEYS)
            if ts_key:
                rec_ts = parse_timestamp(obj.get(ts_key))
                if rec_ts:
                    rec["timestamp"] = rec_ts

            products[pid].add(rec)

    # compute metrics and suggestions
    rows = []
    suggestions = []
    for pid, ps in products.items():
        metrics = ps.compute()
        suggestion = suggest_price_and_reorder(metrics, args.lead_time, args.z, args.desired_margin, args.price_change_limit)
        # merge metric and suggestion for CSV
        row = {**metrics, **suggestion}
        rows.append(row)
        suggestions.append((metrics, suggestion))

    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "products_summary.csv")
    df_out.to_csv(csv_path, index=False)

    # write human-readable suggestions
    txt_path = os.path.join(args.output_dir, "suggestions.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("Price & Reorder Suggestions\n")
        fh.write("===========================\n\n")
        # top price changes
        fh.write("Price change candidates (non-no_change), sorted by impact:\n\n")
        df_price = df_out[df_out["price_action"] != "no_change"].copy()
        if not df_price.empty:
            df_price = df_price.sort_values(by="price_change_pct", ascending=False)
            for _, r in df_price.iterrows():
                fh.write(f"Product: {r.get('product_name') or r.get('product_id')} (id={r.get('product_id')})\n")
                fh.write(f"  Action: {r.get('price_action')}, change: {r.get('price_change_pct'):.3f}, new_price: {r.get('new_price'):.2f}\n")
                fh.write(f"  Reason: {r.get('price_reason')}\n")
                fh.write('\n')
        else:
            fh.write("  No price change suggestions.\n\n")

        fh.write("Reorder recommendations:\n\n")
        df_reorder = df_out[df_out["reorder_qty"] > 0].copy()
        if not df_reorder.empty:
            df_reorder = df_reorder.sort_values(by="reorder_qty", ascending=False)
            for _, r in df_reorder.iterrows():
                fh.write(f"Product: {r.get('product_name') or r.get('product_id')} (id={r.get('product_id')})\n")
                fh.write(f"  Reorder qty: {int(r.get('reorder_qty'))}, reason: {r.get('reorder_reason')}\n")
                fh.write(f"  Avg daily: {r.get('avg_daily_sales'):.3f}, last_inventory: {r.get('last_inventory')}\n")
                fh.write('\n')
        else:
            fh.write("  No reorders recommended.\n\n")

    print(f"Wrote results: {csv_path} and {txt_path}")


if __name__ == "__main__":
    main()
