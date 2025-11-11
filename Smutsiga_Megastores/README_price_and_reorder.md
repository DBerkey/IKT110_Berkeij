Price & Reorder Analysis
=========================

Purpose
-------
This script analyzes a JSONL file with sales/inventory events and produces per-product metrics, price change suggestions, and reorder quantity recommendations.

Outputs
-------
- products_summary.csv  : Detailed per-product metrics and suggested actions.
- suggestions.txt       : Human-friendly summary prioritized for action.

Quick usage (PowerShell):

```powershell
python .\Smutsiga_Megastores\price_and_reorder_analysis.py --input .\path\to\sales.jsonl --output-dir .\analysis_out --lead-time 14 --z 1.65 --desired-margin 0.30
```

Expected/recognized JSON keys
-----------------------------
The script is flexible and will try these keys (in order):
- product id: `product_id`, `id`, `sku`
- product name: `product_name`, `name`, `title`
- price: `price`, `unit_price`, `sale_price`
- quantity sold: `quantity`, `qty`, `quantity_sold`, `units_sold`
- cost: `cost`, `unit_cost`, `unit_cost_price`
- inventory: `inventory`, `stock`, `on_hand`, `quantity_on_hand`
- timestamp: `timestamp`, `time`, `date`, `sold_at`

If fields are missing, the script will make reasonable fallbacks (for instance, if inventory is missing it will recommend an order to cover ~lead_time+30 days of sales).

Main parameters
---------------
- `--lead-time`: lead time in days for reordering (default 14)
- `--z`: safety stock multiplier (e.g. 1.65 for ~95% service level)
- `--desired-margin`: target gross margin (fraction)
- `--price-change-limit`: maximum fraction to change price in one recommendation (default 0.20)

Notes & assumptions
-------------------
- The script uses simple heuristics for price suggestions (margin target, stock levels, and estimated elasticity if historical price variation exists). It's not a full pricing engine.
- Elasticity is estimated by fitting a log-log regression of quantity on price; this requires at least a few observations with price variation.
- Reorder quantities are based on average daily sales, lead time and safety stock (z * std_daily * sqrt(lead_time)).

Next steps / improvements
------------------------
- Add optional parameter to produce per-product KPI charts (sales over time).
- Add a small simulation to project revenue impact of suggested price changes using estimated elasticity.
- Integrate with procurement to convert suggestions into purchase orders.

Contact
-------
If some fields in your JSONL file are named differently, either rename them or open an issue and I can adapt the key mapping.
