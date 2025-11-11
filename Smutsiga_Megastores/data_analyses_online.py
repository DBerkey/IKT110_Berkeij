"""
Author: Douwe Berkeij
Date: 04-11-2025
"""

if __name__ == "__main__":
    import argparse
    import os
    import json
    from collections import defaultdict, Counter
    from datetime import datetime

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from collections import deque

    parser = argparse.ArgumentParser(description="Analyze smutsiga .jsonl transaction file and produce plots")
    parser.add_argument("file", nargs="?", default="C:/Users/berke/Downloads/smutsiga_onlinestore_data/smutsiga_onlinestore_data.jsonl", help="Path to jsonl file")
    parser.add_argument("--out", default="analysis_output", help="Output folder to save plots and summary")
    parser.add_argument("--save-txns", action="store_true", help="Save parsed transactions to CSV for deeper analysis")
    args = parser.parse_args()

    filePath = args.file
    out_root = args.out

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"sales_analysis_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Data accumulators
    total_revenue = 0.0
    total_cogs = 0.0
    total_shipping_out = 0.0
    total_return_shipping = 0.0
    total_refunds = 0.0

    # inventory per product: track stock and weighted-average unit cost
    inventory = {}
    # purchase lots per product for FIFO COGS: deque of {qty, unit_cost}
    purchase_lots = defaultdict(lambda: deque())

    # optionally save parsed transactions
    parsed_txns = []

    # per-product aggregates
    prod_metrics = defaultdict(lambda: defaultdict(float))
    loyalty_counts = Counter()
    txn_counts = Counter()
    returns_count = 0
    # loyalty-level metrics
    loyalty_stats = defaultdict(lambda: defaultdict(float))

    # helpers
    def safe_get(d, key, default=None):
        return d.get(key, default)

    def as_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    def as_int(x, default=0):
        try:
            return int(float(x))
        except Exception:
            return default

    with open(filePath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                txn = json.loads(line)
            except Exception:
                # skip malformed lines
                continue

            txn_type = txn.get("txn_type")
            goods_id = txn.get("goods_id") or "UNKNOWN"
            goods_name = txn.get("goods_name") or goods_id
            # keep last-seen product name for reporting
            prod_metrics[goods_id]["goods_name"] = goods_name

            if txn_type == "supplier_purchase":
                price = as_float(safe_get(txn, "price", 0.0))
                qty = as_int(safe_get(txn, "amount_purchased", 0))
                if goods_id not in inventory:
                    inventory[goods_id] = {"stock": 0, "unit_cost": 0.0}
                # FIFO: append lot
                if qty > 0:
                    purchase_lots[goods_id].append({"qty": qty, "unit_cost": price})
                    inventory[goods_id]["stock"] = inventory[goods_id].get("stock", 0) + qty
                    # update last-seen unit_cost for reporting
                    inventory[goods_id]["unit_cost"] = price
                    prod_metrics[goods_id]["purchased_qty"] += qty
                    prod_metrics[goods_id]["purchased_cost"] += price * qty
                txn_counts["supplier_purchase"] += 1

            elif txn_type == "customer_sale":
                price = safe_get(txn, "price", 0.0)
                qty = int(safe_get(txn, "amount_purchased", 0))
                shipping = float(safe_get(txn, "outbound_shipping_cost_to_us", 0.0))
                free_shipping = bool(txn.get("free_shipping_applied", False))
                loyalty = txn.get("loyalty")
                loyalty = loyalty if loyalty is not None else "None"

                revenue = price * qty
                total_revenue += revenue
                total_shipping_out += shipping
                loyalty_counts[loyalty] += 1
                txn_counts["customer_sale"] += 1

                # determine COGS using FIFO lots
                remaining = qty
                cost = 0.0
                while remaining > 0 and purchase_lots.get(goods_id):
                    lot = purchase_lots[goods_id][0]
                    take = min(remaining, lot["qty"])
                    cost += take * lot["unit_cost"]
                    lot["qty"] -= take
                    remaining -= take
                    if lot["qty"] <= 0:
                        purchase_lots[goods_id].popleft()
                # if remaining > 0, we don't have prior cost info; assume unit_cost 0 for missing lots
                if remaining > 0:
                    # assume cost 0 for missing inventory (could be negative stock)
                    cost += 0.0
                # reduce stock if tracked
                if goods_id in inventory:
                    inventory[goods_id]["stock"] = max(0, inventory[goods_id].get("stock", 0) - qty)

                prod_metrics[goods_id]["revenue"] += revenue
                prod_metrics[goods_id]["sold_qty"] += qty
                prod_metrics[goods_id]["cogs"] += cost
                prod_metrics[goods_id]["shipping_out"] += shipping
                prod_metrics[goods_id]["free_shipping_count"] += 1 if free_shipping else 0
                prod_metrics[goods_id]["loyalty"] = loyalty

                # track at loyalty level
                loyalty_stats[loyalty]["sales_count"] += 1
                loyalty_stats[loyalty]["shipping_total"] += shipping
                loyalty_stats[loyalty]["revenue"] += revenue
                if free_shipping:
                    loyalty_stats[loyalty]["free_shipping_count"] += 1

                # optionally record transaction row
                if args.save_txns:
                    parsed_txns.append({
                        "txn_type": txn_type,
                        "transaction_id": txn.get("transaction_id"),
                        "goods_id": goods_id,
                        "goods_name": goods_name,
                        "price": price,
                        "qty": qty,
                        "shipping": shipping,
                        "free_shipping": free_shipping,
                        "loyalty": loyalty,
                        "revenue": revenue
                    })
            elif txn_type == "customer_return":
                returns_count += 1
                refund_total = float(safe_get(txn, "refund_total", 0.0))
                amount_returned = int(safe_get(txn, "amount_returned", 0))
                return_shipping = float(safe_get(txn, "return_shipping_cost_to_us", 0.0))
                loyalty = txn.get("loyalty")
                loyalty = loyalty if loyalty is not None else "None"

                # track free_return flag
                free_return = bool(txn.get("free_return_applied", False))
                loyalty_stats[loyalty]["returns_count"] += 1
                loyalty_stats[loyalty]["return_shipping_total"] += return_shipping
                if free_return:
                    loyalty_stats[loyalty]["free_return_count"] += 1

                total_refunds += refund_total
                total_return_shipping += return_shipping
                txn_counts["customer_return"] += 1

                # return restores inventory stock and reverses some COGS
                # we add returned units as a new lot at current known unit_cost (best-effort)
                unit_cost = inventory.get(goods_id, {}).get("unit_cost", 0.0)
                if amount_returned > 0:
                    purchase_lots[goods_id].appendleft({"qty": amount_returned, "unit_cost": unit_cost})
                    inventory.setdefault(goods_id, {})
                    inventory[goods_id]["stock"] = inventory[goods_id].get("stock", 0) + amount_returned

                # recovered cost based on unit_cost
                recovered_cost = unit_cost * amount_returned
                # subtract from product-level cogs (but avoid negative)
                prod_metrics[goods_id]["cogs"] = max(0.0, prod_metrics[goods_id].get("cogs", 0.0) - recovered_cost)

                prod_metrics[goods_id]["returned_qty"] += amount_returned
                prod_metrics[goods_id]["refunds"] += refund_total
                prod_metrics[goods_id]["return_shipping"] += return_shipping
                prod_metrics[goods_id]["loyalty"] = loyalty

                # optionally record return transaction
                if args.save_txns:
                    parsed_txns.append({
                        "txn_type": txn_type,
                        "transaction_id": txn.get("transaction_id"),
                        "goods_id": goods_id,
                        "goods_name": goods_name,
                        "refund_total": refund_total,
                        "amount_returned": amount_returned,
                        "return_shipping": return_shipping,
                        "free_return": free_return,
                        "loyalty": loyalty
                    })

            else:
                # Unknown txn type: skip but count
                txn_counts["other"] += 1

    # finalize profit
    total_profit = total_revenue - total_cogs - total_shipping_out - total_return_shipping - total_refunds

    # create DataFrame for products
    rows = []
    for gid, m in prod_metrics.items():
        rows.append({
            "goods_id": gid,
            "goods_name": m.get("goods_name", gid),
            "revenue": m.get("revenue", 0.0),
            "sold_qty": m.get("sold_qty", 0),
            "cogs": m.get("cogs", 0.0),
            "shipping_out": m.get("shipping_out", 0.0),
            "returned_qty": m.get("returned_qty", 0),
            "refunds": m.get("refunds", 0.0),
            "return_shipping": m.get("return_shipping", 0.0),
            "free_shipping_count": m.get("free_shipping_count", 0),
            "loyalty": m.get("loyalty", "None"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No product-level data parsed. Check input file.")
    else:
        df.to_csv(os.path.join(out_dir, "product_summary.csv"), index=False)

        # Supplier summary CSV: list products with supplier purchases
        supplier_rows = []
        total_purchased_cost = 0.0
        total_purchased_qty = 0
        for gid, m in prod_metrics.items():
            pq = int(m.get("purchased_qty", 0))
            pcost = float(m.get("purchased_cost", 0.0))
            if pq > 0:
                avg_price = (pcost / pq) if pq > 0 else 0.0
                supplier_rows.append({
                    "goods_id": gid,
                    "goods_name": m.get("goods_name", gid),
                    "purchased_qty": pq,
                    "purchased_cost": round(pcost,2),
                    "avg_price": round(avg_price,4)
                })
                total_purchased_cost += pcost
                total_purchased_qty += pq

        if supplier_rows:
            import csv
            supplier_csv = os.path.join(out_dir, "supplier_summary.csv")
            with open(supplier_csv, "w", newline='', encoding='utf-8') as scsv:
                writer = csv.DictWriter(scsv, fieldnames=["goods_id","goods_name","purchased_qty","purchased_cost","avg_price"])
                writer.writeheader()
                for r in supplier_rows:
                    writer.writerow(r)

            # write an overall supplier totals file as well
            with open(os.path.join(out_dir, "supplier_totals.txt"), "w", encoding="utf-8") as st:
                st.write(f"Total purchased qty: {total_purchased_qty}\n")
                st.write(f"Total purchased cost: {total_purchased_cost:.2f}\n")

    # write loyalty summary CSV
    if loyalty_stats:
        loy_rows = []
        for loy, ls in loyalty_stats.items():
            sales = int(ls.get("sales_count", 0))
            returns = int(ls.get("returns_count", 0))
            shipping_total = float(ls.get("shipping_total", 0.0))
            return_shipping_total = float(ls.get("return_shipping_total", 0.0))
            free_ship = int(ls.get("free_shipping_count", 0))
            free_ret = int(ls.get("free_return_count", 0))
            revenue_loy = float(ls.get("revenue", 0.0))
            avg_ship = (shipping_total / sales) if sales > 0 else 0.0
            loy_rows.append({
                "loyalty": loy,
                "sales_count": sales,
                "revenue": round(revenue_loy,2),
                "shipping_total": round(shipping_total,2),
                "avg_shipping": round(avg_ship,2),
                "free_shipping_count": free_ship,
                "returns_count": returns,
                "return_shipping_total": round(return_shipping_total,2),
                "free_return_count": free_ret
            })
        loy_df = pd.DataFrame(loy_rows)
        loy_df.to_csv(os.path.join(out_dir, "loyalty_summary.csv"), index=False)

    # (Inventory/returns/loyalty plots are generated below after df_plot exists)

    # Summary text
    summary_path = os.path.join(out_dir, "analysis_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as s:
        s.write(f"Input file: {filePath}\n")
        s.write(f"Transactions processed: {sum(txn_counts.values())}\n")
        s.write(f" - customer_sale: {txn_counts.get('customer_sale',0)}\n")
        s.write(f" - supplier_purchase: {txn_counts.get('supplier_purchase',0)}\n")
        s.write(f" - customer_return: {txn_counts.get('customer_return',0)}\n")
        s.write(f"Total revenue: {total_revenue:.2f}\n")
        # compute total_cogs from product-level cogs (after FIFO adjustments)
        total_cogs = sum([m.get("cogs", 0.0) for m in prod_metrics.values()])
        s.write(f"Total COGS: {total_cogs:.2f}\n")
        s.write(f"Total shipping outbound: {total_shipping_out:.2f}\n")
        s.write(f"Total return shipping: {total_return_shipping:.2f}\n")
        s.write(f"Total refunds: {total_refunds:.2f}\n")
        total_profit = total_revenue - total_cogs - total_shipping_out - total_return_shipping - total_refunds
        s.write(f"Estimated profit: {total_profit:.2f}\n")

    # --- PLOTTING ---
    # 1) Revenue by product (top 20)
    if not df.empty:
        df_plot = df.copy()
        df_plot["revenue"] = df_plot["revenue"].fillna(0)
        top_revenue = df_plot.sort_values("revenue", ascending=False).head(20)
        plt.figure(figsize=(10,6))
        plt.barh(top_revenue["goods_id"].astype(str), top_revenue["revenue"].astype(float))
        plt.xlabel("Revenue")
        plt.title("Top 20 products by revenue")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "top20_revenue.png"))
        plt.close()

        # 2) Profit by product (revenue - cogs)
        df_plot["profit"] = df_plot["revenue"].fillna(0) - df_plot["cogs"].fillna(0)
        top_profit = df_plot.sort_values("profit", ascending=False).head(20)
        plt.figure(figsize=(10,6))
        plt.barh(top_profit["goods_id"].astype(str), top_profit["profit"].astype(float), color="tab:green")
        plt.xlabel("Profit")
        plt.title("Top 20 products by profit")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "top20_profit.png"))
        plt.close()

        # 3) Loyalty distribution (counts)
        loy = pd.Series(loyalty_counts)
        if not loy.empty:
            plt.figure(figsize=(6,6))
            loy.plot.pie(autopct="%1.1f%%", startangle=90)
            plt.ylabel("")
            plt.title("Loyalty tier distribution (transaction counts)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "loyalty_distribution.png"))
            plt.close()

        # 4) Returns summary
        returns_summary = df_plot.sort_values("returned_qty", ascending=False).head(20)
        plt.figure(figsize=(10,6))
        plt.barh(returns_summary["goods_id"].astype(str), returns_summary["returned_qty"].astype(int), color="tab:orange")
        plt.xlabel("Returned quantity")
        plt.title("Top 20 returned products by quantity")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "top_returns.png"))
        plt.close()

        # 5) Transaction type counts
        plt.figure(figsize=(6,4))
        keys = list(txn_counts.keys())
        vals = [txn_counts[k] for k in keys]
        colors = ["tab:blue","tab:orange","tab:green","tab:gray"][:len(keys)]
        plt.bar(keys, vals, color=colors)
        plt.title("Transaction counts by type")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "txn_counts.png"))
        plt.close()

        # --- ADDITIONAL PLOTS ---
        # Inventory: current stock per product (top 20)
        if inventory:
            inv_rows = []
            for gid, inv in inventory.items():
                inv_rows.append({
                    "goods_id": gid,
                    "stock": inv.get("stock", 0),
                    "unit_cost": inv.get("unit_cost", 0.0),
                    "goods_name": prod_metrics.get(gid, {}).get("goods_name", gid)
                })
            inv_df = pd.DataFrame(inv_rows)
            if not inv_df.empty:
                top_stock = inv_df.sort_values("stock", ascending=False).head(20)
                plt.figure(figsize=(10,6))
                plt.barh(top_stock["goods_id"].astype(str), top_stock["stock"].astype(float), color="tab:purple")
                plt.xlabel("Current stock")
                plt.title("Top 20 products by current inventory stock")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "top20_stock.png"))
                plt.close()

        # Return rate per product (returned / sold)
        df_plot["sold_qty_nonzero"] = df_plot["sold_qty"].replace(0, np.nan)
        df_plot["return_rate"] = df_plot["returned_qty"] / df_plot["sold_qty_nonzero"]
        rrate = df_plot[df_plot["sold_qty"] > 0].sort_values("return_rate", ascending=False).head(20)
        if not rrate.empty:
            plt.figure(figsize=(10,6))
            plt.barh(rrate["goods_id"].astype(str), rrate["return_rate"].fillna(0.0), color="tab:red")
            plt.xlabel("Return rate (returned / sold)")
            plt.title("Top 20 products by return rate (min 1 sold)")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "top_return_rates.png"))
            plt.close()

        # Loyalty correlation plots (from loyalty_summary)
        if loyalty_stats:
            loy_df = pd.DataFrame(loy_rows)
            loy_df = loy_df.sort_values("sales_count", ascending=False)
            x = np.arange(len(loy_df))
            plt.figure(figsize=(8,5))
            plt.bar(x - 0.15, loy_df["sales_count"].astype(int), width=0.3, label="Sales")
            plt.bar(x + 0.15, loy_df["returns_count"].astype(int), width=0.3, label="Returns")
            plt.xticks(x, loy_df["loyalty"].astype(str))
            plt.ylabel("Count")
            plt.title("Sales vs Returns by loyalty tier")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "loyalty_sales_vs_returns.png"))
            plt.close()

            # stacked free vs paid returns per loyalty
            free = loy_df["free_return_count"].astype(int)
            paid = (loy_df["returns_count"].astype(int) - free).clip(lower=0)
            plt.figure(figsize=(8,5))
            plt.bar(loy_df["loyalty"].astype(str), free, label="Free returns")
            plt.bar(loy_df["loyalty"].astype(str), paid, bottom=free, label="Paid returns")
            plt.ylabel("Returns count")
            plt.title("Returns by loyalty: free vs paid")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "loyalty_returns_free_paid.png"))
            plt.close()

            # avg shipping per loyalty
            plt.figure(figsize=(8,5))
            plt.bar(loy_df["loyalty"].astype(str), loy_df["avg_shipping"].astype(float), color="tab:cyan")
            plt.ylabel("Average shipping cost (to us)")
            plt.title("Average outbound shipping cost by loyalty tier")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "loyalty_avg_shipping.png"))
            plt.close()

            # scatter: avg shipping vs revenue per loyalty
            plt.figure(figsize=(7,5))
            xs = loy_df["avg_shipping"].astype(float)
            ys = loy_df["revenue"].astype(float)
            plt.scatter(xs, ys)
            for i, txt in enumerate(loy_df["loyalty"]):
                plt.annotate(str(txt), (xs.iat[i], ys.iat[i]), textcoords="offset points", xytext=(5,3))
            plt.xlabel("Avg shipping to us")
            plt.ylabel("Revenue")
            plt.title("Avg shipping vs Revenue by loyalty tier")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "loyalty_shipping_vs_revenue.png"))
            plt.close()

        # transaction-level scatter (shipping vs revenue) if saved transactions requested
        if args.save_txns and parsed_txns:
            txdf = pd.DataFrame(parsed_txns)
            sales_df = txdf[txdf["txn_type"] == "customer_sale"]
            if not sales_df.empty:
                plt.figure(figsize=(8,6))
                # color by loyalty
                loy_unique = list(sales_df["loyalty"].unique())
                colors_map = {l: i for i, l in enumerate(loy_unique)}
                c = sales_df["loyalty"].map(colors_map)
                plt.scatter(sales_df["shipping"].astype(float), sales_df["revenue"].astype(float), c=c, cmap="tab10", alpha=0.7)
                plt.xlabel("Shipping cost to us")
                plt.ylabel("Transaction revenue")
                plt.title("Transaction-level: shipping vs revenue (colored by loyalty)")
                # legend
                handles = []
                for l, idx in colors_map.items():
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', label=str(l), markerfacecolor=plt.cm.tab10(idx), markersize=6))
                plt.legend(handles=handles, title="loyalty")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "tx_shipping_vs_revenue.png"))
                plt.close()

                # save transaction-level CSV as well
                tx_csv = os.path.join(out_dir, "parsed_transactions.csv")
                sales_df.to_csv(tx_csv, index=False)

    print(f"Analysis complete. Outputs written to: {out_dir}")

