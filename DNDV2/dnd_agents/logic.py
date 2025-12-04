import json
import math
import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

_AGENT_SINGLETON = None


class AuctionAgent:
    def __init__(
        self,
        agent_id: str = "agent_ai",
        initial_params: Optional[Dict[str, Any]] = None,
        persist_path: str = "./agent_state.json",
    ):
        self.agent_id = agent_id
        p = initial_params or {}

        self.params = {
            "target_gpp": p.get("target_gpp", 5.0),
            "aggressiveness": p.get("aggressiveness", 1.0),
            "bid_coverage": p.get("bid_coverage", 0.1),
            "max_gold_fraction_per_round": p.get("max_gold_fraction_per_round", 0.2),
            "min_bid": p.get("min_bid", 1),
            "pool_policy": p.get("pool_policy", "conservative"),
            "adaptive": p.get("adaptive", True),
            "pool_min_gpp_multiplier": p.get("pool_min_gpp_multiplier", 1.5),
            "max_points_fraction_per_round": p.get(
                "max_points_fraction_per_round", 0.10
            ),
            "min_point_reserve": p.get("min_point_reserve", 2),
            "pool_required_win_rate": p.get("pool_required_win_rate", 0.10),
        }

        self.metrics = {
            "round": [],
            "gold_spent": [],
            "points_gained": [],
            "gpp": [],
            "win_rate": [],
            "avg_bid": [],
            "pool_points_spent": [],
            "pool_gold_received": [],
        }

        self.cumulative = {"gold_spent": 0, "points_gained": 0, "wins": 0, "bids": 0}
        self.history = []

        self.persist_path = persist_path
        self._try_load()

    @staticmethod
    def expected_value_of_auction(auction: Dict[str, int]) -> float:
        die = int(auction.get("die", 6))
        num = int(auction.get("num", 1))
        bonus = int(auction.get("bonus", 0))
        return num * (die + 1) / 2.0 + bonus

    def env_average_gpp(self, states: Dict[str, Dict[str, int]]) -> float:
        total_gold = 0
        total_points = 0
        for aid, s in states.items():
            if aid == self.agent_id:
                continue
            total_gold += s.get("gold", 0)
            total_points += s.get("points", 0)
        if total_points == 0:
            return float("inf") if total_gold > 0 else 0.0
        return total_gold / total_points

    def adaptive_update(self, round_no, states, prev_auctions):
        if not self.params["adaptive"]:
            return

        my_gpp = (
            self.cumulative["gold_spent"] / max(1, self.cumulative["points_gained"])
            if self.cumulative["points_gained"] > 0
            else float("inf")
        )
        env_gpp = self.env_average_gpp(states)
        ratio = my_gpp / (env_gpp + 1e-9) if env_gpp > 0 else 1.0

        if ratio > 1.2:
            self.params["aggressiveness"] *= 0.9
            self.params["target_gpp"] *= 1.05
        elif ratio < 0.8:
            self.params["aggressiveness"] *= 1.05
            self.params["target_gpp"] *= 0.98

        self.params["aggressiveness"] = float(
            max(0.2, min(2.5, self.params["aggressiveness"]))
        )
        self.params["target_gpp"] = float(
            max(0.5, min(200.0, self.params["target_gpp"]))
        )

    def make_bid(
        self,
        agent_id,
        round_no,
        states,
        auctions,
        prev_auctions,
        pool,
        prev_pool_buys,
        bank_state,
    ):
        my_state = states.get(agent_id, {"gold": 0, "points": 0})
        bids_out = {}
        gold = my_state.get("gold", 0)
        points = my_state.get("points", 0)

        self.adaptive_update(round_no, states, prev_auctions)

        # ---------------- Bank-based spending logic ----------------
        remaining_rounds = len(bank_state["gold_income_per_round"])
        bank_limit = bank_state["bank_limit_per_round"][0]
        bank_interest = bank_state["bank_interest_per_round"][0]

        # Early/mid game: save slightly below bank limit
        if round_no < 600:
            target_gold = int(bank_limit * 0.95)  # keep slightly below limit
            max_spend = max(1, gold - target_gold)
        # Mid-late game: start spending more aggressively
        elif 600 <= round_no <= 700:
            max_spend = max(1, int(gold * 0.5))
        else:  # late game
            max_spend = max(1, gold)

        gold_remaining = max_spend
        bids_placed = 0

        # ---------------- Auction evaluation ----------------
        ev_list = [
            (a_id, self.expected_value_of_auction(a), a) for a_id, a in auctions.items()
        ]
        ev_list.sort(key=lambda x: x[1], reverse=True)

        num_to_consider = max(1, int(len(ev_list) * self.params["bid_coverage"]))
        candidates = ev_list[:num_to_consider]

        for a_id, ev, a in candidates:
            desired_gold = (
                ev * self.params["target_gpp"] * self.params["aggressiveness"]
            )
            bid = int(max(self.params["min_bid"], min(desired_gold, gold_remaining)))

            if bid <= 0:
                break

            prev = prev_auctions.get(a_id)
            if prev and prev.get("bids"):
                prev_win = prev["bids"][0]
                prev_gold = prev_win.get("gold", 0)
                if prev_gold < bid:
                    bid = max(self.params["min_bid"], int((bid + prev_gold) / 2))

            bid = min(bid, gold_remaining, gold)
            if bid <= 0:
                continue

            bids_out[a_id] = bid
            gold_remaining -= bid
            bids_placed += 1
            self.cumulative["bids"] += 1

        # ---------------- Pool logic (unchanged) ----------------
        pool_purchase = 0
        if pool > 0 and points > 0 and self.params["pool_policy"] != "none":
            env_gpp = self.env_average_gpp(states)
            max_candidate = max(
                1, int(points * self.params["max_points_fraction_per_round"])
            )
            chosen = 0

            for x in range(1, max_candidate + 1):
                gpp_if_spent = float(pool) / float(x)
                if (
                    gpp_if_spent
                    >= self.params["target_gpp"]
                    * self.params["pool_min_gpp_multiplier"]
                ):
                    if points - x >= self.params["min_point_reserve"]:
                        chosen = x
                        break

            recent_win_rate = (
                self.cumulative["wins"] / max(1, self.cumulative["bids"])
                if self.cumulative["bids"] > 0
                else 0.0
            )

            if chosen > 0:
                req = self.params["pool_required_win_rate"]
                if self.params["pool_policy"] == "aggressive":
                    req *= 0.5
                if recent_win_rate >= req:
                    pool_purchase = min(points, chosen)

        # ---------------- Metrics and history (unchanged) ----------------
        gold_spent = sum(bids_out.values())
        self.metrics["round"].append(round_no)
        self.metrics["gold_spent"].append(gold_spent)
        self.metrics["points_gained"].append(0)
        self.metrics["gpp"].append(None)
        self.metrics["win_rate"].append(None)
        self.metrics["avg_bid"].append(
            (gold_spent / max(1, bids_placed)) if bids_placed else 0
        )
        self.metrics["pool_points_spent"].append(pool_purchase)
        self.metrics["pool_gold_received"].append(0)

        self.history.append(
            {
                "round": round_no,
                "state": my_state,
                "bids": bids_out,
                "pool_purchase": pool_purchase,
                "params": dict(self.params),
            }
        )

        self._try_persist()

        out = {"bids": bids_out}
        if pool_purchase > 0:
            out["pool"] = pool_purchase
        return out

    # ---------------- Update after round ----------------
    def update_from_round_results(
        self, round_no, prev_auctions, outcomes, pool_result=None
    ):
        rec = None
        if self.history and self.history[-1]["round"] == round_no:
            rec = self.history[-1]
        else:
            for r in reversed(self.history):
                if r["round"] == round_no:
                    rec = r
                    break
        if rec is None:
            return

        bids = rec["bids"]
        gold_spent = sum(bids.values()) + rec.get("pool_purchase", 0)
        points_gained = 0
        wins = 0

        for a_id, bid_amt in bids.items():
            r = outcomes.get(a_id)
            if not r:
                continue
            if r.get("winner") == self.agent_id:
                wins += 1
                points_gained += r.get("reward", 0)

        self.cumulative["gold_spent"] += gold_spent
        self.cumulative["points_gained"] += points_gained
        self.cumulative["wins"] += wins

        self.metrics["points_gained"][-1] = points_gained
        if points_gained > 0:
            self.metrics["gpp"][-1] = gold_spent / points_gained
        else:
            self.metrics["gpp"][-1] = float("inf") if gold_spent > 0 else 0

        rolling = (
            self.cumulative["wins"] / max(1, self.cumulative["bids"])
            if self.cumulative["bids"] > 0
            else 0
        )
        self.metrics["win_rate"][-1] = rolling

        if pool_result is not None:
            self.metrics["pool_gold_received"][-1] = pool_result

        self._try_persist()

    def _try_persist(self):
        try:
            with open(self.persist_path, "w") as f:
                json.dump(
                    {
                        "params": self.params,
                        "metrics": self.metrics,
                        "cumulative": self.cumulative,
                    },
                    f,
                    indent=2,
                )
        except:
            pass

    def _try_load(self):
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path) as f:
                data = json.load(f)
            self.params.update(data.get("params", {}))
            for k, v in data.get("metrics", {}).items():
                self.metrics[k] = v
            for k, v in data.get("cumulative", {}).items():
                self.cumulative[k] = v
        except:
            pass


def make_bid(
    agent_id,
    round_no,
    states,
    auctions,
    prev_auctions,
    pool,
    prev_pool_buys,
    bank_state,
):
    global _AGENT_SINGLETON
    if _AGENT_SINGLETON is None:
        _AGENT_SINGLETON = AuctionAgent(agent_id=agent_id)
    return _AGENT_SINGLETON.make_bid(
        agent_id,
        round_no,
        states,
        auctions,
        prev_auctions,
        pool,
        prev_pool_buys,
        bank_state,
    )
