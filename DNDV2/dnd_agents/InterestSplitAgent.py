"""InterestSplitAgent: adaptation of the user's staged bidding bot for dnd_auction_game.

It keeps memory of previous gold/points to detect stagnation, waits a few rounds
before bidding, and splits auctions into bottom 70% (conservative bids) vs top
30% (aggressive bids funded by gold above the bank limit)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class InterestSplitState:
    prev_gold: int = 0
    prev_points: int = 0
    stagnation_rounds: int = 0


class InterestSplitAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = InterestSplitState()
        self.minimum_reserve = 5_000
        self.wait_rounds = 4
        self.stagnation_trigger = 10
        self.conservative_fraction = 0.25

    def make_bid(
        self,
        agent_id: str,
        round: int,
        states: Dict[str, Dict[str, Any]],
        auctions: Dict[str, Dict[str, Any]],
        prev_auctions: Dict[str, Dict[str, Any]],
        pool: int,
        prev_pool_buys: Dict[str, int],
        bank_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        if agent_id not in states:
            return {"bids": {}, "pool": 0}

        agent_state = states[agent_id]
        current_gold = int(agent_state.get("gold", 0))
        current_points = int(agent_state.get("points", 0))

        if current_points > self.state.prev_points:
            self.state.stagnation_rounds = 0
        else:
            self.state.stagnation_rounds += 1

        bid_multiplier = 1.1 if self.state.stagnation_rounds >= self.stagnation_trigger else 1.0

        if round < self.wait_rounds or not auctions:
            self._snapshot(current_gold, current_points)
            return {"bids": {}, "pool": 0}

        bank_limit = self._current_bank_limit(bank_state)
        gold_above_limit = max(0, current_gold - bank_limit)
        usable_gold = max(0, current_gold - self.minimum_reserve)
        remaining_gold = usable_gold
        bids: Dict[str, int] = {}

        auction_values = [
            (auction_id, auction, self._expected_value(auction))
            for auction_id, auction in auctions.items()
        ]
        auction_values.sort(key=lambda entry: entry[2])
        split_index = int(len(auction_values) * 0.7)
        bottom_segment = auction_values[:split_index]
        top_segment = auction_values[split_index:]

        # Aggressive bids funded by gold above the bank limit
        aggressive_gold = min(gold_above_limit, remaining_gold)
        if top_segment and aggressive_gold > 0:
            per_top_bid = max(1, aggressive_gold // len(top_segment))
            for auction_id, _auction, _value in top_segment:
                if aggressive_gold <= 0 or remaining_gold <= 0:
                    break
                bid = min(per_top_bid, aggressive_gold, remaining_gold)
                bid = int(max(1, bid * bid_multiplier))
                bids[auction_id] = bid
                aggressive_gold -= bid
                remaining_gold -= bid

        # Conservative bids for the bottom 70% share a small fraction of usable gold
        conservative_budget = int(remaining_gold * self.conservative_fraction)
        if bottom_segment and conservative_budget > 0:
            per_bottom_bid = max(1, conservative_budget // len(bottom_segment))
            for auction_id, _auction, _value in bottom_segment:
                if conservative_budget <= 0 or remaining_gold <= 0:
                    break
                bid = min(per_bottom_bid, conservative_budget, remaining_gold)
                bid = int(max(1, bid * bid_multiplier))
                bids.setdefault(auction_id, bid)
                conservative_budget -= bid
                remaining_gold -= bid

        self._snapshot(current_gold - sum(bids.values()), current_points)
        return {"bids": bids, "pool": 0}

    def _snapshot(self, gold: int, points: int) -> None:
        self.state.prev_gold = gold
        self.state.prev_points = points

    @staticmethod
    def _expected_value(auction: Dict[str, Any]) -> float:
        die = int(auction.get("die", 0) or 0)
        num = int(auction.get("num", 0) or 0)
        bonus = float(auction.get("bonus", 0) or 0)
        if die <= 0 or num <= 0:
            return bonus
        return ((die + 1) / 2) * num + bonus

    @staticmethod
    def _current_bank_limit(bank_state: Dict[str, Any]) -> int:
        limits = bank_state.get("bank_limit_per_round")
        if isinstance(limits, list) and limits:
            return int(limits[0])
        return 2_000


__all__ = ["InterestSplitAgent"]
