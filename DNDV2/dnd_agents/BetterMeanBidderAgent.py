"""BetterMeanBidderAgent: bids the rolling mean of historical bids + 1 and taps the pool."""

from __future__ import annotations

import random
from statistics import mean
from typing import Any, Dict, List


class BetterMeanBidderAgent:
    def __init__(self, agent_id: str, seed: int | None = None) -> None:
        self.agent_id = agent_id
        self.random = random.Random(seed)
        self.bid_history: List[float] = []

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

        self._record_prev_bids(prev_auctions)

        state = states[agent_id]
        my_gold = int(state.get("gold", 0))
        my_points = int(state.get("points", 0))

        bids: Dict[str, int] = {}
        if auctions and my_gold > 0:
            average_bid = mean(self.bid_history) if self.bid_history else 1.0
            target = int(max(1, average_bid + 5))

            auction_ids = list(auctions.keys())
            self.random.shuffle(auction_ids)

            remaining_gold = my_gold
            for auction_id in auction_ids:
                if remaining_gold <= 0:
                    break
                bid = min(target, remaining_gold)
                bids[auction_id] = bid
                remaining_gold -= bid

        pool_points = self._decide_pool_spend(pool, my_points, prev_pool_buys)

        return {"bids": bids, "pool": pool_points}

    def _record_prev_bids(self, prev_auctions: Dict[str, Dict[str, Any]]) -> None:
        for info in prev_auctions.values():
            bids = info.get("bids") or []
            if not bids:
                continue
            winning_bid = bids[0].get("gold")
            if isinstance(winning_bid, (int, float)):
                self.bid_history.append(float(winning_bid))

    def _decide_pool_spend(
        self,
        gold_in_pool: int,
        my_points: int,
        prev_pool_buys: Dict[str, int],
    ) -> int:
        if gold_in_pool <= 0 or my_points <= 0:
            return 0

        pressure = min(max(gold_in_pool, 0) / 200.0, 1.0)
        target_ratio = 0.1 + 0.4 * pressure
        desired_spend = int(my_points * target_ratio)
        if desired_spend <= 0:
            desired_spend = 1

        recent_spend = int((prev_pool_buys or {}).get(self.agent_id, 0))
        adjusted_spend = max(desired_spend - recent_spend // 2, 0)

        return min(adjusted_spend, my_points)


__all__ = ["BetterMeanBidderAgent"]
