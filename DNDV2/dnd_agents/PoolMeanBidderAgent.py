"""PoolMeanBidderAgent: bids historical mean + 1 and spends points when pool is juicy."""

from __future__ import annotations

import random
from statistics import mean
from typing import Any, Dict, List


class PoolMeanBidderAgent:
    def __init__(self, agent_id: str, pool_aggression: float = 0.25, seed: int | None = None) -> None:
        self.agent_id = agent_id
        self.pool_aggression = max(0.0, min(pool_aggression, 1.0))
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
        if agent_id not in states or not auctions:
            return {"bids": {}, "pool": 0}

        self._record_prev_bids(prev_auctions)

        my_state = states[agent_id]
        my_gold = int(my_state.get("gold", 0))
        my_points = int(my_state.get("points", 0))
        if my_gold <= 0:
            pool_spend = self._decide_pool_spend(pool, my_points)
            return {"bids": {}, "pool": pool_spend}

        avg_bid = mean(self.bid_history) if self.bid_history else 1.0
        target = int(max(1, avg_bid + 1))

        auction_ids = list(auctions.keys())
        self.random.shuffle(auction_ids)

        bids: Dict[str, int] = {}
        remaining_gold = my_gold
        for auction_id in auction_ids:
            if remaining_gold <= 0:
                break
            bid = min(target, remaining_gold)
            bids[auction_id] = bid
            remaining_gold -= bid

        pool_spend = self._decide_pool_spend(pool, my_points)
        return {"bids": bids, "pool": pool_spend}

    def _decide_pool_spend(self, pool_value: int, my_points: int) -> int:
        if pool_value <= 0 or my_points <= 0:
            return 0
        if pool_value < 20:
            return 0

        desired = int(max(1, pool_value / 40))
        aggressive_cap = int(max(1, my_points * self.pool_aggression))
        desired = max(desired, aggressive_cap)
        return min(my_points, desired)

    def _record_prev_bids(self, prev_auctions: Dict[str, Dict[str, Any]]) -> None:
        for info in prev_auctions.values():
            bids = info.get("bids") or []
            if not bids:
                continue
            winning_bid = bids[0].get("gold")
            if isinstance(winning_bid, (int, float)):
                self.bid_history.append(float(winning_bid))


__all__ = ["PoolMeanBidderAgent"]
