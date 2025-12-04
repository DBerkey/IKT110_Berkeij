"""RandomSplashBidderAgent: sprays random bids across random auctions."""

from __future__ import annotations

import random
from typing import Any, Dict


class RandomSplashBidderAgent:
    def __init__(
        self,
        agent_id: str,
        min_spend_ratio: float = 0.35,
        max_spend_ratio: float = 0.85,
        seed: int | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.min_spend_ratio = max(0.0, min(min_spend_ratio, 1.0))
        self.max_spend_ratio = max(self.min_spend_ratio, min(max_spend_ratio, 1.0))
        self.random = random.Random(seed)

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

        my_state = states[agent_id]
        my_gold = int(my_state.get("gold", 0))
        my_points = int(my_state.get("points", 0))
        if my_gold <= 0:
            pool_spend = self._maybe_buy_pool(pool, my_points)
            return {"bids": {}, "pool": pool_spend}

        spend_ratio = self.random.uniform(self.min_spend_ratio, self.max_spend_ratio)
        spend_budget = max(1, min(int(my_gold * spend_ratio), my_gold))

        auction_ids = list(auctions.keys())
        self.random.shuffle(auction_ids)

        bids: Dict[str, int] = {}
        remaining = spend_budget
        for auction_id in auction_ids:
            if remaining <= 0:
                break
            max_bid = max(1, int(remaining * 0.6))
            bid = self.random.randint(1, min(max_bid, remaining))
            bids[auction_id] = bid
            remaining -= bid

        pool_spend = self._maybe_buy_pool(pool, my_points)
        return {"bids": bids, "pool": pool_spend}

    def _maybe_buy_pool(self, pool_value: int, my_points: int) -> int:
        if pool_value <= 10 or my_points <= 0:
            return 0
        if self.random.random() < 0.25:
            return 1 if my_points >= 1 else 0
        if self.random.random() < 0.1 and my_points >= 2:
            return 2
        return 0


__all__ = ["RandomSplashBidderAgent"]
