"""SimpleMeanBidderAgent: bids the rolling mean of historical bids + 1 on every auction."""

from __future__ import annotations

import random
from statistics import mean
from typing import Any, Dict, List


class SimpleMeanBidderAgent:
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
        if agent_id not in states or not auctions:
            return {"bids": {}, "pool": 0}

        self._record_prev_bids(prev_auctions)

        my_gold = int(states[agent_id].get("gold", 0))
        if my_gold <= 0:
            return {"bids": {}, "pool": 0}

        average_bid = mean(self.bid_history) if self.bid_history else 1.0
        target = int(max(1, average_bid + 1))

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

        return {"bids": bids, "pool": 0}

    def _record_prev_bids(self, prev_auctions: Dict[str, Dict[str, Any]]) -> None:
        for info in prev_auctions.values():
            bids = info.get("bids") or []
            if not bids:
                continue
            winning_bid = bids[0].get("gold")
            if isinstance(winning_bid, (int, float)):
                self.bid_history.append(float(winning_bid))


__all__ = ["SimpleMeanBidderAgent"]
