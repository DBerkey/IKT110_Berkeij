"""FixedSpreadBidderAgent: splits a fixed share of gold evenly across every auction."""

from __future__ import annotations

from typing import Any, Dict


class FixedSpreadBidderAgent:
    def __init__(self, agent_id: str, spend_ratio: float = 0.6) -> None:
        self.agent_id = agent_id
        self.spend_ratio = max(0.0, min(spend_ratio, 1.0))

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

        my_gold = int(states[agent_id].get("gold", 0))
        if my_gold <= 0:
            return {"bids": {}, "pool": 0}

        spend_budget = int(my_gold * self.spend_ratio)
        spend_budget = max(1, min(spend_budget, my_gold))

        num_auctions = len(auctions)
        if spend_budget < num_auctions:
            # not enough gold to hit every auction, so bid 1 on as many as possible
            bids = {auction_id: 1 for auction_id in list(auctions.keys())[:spend_budget]}
            return {"bids": bids, "pool": 0}

        base = spend_budget // num_auctions
        remainder = spend_budget % num_auctions

        bids: Dict[str, int] = {}
        for idx, auction_id in enumerate(auctions.keys()):
            bid = base + (1 if idx < remainder else 0)
            bids[auction_id] = bid

        return {"bids": bids, "pool": 0}


__all__ = ["FixedSpreadBidderAgent"]
