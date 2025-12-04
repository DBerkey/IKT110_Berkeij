"""BraavosLockdownAgent: focuses on stable gold-to-point conversion without pool risk.

Lessons from log 16:
- Phoenix/Sentinel hemorrhaged points via pool buys mid-game and bid way above
  clearing prices while win rate collapsed.
- Simple bots that kept their points (tiny_bid, random_single) still ranked
  because everybody else went negative. This agent therefore prioritizes
  consistent, medium bids, bans pool spending entirely, and limits how quickly
  gold can be depleted if win rate is low.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Deque, Dict, List, Tuple


class BraavosLockdownAgent:
    def __init__(self, agent_id: str, seed: int | None = None) -> None:
        self.agent_id = agent_id
        self.random = random.Random(seed)
        self.bid_history: Deque[int] = deque(maxlen=300)
        self.win_history: Deque[int] = deque(maxlen=300)
        self.global_ratio = 24.0
        self.upper_ratio = 36.0
        self.recent_losses = 0

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

        self._ingest_prev_info(prev_auctions, agent_id)

        my_state = states[agent_id]
        gold = float(my_state.get("gold", 0) or 0)
        points = float(my_state.get("points", 0) or 0)

        total_rounds = round + len(bank_state.get("gold_income_per_round", []))
        progress = 0.0 if total_rounds <= 1 else round / max(total_rounds - 1, 1)

        spend_frac = 0.30 + 0.25 * min(1.0, progress)
        if self._recent_win_rate() < 0.18:
            spend_frac *= 0.65
        spend_frac = min(spend_frac, 0.55)
        budget = gold * spend_frac
        reserve = gold - budget
        reserve = max(reserve, gold * 0.35)
        budget = max(0.0, gold - reserve)

        ev_map = [(aid, self._expected_value(cfg)) for aid, cfg in auctions.items()]
        ev_map.sort(key=lambda item: item[1], reverse=True)

        bids: Dict[str, int] = {}
        tiers = self._split_into_tiers(ev_map)
        remaining = budget
        for tier, items in tiers:
            if remaining <= 1:
                break
            tier_ratio = self._tier_ratio(tier, progress)
            per_item = max(1.0, (remaining * tier_ratio) / max(len(items), 1))
            for aid, ev in items:
                bid = self._bid_amount(ev, per_item)
                bid = min(bid, remaining)
                if bid < 1:
                    continue
                bid = self._jitter(bid)
                bids[aid] = bid
                remaining -= bid
                if remaining <= 1:
                    break

        # No pool buys: every previous attempt caused net point losses
        return {"bids": bids, "pool": 0}

    # ------------------------------------------------------------------
    def _ingest_prev_info(self, prev_auctions: Dict[str, Dict[str, Any]], agent_id: str) -> None:
        for info in prev_auctions.values():
            bids = info.get("bids") or []
            if not bids:
                continue
            win_bid = bids[0].get("gold", 0)
            reward = info.get("reward", 0)
            if isinstance(win_bid, (int, float)) and isinstance(reward, (int, float)) and reward > 0:
                ratio = win_bid / max(reward, 1)
                self.global_ratio = self._blend(self.global_ratio, ratio, 0.12)
                self.upper_ratio = max(self.upper_ratio * 0.98, self.global_ratio * 1.5)
            for pos, bid in enumerate(bids):
                if bid.get("a_id") == agent_id:
                    amount = int(bid.get("gold", 0) or 0)
                    self.bid_history.append(amount)
                    self.win_history.append(1 if pos == 0 else 0)
                    if pos == 0:
                        self.recent_losses = max(0, self.recent_losses - 1)
                    else:
                        self.recent_losses += 1
                    break

    def _split_into_tiers(self, ev_map: List[Tuple[str, float]]) -> List[Tuple[str, List[Tuple[str, float]]]]:
        if not ev_map:
            return []
        n = len(ev_map)
        top = ev_map[: max(1, n // 5)]
        mid = ev_map[max(1, n // 5): max(1, (2 * n) // 3)]
        low = ev_map[max(1, (2 * n) // 3):]
        return [("top", top), ("mid", mid), ("low", low)]

    def _tier_ratio(self, tier: str, progress: float) -> float:
        if tier == "top":
            return 0.5 if progress < 0.5 else 0.35
        if tier == "mid":
            return 0.35 if progress < 0.5 else 0.4
        return 0.15

    def _bid_amount(self, ev: float, base: float) -> int:
        # cap relative to observed gold per point and EV
        cap = ev * self.upper_ratio
        guess = ev * self.global_ratio
        bid = min(cap, max(ev * 12.0, guess, base))
        if self._recent_win_rate() < 0.15:
            bid *= 0.85
        if self.recent_losses > 8:
            bid *= 0.8
        return int(max(1.0, bid))

    def _jitter(self, bid: float) -> int:
        return max(1, int(bid * (1 + self.random.uniform(-0.015, 0.02))))

    def _recent_win_rate(self) -> float:
        if not self.bid_history:
            return 0.25
        return (sum(self.win_history) + 1) / (len(self.win_history) + 2)

    @staticmethod
    def _expected_value(auction: Dict[str, Any]) -> float:
        die = int(auction.get("die", 0) or 0)
        num = int(auction.get("num", 0) or 0)
        bonus = float(auction.get("bonus", 0) or 0)
        if die <= 0 or num <= 0:
            return bonus
        return ((die + 1) / 2) * num + bonus

    @staticmethod
    def _blend(current: float, new: float, weight: float) -> float:
        weight = max(0.0, min(1.0, weight))
        return current * (1 - weight) + new * weight


__all__ = ["BraavosLockdownAgent"]
