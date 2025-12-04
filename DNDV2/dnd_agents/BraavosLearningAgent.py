"""BraavosLearningAgent

Learns clearing-price ratios on the fly. For roughly the first 100 rounds it uses
static fallbacks (spend a fixed fraction of gold, bid with conservative ratios).
As the game progresses it captures winning bid -> reward ratios in EV buckets and
updates bids purely from those observations, so it adapts to whatever meta or
interest pattern the server runs without depending on hard-coded constants.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Tuple


class BraavosLearningAgent:
    BUCKET_SIZE = 5
    TRAINING_ROUNDS = 100
    MAX_BUCKET_SAMPLES = 200

    def __init__(self, agent_id: str, seed: int | None = None) -> None:
        self.agent_id = agent_id
        self.random = random.Random(seed)
        self.bucket_history: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=self.MAX_BUCKET_SAMPLES))
        self.global_history: Deque[float] = deque(maxlen=500)
        self.round_counter = 0
        self.base_ratio = 22.0  # used only before we collect enough data

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

        self._ingest_prev(prev_auctions)
        self.round_counter += 1

        agent_state = states[agent_id]
        gold = float(agent_state.get("gold", 0) or 0)
        points = float(agent_state.get("points", 0) or 0)

        spend_fraction = self._spend_fraction(points, states.values())
        budget = gold * spend_fraction
        budget = max(0.0, budget)
        reserve = gold - budget
        reserve = max(reserve, gold * 0.25)
        budget = max(0.0, gold - reserve)

        ev_map = [(aid, self._expected_value(cfg)) for aid, cfg in auctions.items()]
        ev_map.sort(key=lambda item: item[1], reverse=True)

        bids: Dict[str, int] = {}
        remaining = budget
        for aid, ev in ev_map:
            if remaining <= 1:
                break
            ratio = self._predicted_ratio(ev)
            bid = ev * ratio
            bid = min(bid, remaining)
            if bid < 1:
                continue
            bid = self._jitter(bid)
            bids[aid] = bid
            remaining -= bid

        return {"bids": bids, "pool": 0}

    # ------------------------------------------------------------------
    def _ingest_prev(self, prev_auctions: Dict[str, Dict[str, Any]]) -> None:
        for info in prev_auctions.values():
            bids = info.get("bids") or []
            if not bids:
                continue
            reward = float(info.get("reward", 0) or 0)
            winning_bid = bids[0].get("gold", 0)
            if reward <= 0:
                continue
            ev = self._expected_value(info)
            if ev <= 0:
                continue
            ratio = float(winning_bid) / max(reward, 1.0)
            ratio = max(1.0, ratio)
            bucket = int(ev // self.BUCKET_SIZE) * self.BUCKET_SIZE
            self.bucket_history[bucket].append(ratio)
            self.global_history.append(ratio)

    def _predicted_ratio(self, ev: float) -> float:
        if self.round_counter < self.TRAINING_ROUNDS or not self.global_history:
            return self.base_ratio
        bucket = int(ev // self.BUCKET_SIZE) * self.BUCKET_SIZE
        samples = self.bucket_history.get(bucket)
        if samples:
            sorted_samples = sorted(samples)
            mid = len(sorted_samples) // 2
            bucket_ratio = sorted_samples[mid]
        else:
            sorted_global = sorted(self.global_history)
            mid = len(sorted_global) // 2
            bucket_ratio = sorted_global[mid]
        return max(6.0, bucket_ratio)

    def _spend_fraction(self, my_points: float, all_states: Any) -> float:
        points_list = [float(s.get("points", 0) or 0) for s in all_states]
        median_points = 0.0
        if points_list:
            sorted_pts = sorted(points_list)
            median_points = sorted_pts[len(sorted_pts) // 2]
        deficit = median_points - my_points
        base = 0.32
        if deficit > 0:
            base += min(0.2, deficit / 2000.0)
        else:
            base -= min(0.1, abs(deficit) / 4000.0)
        base = max(0.18, min(0.55, base))
        return base

    def _jitter(self, bid: float) -> int:
        return max(1, int(bid * (1 + self.random.uniform(-0.015, 0.02))))

    @staticmethod
    def _expected_value(auction: Dict[str, Any]) -> float:
        die = int(auction.get("die", 0) or 0)
        num = int(auction.get("num", 0) or 0)
        bonus = float(auction.get("bonus", 0) or 0)
        if die <= 0 or num <= 0:
            return bonus
        return ((die + 1) / 2) * num + bonus


__all__ = ["BraavosLearningAgent"]
