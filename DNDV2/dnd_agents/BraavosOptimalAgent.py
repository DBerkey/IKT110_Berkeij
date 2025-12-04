"""BraavosOptimalAgent: adaptive bidder tuned for the auction house meta.

The agent tracks historical clearing prices per expected-value bucket, adjusts its
spending rate based on bank forecasts, gold interest, table size, and current
rank position, and opportunistically purchases pool gold only when the expected
return per spent point is favorable."""

from __future__ import annotations

import random
from collections import deque
from statistics import median
from typing import Any, Deque, Dict, Iterable, List, Tuple


class BraavosOptimalAgent:
    """Adaptive agent that estimates fair prices and reacts to table dynamics."""

    BUCKET_SIZE = 5
    MIN_RATIO = 8.0
    MAX_RATIO = 90.0

    def __init__(self, agent_id: str, seed: int | None = None) -> None:
        self.agent_id = agent_id
        self.random = random.Random(seed)
        self._bucket_price: Dict[int, Tuple[float, int]] = {}
        self._win_prices: Deque[float] = deque(maxlen=80)
        self._last_processed_round = -1
        self._global_ratio = 28.0  # fallback gold per expected point
        self._gold_per_point_ratio = 24.0  # observed cost of one actual point

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def make_bid(
        self,
        agent_id: str,
        round: int,
        states: Dict[str, Dict[str, float]],
        auctions: Dict[str, Dict[str, Any]],
        prev_auctions: Dict[str, Dict[str, Any]],
        pool: int,
        prev_pool_buys: Dict[str, int],
        bank_state: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        if agent_id not in states:
            return {"bids": {}, "pool": 0}

        self._ingest_prev_round(prev_auctions, round)

        my_state = states[agent_id]
        gold = float(my_state.get("gold", 0) or 0)
        points = float(my_state.get("points", 0) or 0)

        total_rounds = round + len(bank_state.get("gold_income_per_round", []))
        progress = 0.0 if total_rounds <= 1 else round / max(total_rounds - 1, 1)

        median_points = median(self._points_snapshot(states.values()))
        deficit = max(0.0, median_points - points)
        lead = max(0.0, points - median_points)
        table_size = max(len(states), 1)

        interest_rate = (bank_state.get("bank_interest_per_round") or [1.0])[0]
        bank_limit = (bank_state.get("bank_limit_per_round") or [0])[0]

        spend_fraction = (
            0.32
            + 0.45 * progress
            + min(0.25, deficit / 35.0)
            - min(0.18, lead / 60.0)
            + self._income_pressure(bank_state)
        )
        spend_fraction -= min(0.12, max(0.0, interest_rate - 1.0) * 3.0)
        spend_fraction = self._clamp(spend_fraction, 0.18, 0.94)

        reserve = gold * (1 - spend_fraction)
        interest_padding = (
            min(bank_limit, gold) * (0.05 + max(0.0, interest_rate - 1.0) * 1.6)
        )
        reserve = max(reserve, interest_padding)
        reserve = min(reserve, gold)

        bids: Dict[str, int] = {}
        gold_available = gold

        if auctions and gold_available > 0:
            ev_map = {aid: self._expected_points(cfg) for aid, cfg in auctions.items()}
            max_ev = max(ev_map.values()) if ev_map else 0.0
            auction_plan: List[Tuple[str, float, float, float]] = []
            for aid, ev in ev_map.items():
                predicted = self._predict_price(ev)
                priority = self._auction_priority(ev, predicted, max_ev, progress)
                auction_plan.append((aid, ev, predicted, priority))

            auction_plan.sort(key=lambda entry: entry[3], reverse=True)
            priority_floor = 0.01 + 0.02 * progress

            for aid, ev, predicted, priority in auction_plan:
                if priority < priority_floor and ev < max_ev * 0.6:
                    continue
                if gold_available <= reserve:
                    break

                bid = self._determine_bid(
                    predicted,
                    ev,
                    deficit,
                    lead,
                    progress,
                    table_size,
                )
                if bid <= 0:
                    continue

                max_affordable = gold_available - reserve
                if max_affordable <= 0:
                    break
                bid = min(bid, gold_available)
                if bid > max_affordable:
                    bid = max_affordable

                if bid < 1:
                    continue

                jitter = 1 + self.random.uniform(-0.025, 0.035)
                bid = int(max(1, bid * jitter))
                if bid == 0:
                    continue

                bids[aid] = bid
                gold_available -= bid

        pool_purchase = self._decide_pool_purchase(
            points=points,
            deficit=deficit,
            pool=pool,
            prev_pool_buys=prev_pool_buys,
            table_size=table_size,
            progress=progress,
        )

        return {"bids": bids, "pool": pool_purchase}

    # ------------------------------------------------------------------
    # Strategic helpers
    # ------------------------------------------------------------------
    def _ingest_prev_round(self, prev_auctions: Dict[str, Dict[str, Any]], round_number: int) -> None:
        target_round = round_number - 1
        if target_round <= self._last_processed_round:
            return

        for info in prev_auctions.values():
            bids = info.get("bids") or []
            if not bids:
                continue

            winner = bids[0]
            win_bid = float(winner.get("gold", 0) or 0)
            if win_bid <= 0:
                continue

            ev = self._expected_points(info)
            if ev > 0:
                ratio = self._clamp(win_bid / max(ev, 1.0), self.MIN_RATIO, self.MAX_RATIO)
                self._global_ratio = self._blend(self._global_ratio, ratio, 0.15)

            reward = float(info.get("reward", 0) or 0)
            if reward > 0:
                gold_per_point = self._clamp(win_bid / reward, self.MIN_RATIO, self.MAX_RATIO)
                self._gold_per_point_ratio = self._blend(
                    self._gold_per_point_ratio, gold_per_point, 0.10
                )

            bucket = int(ev // self.BUCKET_SIZE) * self.BUCKET_SIZE
            avg, samples = self._bucket_price.get(bucket, (win_bid, 0))
            weight = 0.2 if samples else 1.0
            new_avg = self._blend(avg, win_bid, weight)
            self._bucket_price[bucket] = (new_avg, min(samples + 1, 160))
            self._win_prices.append(win_bid)

        self._last_processed_round = target_round

    def _predict_price(self, ev: float) -> float:
        if ev <= 0:
            return 0.0
        bucket = int(ev // self.BUCKET_SIZE) * self.BUCKET_SIZE
        baseline = self._global_ratio * ev
        baseline = self._clamp(baseline, ev * self.MIN_RATIO * 0.2, ev * self.MAX_RATIO)

        if bucket in self._bucket_price:
            avg, samples = self._bucket_price[bucket]
            if samples > 0:
                blend = min(0.85, samples / (samples + 3))
                baseline = avg * blend + baseline * (1 - blend)

        return max(8.0, baseline)

    def _determine_bid(
        self,
        predicted: float,
        ev: float,
        deficit: float,
        lead: float,
        progress: float,
        table_size: int,
    ) -> float:
        if predicted <= 0:
            return 0.0

        aggression = 1.05 + 0.35 * min(1.0, deficit / 25.0) - 0.18 * min(1.0, lead / 25.0)
        aggression += 0.12 * progress
        if table_size <= 4:
            aggression *= 0.9
        elif table_size >= 10:
            aggression *= 1.05

        bid = predicted * aggression
        floor = ev * max(6.0, 0.5 * self._global_ratio)
        cap = ev * self._global_ratio * (1.8 + 0.4 * progress)
        bid = max(floor, min(cap, bid))
        return bid

    def _auction_priority(self, ev: float, predicted: float, max_ev: float, progress: float) -> float:
        if predicted <= 0:
            return 0.0
        density = ev / predicted
        normalized = ev / max(1.0, max_ev)
        priority = 0.65 * density + 0.35 * normalized
        return priority * (1 + 0.2 * progress)

    def _decide_pool_purchase(
        self,
        points: float,
        deficit: float,
        pool: int,
        prev_pool_buys: Dict[str, int],
        table_size: int,
        progress: float,
    ) -> int:
        if pool <= 0 or points <= 0:
            return 0

        prev_total = sum(prev_pool_buys.values())
        expected_total = max(prev_total, table_size * 2)
        gold_per_point = pool / max(1, expected_total)

        threshold = self._gold_per_point_ratio * (1.15 if deficit > 0 else 1.35)
        if pool >= 4000:
            threshold *= 0.85
        if gold_per_point < threshold:
            return 0

        urgency = deficit + progress * 8.0
        spend_ratio = min(0.55, 0.2 + urgency / 45.0)
        if deficit <= 0:
            spend_ratio *= 0.5
        if pool > 4000:
            spend_ratio = max(spend_ratio, 0.18)

        spend = int(points * spend_ratio)
        keep_floor = 8 if deficit <= 0 else 4
        spend = min(spend, max(0, int(points - keep_floor)))
        return max(spend, 0)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _expected_points(auction: Dict[str, Any]) -> float:
        if "dice" in auction:
            dice = auction.get("dice") or []
            base = sum((sides + 1) / 2 for sides in dice)
            return base + float(auction.get("bonus", 0) or 0)

        die = int(auction.get("die", 0) or 0)
        num = int(auction.get("num", 0) or 0)
        bonus = float(auction.get("bonus", 0) or 0)
        if die <= 0 or num <= 0:
            return bonus
        return ((die + 1) / 2) * num + bonus

    @staticmethod
    def _points_snapshot(states: Iterable[Dict[str, float]]) -> List[float]:
        return [float(state.get("points", 0) or 0) for state in states] or [0.0]

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _blend(current: float, new: float, weight: float) -> float:
        weight = max(0.0, min(1.0, weight))
        return current * (1 - weight) + new * weight

    def _income_pressure(self, bank_state: Dict[str, List[float]]) -> float:
        incomes = bank_state.get("gold_income_per_round") or []
        if len(incomes) < 2:
            return 0.0
        current = incomes[0]
        future_window = incomes[1 : min(5, len(incomes))]
        if not future_window:
            return 0.0
        avg_future = sum(future_window) / len(future_window)
        if avg_future <= 0:
            return 0.0
        delta = (current - avg_future) / avg_future
        return self._clamp(delta / 3.0, -0.15, 0.15)


__all__ = ["BraavosOptimalAgent"]
