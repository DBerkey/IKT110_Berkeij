"""BraavosSentinelAgent: pool-disciplined successor to BraavosOptimal.

Key upgrades over BraavosOptimalAgent:
- Maintains a dynamic point floor so we never liquidate the score completely.
- Learns typical pool participation volume and only buys when ROI beats
  observed gold-per-point prices.
- Suppresses pool risk late game and when we already sit on above-median gold.
"""

from __future__ import annotations

import random
from collections import deque
from statistics import median
from typing import Any, Deque, Dict, Iterable, List, Tuple


class BraavosSentinelAgent:
    BUCKET_SIZE = 5
    MIN_RATIO = 8.0
    MAX_RATIO = 90.0

    MIN_POOL = 800
    LATE_GAME_BRAKE = 0.65

    def __init__(self, agent_id: str, seed: int | None = None) -> None:
        self.agent_id = agent_id
        self.random = random.Random(seed)
        self._bucket_price: Dict[int, Tuple[float, int]] = {}
        self._win_prices: Deque[float] = deque(maxlen=80)
        self._last_processed_round = -1
        self._global_ratio = 28.0
        self._gold_per_point_ratio = 24.0
        self._pool_volume_ema = 0.0

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

        points_array = self._points_snapshot(states.values())
        gold_array = [float(s.get("gold", 0) or 0) for s in states.values()] or [0.0]
        leader_points = max(points_array) if points_array else points
        median_points = median(points_array) if points_array else points
        median_gold = median(gold_array)
        deficit = max(0.0, median_points - points)
        lead = max(0.0, points - median_points)
        table_size = max(len(states), 1)

        total_rounds = round + len(bank_state.get("gold_income_per_round", []))
        progress = 0.0 if total_rounds <= 1 else round / max(total_rounds - 1, 1)

        interest_rate = (bank_state.get("bank_interest_per_round") or [1.0])[0]
        bank_limit = (bank_state.get("bank_limit_per_round") or [0])[0]

        points_floor = self._points_floor(points, median_points, leader_points, progress)

        spend_fraction = (
            0.30
            + 0.42 * progress
            + min(0.22, deficit / 40.0)
            - min(0.16, lead / 70.0)
        )
        if gold < median_gold * 0.8:
            spend_fraction += 0.08
        if gold > median_gold * 1.6:
            spend_fraction -= 0.12
        spend_fraction += self._income_pressure(bank_state)
        spend_fraction -= min(0.1, max(0.0, interest_rate - 1.0) * 2.5)
        spend_fraction = self._clamp(spend_fraction, 0.18, 0.95)

        reserve = gold * (1 - spend_fraction)
        interest_padding = (
            min(bank_limit, gold) * (0.07 + max(0.0, interest_rate - 1.0) * 1.8)
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
            priority_floor = 0.02 + 0.03 * progress

            for aid, ev, predicted, priority in auction_plan:
                if priority < priority_floor and ev < max_ev * 0.55:
                    continue
                if gold_available <= reserve:
                    break

                bid = self._determine_bid(
                    predicted=predicted,
                    ev=ev,
                    deficit=deficit,
                    lead=lead,
                    progress=progress,
                    table_size=table_size,
                    points_floor=points_floor,
                    points=points,
                )
                if bid <= 0:
                    continue

                max_affordable = gold_available - reserve
                if max_affordable <= 0:
                    break
                bid = min(bid, max_affordable)

                if bid < 1:
                    continue

                jitter = 1 + self.random.uniform(-0.02, 0.03)
                bid = int(max(1, bid * jitter))
                if bid == 0:
                    continue

                bids[aid] = bid
                gold_available -= bid

        total_prev_pool = sum(prev_pool_buys.values())
        self._pool_volume_ema = self._blend(
            self._pool_volume_ema or total_prev_pool,
            total_prev_pool,
            0.25,
        )

        pool_purchase = self._decide_pool_purchase(
            points=points,
            points_floor=points_floor,
            lead=lead,
            deficit=deficit,
            pool=pool,
            prev_pool_buys=prev_pool_buys,
            table_size=table_size,
            progress=progress,
            gold=gold,
            median_gold=median_gold,
        )

        return {"bids": bids, "pool": pool_purchase}

    # ------------------------------------------------------------------
    def _ingest_prev_round(self, prev_auctions: Dict[str, Dict[str, Any]], round_number: int) -> None:
        target_round = round_number - 1
        if target_round <= self._last_processed_round:
            return

        for info in prev_auctions.values():
            bids = info.get("bids") or []
            if not bids:
                continue

            win_bid = float(bids[0].get("gold", 0) or 0)
            if win_bid <= 0:
                continue

            ev = self._expected_points(info)
            if ev > 0:
                ratio = self._clamp(win_bid / max(ev, 1.0), self.MIN_RATIO, self.MAX_RATIO)
                self._global_ratio = self._blend(self._global_ratio, ratio, 0.12)

            reward = float(info.get("reward", 0) or 0)
            if reward > 0:
                gold_per_point = self._clamp(win_bid / reward, self.MIN_RATIO, self.MAX_RATIO)
                self._gold_per_point_ratio = self._blend(
                    self._gold_per_point_ratio, gold_per_point, 0.08
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
        baseline = self._clamp(baseline, ev * self.MIN_RATIO * 0.25, ev * self.MAX_RATIO)

        if bucket in self._bucket_price:
            avg, samples = self._bucket_price[bucket]
            if samples > 0:
                blend = min(0.85, samples / (samples + 4))
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
        points_floor: float,
        points: float,
    ) -> float:
        if predicted <= 0:
            return 0.0

        aggression = 1.1 + 0.32 * min(1.0, deficit / 30.0) - 0.15 * min(1.0, lead / 80.0)
        aggression += 0.1 * progress
        if table_size <= 4:
            aggression *= 0.92
        elif table_size >= 10:
            aggression *= 1.05

        if points < points_floor * 1.15:
            aggression *= 1.15

        bid = predicted * aggression
        floor = ev * max(6.0, 0.55 * self._global_ratio)
        cap = ev * self._global_ratio * (1.7 + 0.35 * progress)
        return max(floor, min(cap, bid))

    def _auction_priority(self, ev: float, predicted: float, max_ev: float, progress: float) -> float:
        if predicted <= 0:
            return 0.0
        density = ev / predicted
        normalized = ev / max(1.0, max_ev)
        priority = 0.6 * density + 0.4 * normalized
        return priority * (1 + 0.25 * progress)

    def _decide_pool_purchase(
        self,
        *,
        points: float,
        points_floor: float,
        lead: float,
        deficit: float,
        pool: int,
        prev_pool_buys: Dict[str, int],
        table_size: int,
        progress: float,
        gold: float,
        median_gold: float,
    ) -> int:
        if pool < self.MIN_POOL or points <= points_floor:
            return 0
        if lead < 150:
            return 0
        if progress > 0.9:
            return 0
        if progress > self.LATE_GAME_BRAKE and lead < 500:
            return 0
        if gold >= median_gold * 1.3:
            return 0

        expected_total = max(self._pool_volume_ema, table_size * 2.5)
        value_per_point = pool / max(1.0, expected_total)
        roi_gate = self._gold_per_point_ratio * (1.05 if deficit <= 0 else 1.25)
        if value_per_point < roi_gate:
            return 0

        surplus_points = max(0.0, points - points_floor)
        if surplus_points <= 0:
            return 0

        late_penalty = max(0.0, progress - self.LATE_GAME_BRAKE)
        max_ratio = max(0.08, 0.32 - late_penalty * 0.4)
        if deficit > 0:
            max_ratio *= 0.55

        spend_cap = min(surplus_points, points * max_ratio)
        spend = int(spend_cap)
        if spend <= 0:
            return 0

        keep_buffer = max(25, points_floor * 0.05)
        spend = min(spend, int(points - keep_buffer))
        return max(spend, 0)

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

    @staticmethod
    def _points_floor(points: float, median_points: float, leader_points: float, progress: float) -> float:
        base = max(200.0, 0.2 * leader_points, median_points * (0.35 + 0.25 * progress))
        return min(points, base)


__all__ = ["BraavosSentinelAgent"]
