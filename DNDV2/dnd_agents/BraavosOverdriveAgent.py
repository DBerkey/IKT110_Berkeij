"""BraavosOverdriveAgent: pushes the Braavos line into sustained mid/late aggression.

Key adjustments based on logs 20-25:
- Detects deteriorating point velocity and scoreboard deficits to trigger an
  overdrive state that spends down reserves aggressively instead of hoarding.
- Limits pool conversions to windows where we are comfortably ahead, so we do
  not lock points while chasing interest-focused bots.
- Raises the minimum bid floor in sprint time (last ~25% of rounds) to ensure
  every high-EV item is contested even if it requires overspending.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Tuple


class BraavosOverdriveAgent:
    PRICE_BUCKET = 5
    MIN_RATIO = 5.0
    MAX_RATIO = 120.0
    DEFICIT_SURGE = 4500.0
    TREND_WINDOW = 28
    LATE_PROGRESS = 0.75
    SPRINT_PROGRESS = 0.85
    POOL_COOLDOWN = 32
    POOL_COOLDOWN_STRESSED = 70

    def __init__(self, agent_id: str, seed: int | None = None) -> None:
        self.agent_id = agent_id
        self.random = random.Random(seed)
        self._bucket_prices: Dict[int, Tuple[float, int]] = {}
        self._market_ratio = 28.0
        self._gold_per_point = 24.0
        self._pool_volume_ema = 0.0
        self._last_processed_round = -1
        self._points_track: Deque[Tuple[int, float]] = deque(maxlen=self.TREND_WINDOW)
        self._leader_track: Deque[Tuple[int, float]] = deque(maxlen=self.TREND_WINDOW)
        self._last_pool_round = -10**6

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

        point_values = self._points_snapshot(states.values())
        gold_values = [float(s.get("gold", 0) or 0) for s in states.values()] or [0.0]
        leader_points = max(point_values) if point_values else points
        median_points = self._median(point_values) if point_values else points
        median_gold = self._median(gold_values)
        deficit = max(0.0, leader_points - points)
        lead = max(0.0, points - median_points)
        table_size = max(1, len(states))

        total_rounds = round + len(bank_state.get("gold_income_per_round", []))
        progress = 0.0 if total_rounds <= 1 else round / max(total_rounds - 1, 1)

        self._update_trends(round, points, leader_points)
        trend_gap = self._trend_gap()

        chasing = deficit > 0
        overdrive = deficit > self.DEFICIT_SURGE or (progress >= self.LATE_PROGRESS and chasing)
        overdrive = overdrive or trend_gap > 6.0
        sprint = progress >= self.SPRINT_PROGRESS or deficit > 7000.0

        spend_fraction = 0.38 + 0.32 * progress
        spend_fraction += self._clamp(deficit / max(leader_points, 1.0), -0.15, 0.35)
        spend_fraction += self._clamp(trend_gap / 16.0, -0.08, 0.18)
        spend_fraction -= min(0.16, lead / max(points + 1.0, 1.0) * 0.3)
        spend_fraction += self._income_pressure(bank_state)
        spend_fraction = self._clamp(spend_fraction, 0.3, 0.92)
        if overdrive:
            spend_fraction = max(spend_fraction, 0.85)
        if sprint:
            spend_fraction = max(spend_fraction, 0.94)

        reserve = gold * (1 - spend_fraction)
        min_reserve = gold * (0.07 if overdrive else 0.12)
        reserve = max(min_reserve, min(reserve, gold))

        bids: Dict[str, int] = {}
        gold_available = gold

        if auctions and gold_available > 0:
            ev_map = {aid: self._expected_points(cfg) for aid, cfg in auctions.items()}
            max_ev = max(ev_map.values()) if ev_map else 0.0
            plan: List[Tuple[str, float, float, float]] = []
            for aid, ev in ev_map.items():
                predicted = self._predict_price(ev)
                priority = self._auction_priority(
                    ev=ev,
                    predicted=predicted,
                    max_ev=max_ev,
                    deficit=deficit,
                    trend_gap=trend_gap,
                    progress=progress,
                )
                plan.append((aid, ev, predicted, priority))

            plan.sort(key=lambda entry: entry[3], reverse=True)
            ev_floor = 22.0 + 8.0 * progress
            if chasing:
                ev_floor = max(ev_floor, 30.0)
            if overdrive:
                ev_floor = max(ev_floor, 36.0)

            priority_floor = 0.02 + 0.05 * progress
            max_targets = 4 + (1 if progress > 0.45 else 0) + (1 if progress > 0.7 else 0)
            if sprint:
                max_targets += 2

            taken = 0
            for aid, ev, predicted, priority in plan:
                if taken >= max_targets:
                    break
                if ev < ev_floor or priority < priority_floor:
                    continue
                if gold_available <= reserve:
                    break

                bid = self._determine_bid(
                    predicted=predicted,
                    ev=ev,
                    deficit=deficit,
                    lead=lead,
                    progress=progress,
                    overdrive=overdrive,
                    sprint=sprint,
                )
                if bid <= 0:
                    continue

                spend_cap = gold_available - reserve
                if spend_cap <= 0:
                    break

                bid = min(bid, spend_cap)
                if bid < 1:
                    continue

                jitter = 1 + self.random.uniform(-0.02, 0.04)
                bids[aid] = int(max(1, bid * jitter))
                gold_available -= bids[aid]
                taken += 1

        total_prev_pool = sum(prev_pool_buys.values())
        self._pool_volume_ema = self._blend(
            self._pool_volume_ema or total_prev_pool or table_size * 2,
            max(total_prev_pool, 1),
            0.25,
        )

        pool_purchase = self._pool_allocation(
            points=points,
            deficit=deficit,
            lead=lead,
            pool=pool,
            progress=progress,
            gold=gold,
            median_gold=median_gold,
            overdrive=overdrive,
            sprint=sprint,
            score_floor=self._points_floor(points, median_points, leader_points, progress),
            table_size=table_size,
            prev_pool_buys=prev_pool_buys,
            round_number=round,
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
                self._market_ratio = self._blend(self._market_ratio, ratio, 0.12)

            reward = float(info.get("reward", 0) or 0)
            if reward > 0:
                gold_per_point = self._clamp(win_bid / reward, self.MIN_RATIO, self.MAX_RATIO)
                self._gold_per_point = self._blend(self._gold_per_point, gold_per_point, 0.08)

            bucket = int(ev // self.PRICE_BUCKET) * self.PRICE_BUCKET
            avg, samples = self._bucket_prices.get(bucket, (win_bid, 0))
            weight = 0.2 if samples else 1.0
            self._bucket_prices[bucket] = (self._blend(avg, win_bid, weight), min(samples + 1, 200))

        self._last_processed_round = target_round

    def _predict_price(self, ev: float) -> float:
        if ev <= 0:
            return 0.0
        bucket = int(ev // self.PRICE_BUCKET) * self.PRICE_BUCKET
        baseline = self._market_ratio * ev
        baseline = self._clamp(baseline, ev * self.MIN_RATIO * 0.4, ev * self.MAX_RATIO)
        if bucket in self._bucket_prices:
            avg, samples = self._bucket_prices[bucket]
            blend = min(0.9, samples / (samples + 5))
            baseline = avg * blend + baseline * (1 - blend)
        return max(6.0, baseline)

    def _determine_bid(
        self,
        *,
        predicted: float,
        ev: float,
        deficit: float,
        lead: float,
        progress: float,
        overdrive: bool,
        sprint: bool,
    ) -> float:
        if predicted <= 0 or ev <= 0:
            return 0.0

        aggression = 1.05 + 0.3 * progress
        aggression += self._clamp(deficit / 6000.0, -0.05, 0.45)
        aggression -= min(0.2, lead / 8000.0)
        if overdrive:
            aggression += 0.35
        if sprint:
            aggression += 0.2

        bid = predicted * aggression
        floor = ev * max(6.0, self._market_ratio * (0.5 + 0.3 * progress))
        if overdrive:
            floor = max(floor, ev * self._market_ratio * 0.9)
        cap = ev * self._market_ratio * (1.8 + 0.35 * progress + (0.25 if sprint else 0.0))
        return max(floor, min(cap, bid))

    def _auction_priority(
        self,
        *,
        ev: float,
        predicted: float,
        max_ev: float,
        deficit: float,
        trend_gap: float,
        progress: float,
    ) -> float:
        if predicted <= 0:
            return 0.0
        density = ev / predicted
        normalized = ev / max(1.0, max_ev)
        deficit_weight = self._clamp(deficit / 9000.0, 0.0, 0.6)
        trend_bias = self._clamp(trend_gap / 18.0, -0.05, 0.25)
        return (0.5 * density + 0.4 * normalized + deficit_weight * 0.3 + trend_bias) * (1 + 0.2 * progress)

    def _pool_allocation(
        self,
        *,
        points: float,
        deficit: float,
        lead: float,
        pool: int,
        progress: float,
        gold: float,
        median_gold: float,
        overdrive: bool,
        sprint: bool,
        score_floor: float,
        table_size: int,
        prev_pool_buys: Dict[str, int],
        round_number: int,
    ) -> int:
        if pool < 400 or points <= score_floor or progress >= 0.85:
            return 0
        if lead < 600 and deficit <= 0:
            return 0
        if gold < median_gold * 0.8:
            return 0
        if deficit > 12000:
            return 0

        stressful = overdrive or sprint or deficit > 0
        cooldown = self.POOL_COOLDOWN_STRESSED if stressful else self.POOL_COOLDOWN
        if round_number - self._last_pool_round < cooldown:
            return 0

        expected_total = max(self._pool_volume_ema, sum(prev_pool_buys.values()), table_size * 2.0)
        value_per_point = pool / max(1.0, expected_total)
        roi_gate = self._gold_per_point * (1.15 if stressful else 0.95)
        if value_per_point < roi_gate:
            return 0

        surplus = points - score_floor
        budget_ratio = 0.22 if lead > 2000 else 0.15
        if stressful:
            budget_ratio *= 0.55
        budget_ratio *= self._clamp(1.0 - progress, 0.4, 1.0)
        spend_points = min(points * budget_ratio, surplus)
        spend_points = int(max(spend_points, 0))
        if spend_points > 0:
            self._last_pool_round = round_number
        return spend_points

    # ------------------------------------------------------------------
    def _update_trends(self, round_number: int, points: float, leader_points: float) -> None:
        self._points_track.append((round_number, points))
        self._leader_track.append((round_number, leader_points))

    def _trend_gap(self) -> float:
        def slope(track: Deque[Tuple[int, float]]) -> float:
            if len(track) < 2:
                return 0.0
            start_round, start_points = track[0]
            end_round, end_points = track[-1]
            delta_round = max(1, end_round - start_round)
            return (end_points - start_points) / delta_round

        return slope(self._leader_track) - slope(self._points_track)

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
    def _median(values: List[float]) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2:
            return sorted_vals[mid]
        return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])

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
        base = max(400.0, 0.2 * leader_points, median_points * (0.45 + 0.3 * progress))
        return min(points, base)


__all__ = ["BraavosOverdriveAgent"]
