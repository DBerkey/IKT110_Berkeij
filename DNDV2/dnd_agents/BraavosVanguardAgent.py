"""BraavosVanguardAgent: crisis-responsive successor to the Braavos line.

Key ideas borrowed from the existing Braavos bots but tightened around their
observed issues in logs 20-23:
- Detects runaway leaders early and raises its spend floor so we stop hoarding
  30k+ gold while trailing (BraavosOptimal's main failure).
- Tracks realised gold-per-point ratios and bids above them whenever we are in
  a chase state so Phoenix/Sentinel level stalls do not happen.
- Converts excess points into pool shares only when ROI clearly beats the
  observed conversion rate and we have a scoreboard cushion.
"""

from __future__ import annotations

import random
from collections import deque
from statistics import median
from typing import Any, Deque, Dict, Iterable, List, Tuple


class BraavosVanguardAgent:
    BUCKET_SIZE = 4
    MIN_RATIO = 6.0
    MAX_RATIO = 110.0
    CHASE_DEFICIT = 2500.0
    PANIC_GOLD = 18000.0
    MIN_POOL = 600

    def __init__(self, agent_id: str, seed: int | None = None) -> None:
        self.agent_id = agent_id
        self.random = random.Random(seed)
        self._bucket_price: Dict[int, Tuple[float, int]] = {}
        self._win_prices: Deque[float] = deque(maxlen=120)
        self._last_processed_round = -1
        self._global_ratio = 26.0
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
        deficit = max(0.0, leader_points - points)
        lead = max(0.0, points - median_points)
        table_size = max(len(states), 1)

        total_rounds = round + len(bank_state.get("gold_income_per_round", []))
        progress = 0.0 if total_rounds <= 1 else round / max(total_rounds - 1, 1)

        interest_rate = (bank_state.get("bank_interest_per_round") or [1.0])[0]
        bank_limit = (bank_state.get("bank_limit_per_round") or [0])[0]

        points_floor = self._points_floor(points, median_points, leader_points, progress)
        chasing = deficit > self.CHASE_DEFICIT or (leader_points > 0 and points < leader_points * 0.65)
        score_lead = points - leader_points

        spend_fraction = 0.34 + 0.27 * progress
        spend_fraction += self._clamp(deficit / max(leader_points, 1.0), 0.0, 0.35)
        spend_fraction -= min(0.2, lead / max(points + 1.0, 1.0) * 0.35)
        if chasing:
            spend_fraction = max(spend_fraction, 0.68)
        if gold > median_gold * 1.4 or gold > self.PANIC_GOLD:
            spend_fraction += 0.08
        if gold < median_gold * 0.7:
            spend_fraction -= 0.06
        if progress > 0.82:
            spend_fraction = max(spend_fraction, 0.78)
        spend_fraction += self._income_pressure(bank_state)
        spend_fraction -= min(0.12, max(0.0, interest_rate - 1.0) * 0.3)
        spend_fraction = self._clamp(spend_fraction, 0.28, 0.95)

        reserve = gold * (1 - spend_fraction)
        interest_padding = min(bank_limit, gold) * (0.05 + max(0.0, interest_rate - 1.0))
        reserve = max(reserve, interest_padding, gold * (0.05 if chasing else 0.12))
        reserve = min(reserve, gold)

        bids: Dict[str, int] = {}
        gold_available = gold

        if auctions and gold_available > 0:
            ev_map = {aid: self._expected_points(cfg) for aid, cfg in auctions.items()}
            max_ev = max(ev_map.values()) if ev_map else 0.0
            auction_plan: List[Tuple[str, float, float, float]] = []
            for aid, ev in ev_map.items():
                predicted = self._predict_price(ev)
                priority = self._auction_priority(ev, predicted, max_ev, deficit, progress)
                auction_plan.append((aid, ev, predicted, priority))

            auction_plan.sort(key=lambda entry: entry[3], reverse=True)
            priority_floor = 0.015 + 0.04 * progress
            ev_floor = 22.0 if not chasing else 32.0
            max_targets = 6 if chasing else 4
            max_targets += 2 if progress > 0.75 else 0
            taken = 0

            for aid, ev, predicted, priority in auction_plan:
                if taken >= max_targets:
                    break
                if priority < priority_floor and (not chasing or ev < max_ev * 0.55):
                    continue
                if ev < ev_floor:
                    continue
                if gold_available <= reserve:
                    break

                bid = self._determine_bid(
                    predicted=predicted,
                    ev=ev,
                    deficit=deficit,
                    lead=lead,
                    progress=progress,
                    chasing=chasing,
                )
                if bid <= 0:
                    continue

                spend_cap = gold_available - reserve
                if spend_cap <= 0:
                    break
                bid = min(bid, spend_cap)

                if bid < 1:
                    continue

                jitter = 1 + self.random.uniform(-0.015, 0.035)
                bid = int(max(1, bid * jitter))
                bids[aid] = bid
                gold_available -= bid
                taken += 1

        total_prev_pool = sum(prev_pool_buys.values())
        self._pool_volume_ema = self._blend(
            self._pool_volume_ema or total_prev_pool or table_size * 2,
            max(total_prev_pool, 1),
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
            leader_points=leader_points,
            score_lead=score_lead,
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
                self._global_ratio = self._blend(self._global_ratio, ratio, 0.1)

            reward = float(info.get("reward", 0) or 0)
            if reward > 0:
                gold_per_point = self._clamp(win_bid / reward, self.MIN_RATIO, self.MAX_RATIO)
                self._gold_per_point_ratio = self._blend(self._gold_per_point_ratio, gold_per_point, 0.08)

            bucket = int(ev // self.BUCKET_SIZE) * self.BUCKET_SIZE
            avg, samples = self._bucket_price.get(bucket, (win_bid, 0))
            weight = 0.18 if samples else 1.0
            new_avg = self._blend(avg, win_bid, weight)
            self._bucket_price[bucket] = (new_avg, min(samples + 1, 200))
            self._win_prices.append(win_bid)

        self._last_processed_round = target_round

    def _predict_price(self, ev: float) -> float:
        if ev <= 0:
            return 0.0
        bucket = int(ev // self.BUCKET_SIZE) * self.BUCKET_SIZE
        baseline = self._global_ratio * ev
        baseline = self._clamp(baseline, ev * self.MIN_RATIO * 0.4, ev * self.MAX_RATIO)

        if bucket in self._bucket_price:
            avg, samples = self._bucket_price[bucket]
            if samples > 0:
                blend = min(0.9, samples / (samples + 6))
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
        chasing: bool,
    ) -> float:
        if predicted <= 0 or ev <= 0:
            return 0.0

        aggression = 1.05 + 0.25 * progress
        aggression += self._clamp(deficit / 6000.0, 0.0, 0.45)
        aggression -= min(0.18, lead / 8000.0)
        if chasing:
            aggression += 0.25

        bid = predicted * aggression
        floor = ev * max(6.0, self._global_ratio * (0.35 + 0.25 * progress))
        cap = ev * self._global_ratio * (1.8 + 0.4 * progress + (0.25 if chasing else 0.0))
        return max(floor, min(cap, bid))

    def _auction_priority(
        self,
        ev: float,
        predicted: float,
        max_ev: float,
        deficit: float,
        progress: float,
    ) -> float:
        if predicted <= 0:
            return 0.0
        density = ev / predicted
        normalized = ev / max(1.0, max_ev)
        deficit_weight = self._clamp(deficit / 8000.0, 0.0, 0.5)
        return (0.5 * density + 0.4 * normalized + deficit_weight * 0.3) * (1 + 0.2 * progress)

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
        leader_points: float,
        score_lead: float,
    ) -> int:
        if pool < self.MIN_POOL:
            return 0
        if points <= points_floor:
            return 0
        if deficit > 0:
            return 0
        if score_lead < 800:
            return 0
        if progress > 0.9:
            return 0
        if gold < median_gold * 0.6:
            return 0

        expected_total = max(self._pool_volume_ema, sum(prev_pool_buys.values()), table_size * 2.0)
        value_per_point = pool / max(1.0, expected_total)
        roi_gate = self._gold_per_point_ratio * 0.85
        if value_per_point < roi_gate:
            return 0

        surplus_points = max(0.0, points - points_floor)
        spend_ratio = 0.35 if score_lead > 2000 else 0.18
        spend_ratio *= self._clamp(1.0 - progress, 0.4, 1.0)
        spend_points = min(surplus_points, points * spend_ratio)
        spend_points = min(spend_points, max(points - points_floor * 1.05, 0.0))
        return max(int(spend_points), 0)

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
        base = max(300.0, 0.18 * leader_points, median_points * (0.4 + 0.3 * progress))
        return min(points, base)


__all__ = ["BraavosVanguardAgent"]
