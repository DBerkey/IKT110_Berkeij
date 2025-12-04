"""BraavosLegionAgent: combines Braavos-style discipline with Abbathor inflation control.

Key ideas:
- Keep Braavos bucket-based market modelling, points floor safeguards, and selective pool sells.
- uses EMA-driven market cost-per-point tracker and wealth-pressure spending logic.
- Inflate bids when we are hoarding gold or when clearing prices spike far beyond historical ratios.
"""

from __future__ import annotations

import random
from collections import deque
from statistics import median
from typing import Any, Deque, Dict, Iterable, List, Tuple


class BraavosLegionAgent:
    BUCKET_SIZE = 5
    MIN_RATIO = 8.0
    MAX_RATIO = 130.0

    MIN_POOL = 800
    LATE_GAME_BRAKE = 0.72
    MAX_SINGLE_BID_RATIO = 0.45

    MARKET_CP_ALPHA = 0.18
    MARKET_CP_MIN = 6.0
    AGGRESSION_PRESSURE_MAX = 4.5

    def __init__(self, agent_id: str, seed: int | None = None) -> None:
        self.agent_id = agent_id
        self.random = random.Random(seed)
        self._bucket_price: Dict[int, Tuple[float, int]] = {}
        self._win_prices: Deque[float] = deque(maxlen=100)
        self._last_processed_round = -1
        self._global_ratio = 28.0
        self._gold_per_point_ratio = 24.0
        self._pool_volume_ema = 0.0
        self._market_cp = 25.0
        self._wealth_pressure = 1.0

    # ------------------------------------------------------------------
    def make_bid(
        self,
        agent_id: str,
        round_number: int,
        states: Dict[str, Dict[str, float]],
        auctions: Dict[str, Dict[str, Any]],
        prev_auctions: Dict[str, Dict[str, Any]],
        pool: int,
        prev_pool_buys: Dict[str, int],
        bank_state: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        if agent_id not in states:
            return {"bids": {}, "pool": 0}

        self._ingest_prev_round(prev_auctions, round_number)

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

        total_rounds = round_number + len(bank_state.get("gold_income_per_round", []))
        rounds_left = max(1, total_rounds - round_number)
        progress = 0.0 if total_rounds <= 1 else round_number / max(total_rounds - 1, 1)

        interest_rate = (bank_state.get("bank_interest_per_round") or [1.0])[0]
        bank_limit = (bank_state.get("bank_limit_per_round") or [0])[0]

        points_floor = self._points_floor(points, median_points, leader_points, progress)

        spend_fraction = (
            0.31
            + 0.45 * progress
            + min(0.24, deficit / 36.0)
            - min(0.17, lead / 80.0)
        )
        if gold < median_gold * 0.8:
            spend_fraction += 0.09
        if gold > median_gold * 1.7:
            spend_fraction -= 0.14
        spend_fraction += self._income_pressure(bank_state)
        spend_fraction -= min(0.12, max(0.0, interest_rate - 1.0) * 2.4)
        spend_fraction = self._clamp(spend_fraction, 0.2, 0.97)

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
            self._update_wealth_pressure(gold, ev_map, rounds_left, len(auctions))

            auction_plan: List[Tuple[str, float, float, float]] = []
            for aid, ev in ev_map.items():
                predicted = self._predict_price(ev)
                priority = self._auction_priority(ev, predicted, max_ev, progress)
                auction_plan.append((aid, ev, predicted, priority))

            auction_plan.sort(key=lambda entry: entry[3], reverse=True)
            priority_floor = 0.018 + 0.03 * progress

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

                if rounds_left > 12:
                    bid = min(bid, gold * self.MAX_SINGLE_BID_RATIO)
                bid = min(bid, max_affordable)

                if bid < 1:
                    continue

                jitter = 1 + self.random.uniform(-0.01, 0.07)
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

        ratios: List[float] = []
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
                self._global_ratio = self._blend(self._global_ratio, ratio, 0.13)

            reward = float(info.get("reward", 0) or 0)
            if reward > 0:
                gpp = self._clamp(win_bid / reward, self.MIN_RATIO, self.MAX_RATIO)
                self._gold_per_point_ratio = self._blend(
                    self._gold_per_point_ratio, gpp, 0.08
                )
                ratios.append(gpp)

            bucket = int(ev // self.BUCKET_SIZE) * self.BUCKET_SIZE
            avg, samples = self._bucket_price.get(bucket, (win_bid, 0))
            weight = 0.22 if samples else 1.0
            new_avg = self._blend(avg, win_bid, weight)
            self._bucket_price[bucket] = (new_avg, min(samples + 1, 180))
            self._win_prices.append(win_bid)

        if ratios:
            self._market_cp = self._blend(self._market_cp, median(ratios), self.MARKET_CP_ALPHA)
            self._market_cp = max(self.MARKET_CP_MIN, self._market_cp)

        self._last_processed_round = target_round

    def _predict_price(self, ev: float) -> float:
        if ev <= 0:
            return 0.0
        bucket = int(ev // self.BUCKET_SIZE) * self.BUCKET_SIZE
        baseline = self._global_ratio * ev
        baseline = self._clamp(baseline, ev * self.MIN_RATIO * 0.3, ev * self.MAX_RATIO)

        if bucket in self._bucket_price:
            avg, samples = self._bucket_price[bucket]
            if samples > 0:
                blend = min(0.88, samples / (samples + 4))
                baseline = avg * blend + baseline * (1 - blend)

        return max(10.0, baseline)

    def _determine_bid(
        self,
        *,
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

        aggression = 1.08 + 0.33 * min(1.0, deficit / 32.0) - 0.15 * min(1.0, lead / 85.0)
        aggression += 0.12 * progress
        if table_size <= 4:
            aggression *= 0.9
        elif table_size >= 10:
            aggression *= 1.05

        if points < points_floor * 1.1:
            aggression *= 1.12
        if progress > self.LATE_GAME_BRAKE and deficit > 0:
            aggression *= 1.15

        bid = predicted * aggression * self._wealth_pressure
        floor = ev * max(6.2, 0.58 * self._global_ratio)
        cap = ev * self._global_ratio * (1.9 + 0.45 * progress)
        return max(floor, min(cap, bid))

    def _auction_priority(self, ev: float, predicted: float, max_ev: float, progress: float) -> float:
        if predicted <= 0:
            return 0.0
        density = ev / predicted
        normalized = ev / max(1.0, max_ev)
        priority = 0.6 * density + 0.4 * normalized
        return priority * (1 + 0.27 * progress)

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
        if progress > 0.92:
            return 0
        if progress > self.LATE_GAME_BRAKE and lead < 500:
            return 0
        if gold >= median_gold * 1.3:
            return 0

        expected_total = max(self._pool_volume_ema, table_size * 2.5)
        value_per_point = pool / max(1.0, expected_total)
        roi_gate = self._gold_per_point_ratio * (1.05 if deficit <= 0 else 1.3)
        if value_per_point < roi_gate:
            return 0

        surplus_points = max(0.0, points - points_floor)
        if surplus_points <= 0:
            return 0

        late_penalty = max(0.0, progress - self.LATE_GAME_BRAKE)
        max_ratio = max(0.08, 0.28 - late_penalty * 0.4)
        if deficit > 0:
            max_ratio *= 0.5

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
        future_window = incomes[1 : min(6, len(incomes))]
        if not future_window:
            return 0.0
        avg_future = sum(future_window) / len(future_window)
        if avg_future <= 0:
            return 0.0
        delta = (current - avg_future) / avg_future
        return self._clamp(delta / 3.0, -0.16, 0.16)

    @staticmethod
    def _points_floor(points: float, median_points: float, leader_points: float, progress: float) -> float:
        base = max(220.0, 0.22 * leader_points, median_points * (0.36 + 0.27 * progress))
        return min(points, base)

    def _update_wealth_pressure(
        self,
        gold: float,
        ev_map: Dict[str, float],
        rounds_left: int,
        auction_count: int,
    ) -> None:
        board_ev = sum(ev_map.values())
        if rounds_left <= 0 or board_ev <= 0:
            self._wealth_pressure = 1.0
            return

        target_burn = gold / rounds_left
        avg_clear = (board_ev * self._market_cp) / max(auction_count, 1)
        if avg_clear <= 0:
            self._wealth_pressure = 1.0
            return

        ratio = target_burn / max(1.0, avg_clear)
        self._wealth_pressure = self._clamp(1.0 + (ratio - 1.0) * 0.85, 0.85, self.AGGRESSION_PRESSURE_MAX)


__all__ = ["BraavosLegionAgent"]
