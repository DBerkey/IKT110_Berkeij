"""BraavosPhoenixAgent: momentum-aware evolution of BraavosSentinel.

Differences from the Sentinel agent:
- Tracks rolling win rate and point stagnation; if both slump (like the round-600
  plateau in log 14) the agent enters *burst mode* which lowers reserve targets,
  bids on more auctions, and raises aggression until momentum returns.
- Introduces a gold-recovery pool policy: when burst mode is active or cash
  drops below table median, surplus points are selectively traded for pool gold
  (if the pool's gold/point price beats observed auction ROI).
- Maintains the EV bucket price model but with faster adaptation so clearing
  price spikes are followed more quickly.
"""

from __future__ import annotations

import random
from collections import deque
from statistics import median
from typing import Any, Deque, Dict, Iterable, List, Tuple


class BraavosPhoenixAgent:
    BUCKET_SIZE = 5
    MIN_RATIO = 8.0
    MAX_RATIO = 90.0

    BURST_STALL_ROUNDS = 40
    BURST_WINRATE = 0.18
    MIN_POOL = 1200

    def __init__(self, agent_id: str, seed: int | None = None) -> None:
        self.agent_id = agent_id
        self.random = random.Random(seed)
        self._bucket_price: Dict[int, Tuple[float, int]] = {}
        self._win_prices: Deque[float] = deque(maxlen=120)
        self._last_processed_round = -1
        self._global_ratio = 27.0
        self._gold_per_point_ratio = 23.0
        self._pool_volume_ema = 0.0
        self._recent_bids: Deque[int] = deque(maxlen=400)
        self._recent_wins: Deque[int] = deque(maxlen=400)
        self._last_points = 0.0
        self._last_gain_round = 0

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

        self._ingest_prev_round(prev_auctions, round, agent_id)

        my_state = states[agent_id]
        gold = float(my_state.get("gold", 0) or 0)
        points = float(my_state.get("points", 0) or 0)

        points_array = self._points_snapshot(states.values())
        gold_array = [float(s.get("gold", 0) or 0) for s in states.values()] or [0.0]
        median_points = median(points_array) if points_array else points
        median_gold = median(gold_array)
        leader_points = max(points_array) if points_array else points
        deficit = max(0.0, median_points - points)
        lead = max(0.0, points - median_points)
        table_size = max(len(states), 1)

        total_rounds = round + len(bank_state.get("gold_income_per_round", []))
        progress = 0.0 if total_rounds <= 1 else round / max(total_rounds - 1, 1)

        interest_rate = (bank_state.get("bank_interest_per_round") or [1.0])[0]
        bank_limit = (bank_state.get("bank_limit_per_round") or [0])[0]

        win_rate = self._recent_win_rate()
        stall_rounds = self._update_stall(round, points)
        burst_mode = stall_rounds >= self.BURST_STALL_ROUNDS and win_rate <= self.BURST_WINRATE

        points_floor = self._points_floor(points, median_points, leader_points, progress)

        spend_fraction = (
            0.33
            + 0.44 * progress
            + min(0.22, deficit / 40.0)
            - min(0.14, lead / 80.0)
        )
        if gold < median_gold * 0.85:
            spend_fraction += 0.12
        if burst_mode:
            spend_fraction += 0.18
        spend_fraction += self._income_pressure(bank_state)
        spend_fraction -= min(0.12, max(0.0, interest_rate - 1.0) * 2.2)
        spend_fraction = self._clamp(spend_fraction, 0.2, 0.96)

        reserve = gold * (1 - spend_fraction)
        interest_padding = (
            min(bank_limit, gold) * (0.06 + max(0.0, interest_rate - 1.0) * 1.6)
        )
        reserve = max(reserve, interest_padding)
        if burst_mode:
            reserve *= 0.6
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
            priority_floor = 0.015 + 0.025 * progress
            if burst_mode:
                priority_floor *= 0.5

            for aid, ev, predicted, priority in auction_plan:
                if priority < priority_floor and ev < max_ev * 0.5:
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
                    burst_mode=burst_mode,
                    win_rate=win_rate,
                )
                if bid <= 0:
                    continue

                max_affordable = gold_available - reserve
                if max_affordable <= 0:
                    break
                bid = min(bid, max_affordable)

                if bid < 1:
                    continue

                jitter = 1 + self.random.uniform(-0.02, 0.04)
                bid = int(max(1, bid * jitter))
                if bid == 0:
                    continue

                bids[aid] = bid
                gold_available -= bid

        total_prev_pool = sum(prev_pool_buys.values())
        self._pool_volume_ema = self._blend(
            self._pool_volume_ema or total_prev_pool,
            total_prev_pool,
            0.2,
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
            burst_mode=burst_mode,
            win_rate=win_rate,
        )

        return {"bids": bids, "pool": pool_purchase}

    # ------------------------------------------------------------------
    def _ingest_prev_round(
        self,
        prev_auctions: Dict[str, Dict[str, Any]],
        round_number: int,
        agent_id: str,
    ) -> None:
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
                self._global_ratio = self._blend(self._global_ratio, ratio, 0.18)

            reward = float(info.get("reward", 0) or 0)
            if reward > 0:
                gold_per_point = self._clamp(win_bid / reward, self.MIN_RATIO, self.MAX_RATIO)
                self._gold_per_point_ratio = self._blend(
                    self._gold_per_point_ratio, gold_per_point, 0.1
                )

            bucket = int(ev // self.BUCKET_SIZE) * self.BUCKET_SIZE
            avg, samples = self._bucket_price.get(bucket, (win_bid, 0))
            weight = 0.25 if samples else 1.0
            new_avg = self._blend(avg, win_bid, weight)
            self._bucket_price[bucket] = (new_avg, min(samples + 1, 200))
            self._win_prices.append(win_bid)

            for position, bid in enumerate(bids):
                if bid.get("a_id") == agent_id:
                    self._recent_bids.append(1)
                    self._recent_wins.append(1 if position == 0 else 0)
                    break

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
                blend = min(0.85, samples / (samples + 5))
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
        burst_mode: bool,
        win_rate: float,
    ) -> float:
        if predicted <= 0:
            return 0.0

        aggression = 1.08 + 0.35 * min(1.0, deficit / 35.0) - 0.12 * min(1.0, lead / 90.0)
        aggression += 0.1 * progress
        if table_size <= 4:
            aggression *= 0.9
        elif table_size >= 10:
            aggression *= 1.05
        if burst_mode:
            aggression *= 1.25
        if win_rate < 0.12:
            aggression *= 1.1
        if points < points_floor * 1.1:
            aggression *= 1.1

        bid = predicted * aggression
        floor = ev * max(6.5, 0.6 * self._global_ratio)
        cap = ev * self._global_ratio * (1.65 + 0.4 * progress + (0.2 if burst_mode else 0))
        return max(floor, min(cap, bid))

    def _auction_priority(self, ev: float, predicted: float, max_ev: float, progress: float) -> float:
        if predicted <= 0:
            return 0.0
        density = ev / predicted
        normalized = ev / max(1.0, max_ev)
        priority = 0.58 * density + 0.42 * normalized
        return priority * (1 + 0.28 * progress)

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
        burst_mode: bool,
        win_rate: float,
    ) -> int:
        if pool < self.MIN_POOL or points <= points_floor:
            return 0

        gold_hungry = gold < median_gold * 0.9
        buy_signal = burst_mode or gold_hungry or win_rate < 0.15
        if not buy_signal:
            return 0

        expected_total = max(self._pool_volume_ema, table_size * 2.5)
        value_per_point = pool / max(1.0, expected_total)
        roi_gate = self._gold_per_point_ratio * (1.0 if gold_hungry else 1.15)
        if value_per_point < roi_gate:
            return 0

        surplus_points = max(0.0, points - max(points_floor, lead * 0.4 + 200))
        if surplus_points <= 0:
            return 0

        late_penalty = max(0.0, progress - 0.85)
        base_ratio = 0.15 if burst_mode else 0.1
        if gold_hungry:
            base_ratio += 0.05
        spend_ratio = max(0.05, base_ratio - late_penalty * 0.07)
        if deficit > 0:
            spend_ratio *= 0.6

        spend_cap = min(surplus_points, points * spend_ratio)
        spend = int(spend_cap)
        if spend <= 0:
            return 0

        keep_buffer = max(50, points_floor * 0.06)
        spend = min(spend, int(points - keep_buffer))
        return max(spend, 0)

    # ------------------------------------------------------------------
    def _recent_win_rate(self) -> float:
        if not self._recent_bids:
            return 0.0
        return (sum(self._recent_wins) + 1) / (len(self._recent_bids) + 2)

    def _update_stall(self, round_number: int, points: float) -> int:
        if points > self._last_points:
            self._last_points = points
            self._last_gain_round = round_number
        return round_number - self._last_gain_round

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
        base = max(250.0, 0.2 * leader_points, median_points * (0.4 + 0.25 * progress))
        return min(points, base)


__all__ = ["BraavosPhoenixAgent"]
