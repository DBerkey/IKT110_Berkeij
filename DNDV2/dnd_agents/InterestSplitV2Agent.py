"""InterestSplitV2Agent: interest-split bidding with adaptive pool buying.

This version was tuned after reviewing recent auction logs where the previous
agent routinely sat on points leads without converting them into gold. The
logic now tracks clearing-price ratios, dynamically balances conservative vs
aggressive bids, and spends points on the gold pool whenever we have surplus
points or prolonged stagnation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class InterestSplitV2State:
    prev_gold: int = 0
    prev_points: int = 0
    stagnation_rounds: int = 0
    price_ratio: float = 24.0


class InterestSplitV2Agent:
    def __init__(self, agent_id: str, seed: int | None = None) -> None:
        self.agent_id = agent_id
        self.random = random.Random(seed)
        self.state = InterestSplitV2State()
        self.wait_rounds = 4
        self.minimum_reserve = 4_500
        self.bottom_fraction = 0.65
        self.stagnation_trigger = 8
        self.max_single_bid_ratio = 0.38
        self.pool_min_gold = 400
        self.pool_pressure_divisor = 12_000
        self.late_progress_threshold = 0.75
        self.endgame_progress_threshold = 0.9

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
        if agent_id not in states:
            return {"bids": {}, "pool": 0}

        self._update_price_ratio(prev_auctions)

        agent_state = states[agent_id]
        current_gold = int(agent_state.get("gold", 0))
        current_points = int(agent_state.get("points", 0))

        points_snapshot = self._points_snapshot(states.values())
        median_points = median(points_snapshot) if points_snapshot else float(current_points)
        leader_points = max(points_snapshot) if points_snapshot else float(current_points)
        deficit = max(0.0, median_points - current_points)
        lead = max(0.0, current_points - median_points)

        progress = self._progress(round, bank_state)
        points_floor = self._points_floor(median_points, leader_points, progress)

        pool_purchase = self._decide_pool_purchase(
            pool=pool,
            points=current_points,
            points_floor=points_floor,
            median_points=median_points,
            leader_points=leader_points,
            progress=progress,
            prev_pool_buys=prev_pool_buys,
            deficit=deficit,
            lead=lead,
        )

        self._update_stagnation(current_points)

        if round < self.wait_rounds or not auctions:
            self._snapshot(current_gold, current_points)
            return {"bids": {}, "pool": pool_purchase}

        bank_limit = self._current_bank_limit(bank_state)
        interest_rate = float((bank_state.get("bank_interest_per_round") or [1.0])[0] or 1.0)
        gold_above_limit = max(0, current_gold - bank_limit)

        reserve = self.minimum_reserve + int(bank_limit * (0.35 if progress < 0.5 else 0.2))
        reserve += int(current_gold * max(0.04, min(0.12, interest_rate - 1.0)))
        if deficit > 40:
            reserve = int(reserve * 0.85)
        if lead > 200:
            reserve = int(reserve * 0.65)
        if self.state.stagnation_rounds >= self.stagnation_trigger:
            reserve = int(reserve * 0.9)
        if current_gold > bank_limit * 1.6:
            reserve = int(reserve * 0.85)
        if current_gold > bank_limit * 2.2:
            reserve = int(reserve * 0.7)
        if self.state.price_ratio < 18:
            reserve = int(reserve * 0.85)
        if self.state.price_ratio < 14:
            reserve = int(reserve * 0.72)
        endgame_pressure = max(0.0, progress - self.late_progress_threshold)
        if endgame_pressure > 0:
            reserve = int(reserve * (1.0 - 0.3 * endgame_pressure))
        if progress > self.endgame_progress_threshold:
            reserve = int(reserve * 0.55)
        reserve = max(0, min(reserve, max(0, current_gold - 250)))

        usable_gold = max(0, current_gold - reserve)
        if usable_gold <= 0:
            self._snapshot(current_gold, current_points)
            return {"bids": {}, "pool": pool_purchase}

        gold_pressure = max(0.0, current_gold - reserve - bank_limit)
        agg_fraction = 0.35
        agg_fraction += min(0.22, deficit / 220.0)
        agg_fraction += min(0.18, self.state.stagnation_rounds / 15.0)
        agg_fraction -= min(0.18, lead / 420.0)
        agg_fraction += min(0.25, gold_pressure / max(1.0, bank_limit * 3.0))
        if progress > self.late_progress_threshold:
            agg_fraction += 0.12
        agg_fraction = max(0.2, min(0.8, agg_fraction))

        base_aggressive = int(usable_gold * agg_fraction)
        aggressive_budget = min(usable_gold, max(base_aggressive, min(gold_above_limit, usable_gold)))
        conservative_budget = max(0, usable_gold - aggressive_budget)

        auction_values: List[Tuple[str, Dict[str, Any], float]] = [
            (auction_id, auction, self._expected_value(auction))
            for auction_id, auction in auctions.items()
        ]
        auction_values = [entry for entry in auction_values if entry[2] > 0]
        if not auction_values:
            self._snapshot(current_gold, current_points)
            return {"bids": {}, "pool": pool_purchase}

        auction_values.sort(key=lambda entry: entry[2])
        split_index = max(1, int(len(auction_values) * self.bottom_fraction))
        bottom_segment = auction_values[:split_index]
        top_segment = list(reversed(auction_values[split_index:]))

        aggressive_multiplier = 1.05 + min(0.35, deficit / 240.0)
        aggressive_multiplier += min(0.25, self.state.stagnation_rounds / 20.0)
        aggressive_multiplier -= min(0.2, lead / 480.0)
        aggressive_multiplier += min(0.3, gold_pressure / max(1.0, bank_limit * 4.0))
        if progress > self.late_progress_threshold:
            aggressive_multiplier += 0.15
        aggressive_multiplier = max(0.9, aggressive_multiplier)

        conservative_multiplier = 0.75 + min(0.18, deficit / 260.0)
        conservative_multiplier = max(0.55, min(1.05, conservative_multiplier))

        max_single_bid = max(150, int(current_gold * self.max_single_bid_ratio))

        bids: Dict[str, int] = {}
        bids.update(
            self._allocate_segment(
                segment=top_segment,
                budget=aggressive_budget,
                max_single_bid=max_single_bid,
                multiplier=aggressive_multiplier,
            )
        )
        bids.update(
        spent_gold = sum(bids.values())
        remaining = max(0, usable_gold - spent_gold)
        if remaining > 0 and progress > self.late_progress_threshold and top_segment:
            top_up_segment = top_segment[: max(1, len(top_segment) // 2)]
            top_up = self._allocate_segment(
                segment=top_up_segment,
                budget=remaining,
                max_single_bid=max_single_bid,
                multiplier=aggressive_multiplier * 1.1,
            )
            for auction_id, extra in top_up.items():
                bids[auction_id] = bids.get(auction_id, 0) + extra
            spent_gold = sum(bids.values())
                segment=bottom_segment,
                budget=conservative_budget,
                max_single_bid=max_single_bid,
                multiplier=conservative_multiplier,
            )
        )

        spent_gold = sum(bids.values())
        self._snapshot(max(0, current_gold - spent_gold), current_points)

        return {"bids": bids, "pool": pool_purchase}

    def _allocate_segment(
        self,
        *,
        segment: List[Tuple[str, Dict[str, Any], float]],
        budget: int,
        max_single_bid: int,
        multiplier: float,
    ) -> Dict[str, int]:
        if not segment or budget <= 0:
            return {}

        remaining = budget
        max_ev = max((value for _, _, value in segment), default=1.0)
        bids: Dict[str, int] = {}

        for index, (auction_id, _auction, ev) in enumerate(segment):
            if remaining <= 0:
                break

            weight = ev / max_ev if max_ev > 0 else 1.0
            share = max(1.0, remaining / max(1, len(segment) - index))
            target = ev * self.state.price_ratio * multiplier
            scaled = min(target, share * (0.55 + weight))
            jitter = 1.0 + self.random.uniform(-0.05, 0.08)
            bid = int(max(1, min(remaining, max_single_bid, scaled * jitter)))
            if bid <= 0:
                continue

            bids[auction_id] = bid
            remaining -= bid

        return bids

    def _decide_pool_purchase(
        self,
        *,
        pool: int,
        points: int,
        points_floor: float,
        median_points: float,
        leader_points: float,
        progress: float,
        prev_pool_buys: Dict[str, int],
        deficit: float,
        lead: float,
    ) -> int:
        if pool <= self.pool_min_gold or points <= 0:
            return 0

        surplus_points = max(0.0, points - max(points_floor, 0.0))
        if surplus_points <= 10:
            return 0
        if deficit > 80 and lead <= 0:
            return 0

        pressure = min(1.0, pool / float(self.pool_pressure_divisor))
        stagnation_bonus = min(0.3, self.state.stagnation_rounds / 12.0)
        lead_bonus = min(0.2, lead / max(1.0, leader_points))
        target_ratio = 0.07 + 0.3 * pressure + stagnation_bonus + lead_bonus

        if progress > 0.9:
            target_ratio *= 0.4
        if deficit > 0:
            target_ratio *= 0.55

        desired_spend = int(surplus_points * target_ratio)
        recent = int(prev_pool_buys.get(self.agent_id, 0) or 0)
        desired_spend = max(0, desired_spend - recent // 2)

        keep_buffer = max(30, int(points_floor * 0.1))
        spend_cap = max(0, points - keep_buffer)
        desired_spend = min(desired_spend, spend_cap)

        return max(0, desired_spend)

    def _update_price_ratio(self, prev_auctions: Dict[str, Dict[str, Any]]) -> None:
        ratios: List[float] = []
        for info in prev_auctions.values():
            bids = info.get("bids") or []
            if not bids:
                continue
            win_gold = float(bids[0].get("gold", 0) or 0)
            if win_gold <= 0:
                continue
            ev = self._expected_value(info)
            if ev <= 0:
                continue
            ratios.append(win_gold / max(1.0, ev))

        if not ratios:
            return

        avg_ratio = sum(ratios) / len(ratios)
        self.state.price_ratio = self.state.price_ratio * 0.75 + avg_ratio * 0.25

    def _update_stagnation(self, current_points: int) -> None:
        if current_points > self.state.prev_points:
            self.state.stagnation_rounds = 0
        else:
            self.state.stagnation_rounds += 1

    def _snapshot(self, gold: int, points: int) -> None:
        self.state.prev_gold = gold
        self.state.prev_points = points

    @staticmethod
    def _expected_value(auction: Dict[str, Any]) -> float:
        die = int(auction.get("die", 0) or 0)
        num = int(auction.get("num", 0) or 0)
        bonus = float(auction.get("bonus", 0) or 0)
        if die <= 0 or num <= 0:
            return bonus
        return ((die + 1) / 2) * num + bonus

    @staticmethod
    def _current_bank_limit(bank_state: Dict[str, Any]) -> int:
        limits = bank_state.get("bank_limit_per_round")
        if isinstance(limits, list) and limits:
            return int(limits[0])
        return 2_000

    @staticmethod
    def _points_snapshot(states: Iterable[Dict[str, Any]]) -> List[float]:
        snapshot: List[float] = []
        for state in states:
            snapshot.append(float(state.get("points", 0) or 0))
        return snapshot

    @staticmethod
    def _points_floor(median_points: float, leader_points: float, progress: float) -> float:
        base = max(160.0, median_points * (0.55 + 0.18 * progress))
        lead_floor = leader_points * (0.25 + 0.15 * progress)
        return max(base, lead_floor)

    @staticmethod
    def _progress(round_number: int, bank_state: Dict[str, Any]) -> float:
        incomes = bank_state.get("gold_income_per_round") or []
        remaining = len(incomes)
        total_rounds = round_number + max(remaining, 1)
        max_round = max(total_rounds - 1, 1)
        return max(0.0, min(1.0, round_number / max_round))


__all__ = ["InterestSplitV2Agent"]
