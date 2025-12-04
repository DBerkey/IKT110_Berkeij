"""Braavos Inflation Agent: resilient, price-tracking bidder that never stalls."""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

try:  # pragma: no cover - fallback for type-checkers
    from dnd_auction_game import AuctionGameClient  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    AuctionGameClient = None  # type: ignore[assignment]

AVG_ROLL = {
    1: 1.0,
    2: 1.5,
    3: 2.0,
    4: 2.5,
    6: 3.5,
    8: 4.5,
    10: 5.5,
    12: 6.5,
    20: 10.5,
}

EMA_ALPHA = 0.2


@dataclass
class BraavosInflationState:
    market_cp: float = 10.0
    prev_points: float = 0.0
    prev_gold: float = 1000.0
    stagnation_rounds: int = 0


class BraavosInflationAgent:
    def __init__(self, *, seed: int | None = None, total_rounds: int | None = None) -> None:
        self.random = random.Random(seed)
        self.total_rounds = total_rounds or 1000
        self.state = BraavosInflationState()

    def make_bid(
        self,
        agent_id: str,
        round_number: int,
        states: Dict[str, Dict[str, Any]],
        auctions: Dict[str, Dict[str, Any]],
        prev_auctions: Dict[str, Dict[str, Any]],
        pool_gold: int,
        prev_pool_buys: Dict[str, int],
        bank_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        agent_state = states.get(agent_id)
        if not agent_state:
            return {"bids": {}, "pool": 0}

        self._update_price_ratio(prev_auctions, agent_id)
        my_gold = int(agent_state.get("gold", 0))
        my_points = int(agent_state.get("points", 0))
        self._update_stagnation(my_points)

        if my_gold <= 0 and pool_gold <= 0:
            return {"bids": {}, "pool": 0}

        rounds_left = self._rounds_left(round_number, bank_state)
        progress = self._progress(round_number, rounds_left)
        bank_limit = self._current_bank_limit(bank_state)

        points_snapshot = self._points_snapshot(states.values())
        median_points = statistics.median(points_snapshot) if points_snapshot else float(my_points)
        leader_points = max(points_snapshot) if points_snapshot else float(my_points)
        deficit = max(0.0, median_points - my_points)
        lead = max(0.0, my_points - median_points)

        reserve = self._reserve_target(my_gold, bank_limit, progress, deficit, lead)
        usable_gold = max(0, my_gold - reserve)

        target_burn = max(1.0, (usable_gold + reserve) / max(1, rounds_left))
        aggression = self._aggression_factor(deficit, lead, usable_gold, target_burn, progress)

        bids = self._allocate_bids(
            auctions=auctions,
            usable_gold=usable_gold,
            aggression=aggression,
            bank_limit=bank_limit,
            progress=progress,
        )

        sell_points = self._decide_pool_sale(
            pool_gold=pool_gold,
            my_points=my_points,
            median_points=median_points,
            leader_points=leader_points,
            bank_limit=bank_limit,
            progress=progress,
            prev_pool_buys=prev_pool_buys,
        )

        spent_gold = sum(bids.values())
        remaining_gold = max(0, my_gold - spent_gold)
        self._snapshot(remaining_gold, my_points)

        return {"bids": bids, "pool": sell_points}

    # ------------------------------------------------------------------
    # Mechanics
    # ------------------------------------------------------------------
    def _allocate_bids(
        self,
        *,
        auctions: Dict[str, Dict[str, Any]],
        usable_gold: int,
        aggression: float,
        bank_limit: int,
        progress: float,
    ) -> Dict[str, int]:
        if usable_gold <= 0 or not auctions:
            return {}

        stagnation_bonus = min(0.25, self.state.stagnation_rounds / 12.0)
        inflation_bonus = min(0.2, max(0.0, self.state.market_cp - 25.0) / 80.0)
        burn_ratio = 0.55 + 0.35 * progress + stagnation_bonus + inflation_bonus
        burn_ratio = min(0.95, burn_ratio)
        target_budget = int(min(usable_gold, max(int(usable_gold * 0.45), int(usable_gold * burn_ratio))))

        single_ratio = 0.35 + 0.25 * progress + min(0.2, self.state.stagnation_rounds / 15.0)
        if progress > 0.85:
            single_ratio += 0.25
        max_single = int(max(300, (usable_gold + bank_limit) * min(0.95, single_ratio)))

        scored = []
        for a_id, auction in auctions.items():
            ev = self._expected_value(auction)
            if ev <= 0:
                continue
            scored.append((ev, a_id, auction))
        if not scored:
            return {}

        scored.sort(reverse=True)
        bids: Dict[str, int] = {}
        remaining = target_budget
        cp = max(1.0, self.state.market_cp)

        top_count = max(1, len(scored) // 3)
        top_segment = scored[:top_count]
        rest_segment = scored[top_count:]

        remaining = self._bid_into_segment(top_segment, remaining, bids, cp, aggression * 1.1, max_single)
        self._bid_into_segment(rest_segment, remaining, bids, cp, aggression, max_single)

        return bids

    def _bid_into_segment(
        self,
        segment: List[Tuple[float, str, Dict[str, Any]]],
        remaining: int,
        bids: Dict[str, int],
        cp: float,
        aggression: float,
        max_single: int,
    ) -> int:
        for ev, a_id, _auction in segment:
            if remaining <= 0:
                break
            base = ev * cp * aggression
            jitter = self.random.uniform(0.95, 1.08)
            bid = int(min(max_single, remaining, base * jitter))
            if bid <= 0:
                continue
            bids[a_id] = bid
            remaining -= bid
        return remaining

    def _reserve_target(
        self,
        my_gold: int,
        bank_limit: int,
        progress: float,
        deficit: float,
        lead: float,
    ) -> int:
        # inflation pressure collapses reserves when market CP explodes
        inflation_pressure = max(1.0, min(4.0, self.state.market_cp / 20.0))
        bank_component = bank_limit * (0.28 if progress < 0.5 else 0.16)
        gold_component = my_gold * (0.06 if progress < 0.75 else 0.035)
        reserve = int((bank_component + gold_component) / inflation_pressure)

        if deficit > 80:
            reserve = int(reserve * 0.7)
        if lead > 250 and progress > 0.5:
            reserve = int(reserve * 0.6)
        if self.state.stagnation_rounds > 5:
            reserve = int(reserve * 0.65)
        if progress > 0.9:
            reserve = int(reserve * 0.35)

        hard_cap = max(0, my_gold - 150)
        return max(0, min(reserve, hard_cap))

    def _aggression_factor(
        self,
        deficit: float,
        lead: float,
        usable_gold: int,
        target_burn: float,
        progress: float,
    ) -> float:
        factor = 1.0
        factor += min(0.8, deficit / 320.0)
        factor += min(0.6, self.state.stagnation_rounds / 10.0)
        factor += min(0.4, max(0.0, usable_gold - target_burn) / max(1.0, target_burn * 2.0))
        factor += 0.2 * progress
        factor -= min(0.4, lead / 600.0)
        return max(0.8, min(3.5, factor))

    def _decide_pool_sale(
        self,
        *,
        pool_gold: int,
        my_points: int,
        median_points: float,
        leader_points: float,
        bank_limit: int,
        progress: float,
        prev_pool_buys: Dict[str, int],
    ) -> int:
        if pool_gold <= 0 or my_points <= 0:
            return 0

        lagging = max(0.0, median_points - my_points)
        points_floor = max(220.0, median_points * (0.6 + 0.25 * progress))
        lead_floor = leader_points * (0.4 + 0.25 * progress)
        keep_points = int(max(points_floor, lead_floor))
        surplus_points = max(0, my_points - keep_points)
        if surplus_points <= 0:
            return 0

        early_floor = 0.3 if progress < 0.8 else 0.2
        gold_floor = int(bank_limit * early_floor)
        gold_pressure = max(0, gold_floor - self.state.prev_gold)
        gold_emergency = gold_pressure > 0 and self.state.prev_gold < bank_limit * 0.35
        if not gold_emergency and progress < 0.95:
            return 0
        if self.state.stagnation_rounds < 4 and lagging < 120 and progress < 0.95:
            return 0

        desired_gold = int(gold_pressure * (1.0 + progress)) + int(self.state.stagnation_rounds * 20)
        desired_gold = max(desired_gold, int(bank_limit * 0.15))
        desired_points = min(surplus_points, desired_gold)

        recent = int(prev_pool_buys.get("braavos_inflation", 0) or 0)
        desired_points = max(0, desired_points - recent // 2)
        desired_points = min(desired_points, pool_gold)
        return desired_points

    def _update_price_ratio(self, prev_auctions: Dict[str, Dict[str, Any]], my_id: str) -> None:
        ratios: List[float] = []
        for info in prev_auctions.values():
            reward = float(info.get("reward", 0))
            bids = info.get("bids") or []
            if reward <= 0 or not bids:
                continue
            winner = bids[0]
            if winner.get("gold", 0) <= 0:
                continue
            ratios.append(float(winner["gold"]) / reward)
        if not ratios:
            return
        median_cp = statistics.median(ratios)
        self.state.market_cp = (1 - EMA_ALPHA) * self.state.market_cp + EMA_ALPHA * median_cp
        self.state.market_cp = max(1.0, self.state.market_cp)

    def _rounds_left(self, round_number: int, bank_state: Dict[str, Any]) -> int:
        if not isinstance(round_number, int):
            round_number = 0
        income = bank_state.get("gold_income_per_round") if isinstance(bank_state, dict) else None
        if isinstance(income, list) and income:
            return max(1, len(income))
        return max(1, self.total_rounds - round_number)

    def _progress(self, round_number: int, rounds_left: int) -> float:
        denom = max(1, round_number + rounds_left)
        return max(0.0, min(1.0, round_number / denom))

    def _current_bank_limit(self, bank_state: Dict[str, Any]) -> int:
        limits = bank_state.get("bank_limit_per_round") if isinstance(bank_state, dict) else None
        if isinstance(limits, list) and limits:
            return int(limits[0])
        return 5000

    def _expected_value(self, auction: Dict[str, Any]) -> float:
        die = int(auction.get("die", 0) or 0)
        num = int(auction.get("num", 0) or 0)
        bonus = float(auction.get("bonus", 0) or 0)
        return max(0.0, num * AVG_ROLL.get(die, 0.0) + bonus)

    def _points_snapshot(self, states: Iterable[Dict[str, Any]]) -> List[float]:
        return [float(state.get("points", 0) or 0) for state in states]

    def _update_stagnation(self, current_points: int) -> None:
        if current_points > self.state.prev_points:
            self.state.stagnation_rounds = 0
        else:
            self.state.stagnation_rounds += 1

    def _snapshot(self, gold: int, points: int) -> None:
        self.state.prev_gold = gold
        self.state.prev_points = points


def make_bid(
    agent_id: str,
    round_number: int,
    states: Dict[str, Dict[str, Any]],
    auctions: Dict[str, Dict[str, Any]],
    prev_auctions: Dict[str, Dict[str, Any]],
    pool_gold: int,
    prev_pool_buys: Dict[str, int],
    bank_state: Dict[str, Any],
) -> Dict[str, Any]:
    if not hasattr(make_bid, "_agent"):
        make_bid._agent = BraavosInflationAgent()  # type: ignore[attr-defined]

    return make_bid._agent.make_bid(  # type: ignore[attr-defined]
        agent_id,
        round_number,
        states,
        auctions,
        prev_auctions,
        pool_gold,
        prev_pool_buys,
        bank_state,
    )


def main() -> None:
    if AuctionGameClient is None:  # pragma: no cover
        raise RuntimeError("AuctionGameClient import missing")

    client = AuctionGameClient(agent_name="braavos_inflation")
    client.run(make_bid)


if __name__ == "__main__":  # pragma: no cover
    main()
