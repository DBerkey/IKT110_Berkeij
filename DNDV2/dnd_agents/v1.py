"""Adaptive auction agent implementation for DND auction game."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, cast


PHASE_EARLY = "EARLY"
PHASE_MID = "MID"
PHASE_LATE = "LATE"
PHASE_FINISH = "FINISH"


class CircularBuffer:
	"""Fixed-size buffer that keeps the last *size* values pushed."""

	def __init__(self, size: int) -> None:
		self.size = max(1, size)
		self._values: List[float] = []

	def push(self, value: float) -> None:
		if len(self._values) >= self.size:
			self._values.pop(0)
		self._values.append(value)

	def to_list(self) -> List[float]:
		return list(self._values)

	def __len__(self) -> int:  # pragma: no cover - trivial
		return len(self._values)


def approx_quantiles(buffer: CircularBuffer) -> Tuple[float, float, float]:
	"""Return Q1/Q2/Q3 for the historical bid buffer."""

	data = buffer.to_list()
	if not data:
		return 0.0, 0.0, 0.0

	sorted_vals = sorted(data)

	def percentile(p: float) -> float:
		if len(sorted_vals) == 1:
			return sorted_vals[0]
		idx = p * (len(sorted_vals) - 1)
		lower = math.floor(idx)
		upper = math.ceil(idx)
		if lower == upper:
			return sorted_vals[lower]
		weight = idx - lower
		return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight

	return percentile(0.25), percentile(0.5), percentile(0.75)


@dataclass
class PlayerRecord:
	gold: float = 0.0
	points: float = 0.0
	wins: int = 0
	last_seen_round: int = 0
	last_active_round: int = 0
	activity: float = 0.0

	def register_snapshot(self, round_number: int, gold: float, points: float) -> None:
		self.gold = gold
		self.points = points
		self.last_seen_round = round_number

	def register_win(self, round_number: int, bid_amount: float) -> None:
		self.wins += 1
		self.last_active_round = round_number
		self.activity += bid_amount


def auction_expected_value(auction: Dict[str, int]) -> float:
	die = auction.get("die", 0)
	num = auction.get("num", 0)
	bonus = auction.get("bonus", 0)
	if die <= 0 or num <= 0:
		return float(bonus)
	return ((die + 1) / 2) * num + bonus


class AdaptiveAuctionAgent:
	"""Implements the bidding model described in the prompt."""

	WINDOW_SIZE = 200
	SMALL_EPS = 1e-6
	GOLD_TO_POINT_RATIO = 0.5
	POINT_LEAD_THRESHOLD = 4
	SMALL_TABLE_THRESHOLD = 8
	POOL_PRIORITY_TRIGGER = 0.75
	POOL_ABS_TRIGGER = 900
	ACTIVITY_DECAY = 0.08
	ACTIVITY_NORM = 2000
	WEALTH_NORM = 1500

	pool_weights = {
		PHASE_EARLY: 1.3,
		PHASE_MID: 1.6,
		PHASE_LATE: 1.9,
		PHASE_FINISH: 2.0,
	}

	reserve_fraction = {
		PHASE_EARLY: 0.35,
		PHASE_MID: 0.25,
		PHASE_LATE: 0.05,
		PHASE_FINISH: 0.05,
	}

	max_bid_fraction = {
		PHASE_EARLY: 0.55,
		PHASE_MID: 0.70,
		PHASE_LATE: 0.90,
		PHASE_FINISH: 0.95,
	}

	value_factors = {
		PHASE_EARLY: 0.95,
		PHASE_MID: 1.05,
		PHASE_LATE: 1.20,
		PHASE_FINISH: 1.30,
	}

	ev_percentiles = {
		PHASE_EARLY: 0.40,
		PHASE_MID: 0.55,
		PHASE_LATE: 0.70,
		PHASE_FINISH: 0.85,
	}

	def __init__(self) -> None:
		self.win_hist = CircularBuffer(self.WINDOW_SIZE)
		self.player_stats: Dict[str, PlayerRecord] = {}
		self.last_sync_round = -1
		self.random = random.Random()
		self.round_number = 0
		self.my_gold = 0.0
		self.my_points = 0.0
		self.agent_id = ""

	# ------------------------------------------------------------------
	# Public entry point
	# ------------------------------------------------------------------
	def make_bid(
		self,
		agent_id: str,
		round_number: int,
		states: Dict[str, Dict[str, float]],
		auctions: Dict[str, Dict[str, int]],
		prev_auctions: Dict[str, Dict[str, Any]],
		pool: int,
		prev_pool_buys: Dict[str, int],  # noqa: ARG002 (not used yet)
		bank_state: Dict[str, List[float]],  # noqa: ARG002 (reserved for future heuristics)
	) -> Dict[str, Any]:
		if agent_id not in states:
			return {"bids": {}, "pool": 0}

		self.agent_id = agent_id
		self.round_number = round_number
		self.my_gold = float(states[agent_id].get("gold", 0))
		self.my_points = float(states[agent_id].get("points", 0))

		self._ingest_prev_round(round_number, prev_auctions)
		self._sync_player_snapshots(round_number, states)

		effective_players = max(1.0, self._estimate_effective_players())
		median_points = self._median_points(states.values())
		phase = self._determine_phase(round_number, effective_players, self.my_points, median_points)

		q1, q2, q3 = approx_quantiles(self.win_hist)
		iqr = max(q3 - q1, self.SMALL_EPS)
		small_target = q3 + 0.3 * iqr
		big_target = q3 + 0.9 * iqr
		snipe_target = q3 + 1.6 * iqr

		pool_priority = self._pool_priority(pool, phase, effective_players)

		ev_by_auction = {auction_id: auction_expected_value(cfg) for auction_id, cfg in auctions.items()}
		ev_threshold = self._ev_threshold_for_phase(phase, ev_by_auction.values())

		bids: Dict[str, int] = {}
		for auction_id, auction in auctions.items():
			bid = self._compute_bid_for_auction(
				auction=auction,
				ev=ev_by_auction[auction_id],
				pool=pool,
				phase=phase,
				effective_players=effective_players,
				pool_priority=pool_priority,
				thresholds=(small_target, big_target, snipe_target, iqr),
				ev_threshold=ev_threshold,
			)
			if bid > 0:
				bids[auction_id] = bid

		pool_purchase = self._decide_pool_purchase(pool_priority, phase)
		self._decay_inactive_players(round_number)

		return {"bids": bids, "pool": pool_purchase}

	# ------------------------------------------------------------------
	# Core logic helpers
	# ------------------------------------------------------------------
	def _compute_bid_for_auction(
		self,
		auction: Dict[str, int],
		ev: float,
		pool: int,
		phase: str,
		effective_players: float,
		pool_priority: float,
		thresholds: Tuple[float, float, float, float],
		ev_threshold: float,
	) -> int:
		if self.my_gold <= 0:
			return 0

		small_target, big_target, snipe_target, iqr = thresholds

		pool_focus = pool >= self.POOL_ABS_TRIGGER or pool_priority >= self.POOL_PRIORITY_TRIGGER
		target_bid = 0.0

		if pool_focus:
			min_pool_points_gain = 0.5
			if pool_priority >= min_pool_points_gain:
				target_bid = snipe_target + iqr + pool_priority
			else:
				target_bid = 0.0
		else:
			if ev < ev_threshold:
				return 0
			target_bid = big_target if ev >= ev_threshold * 1.1 else small_target

		value_cap = ev * self.value_factors[phase]
		bid = min(target_bid, value_cap)

		max_affordable = self.my_gold * self.max_bid_fraction[phase]
		bid = min(bid, max_affordable)

		min_reserve = self.my_gold * self.reserve_fraction[phase]
		if self.my_gold - bid < min_reserve:
			bid = max(0.0, self.my_gold - min_reserve)

		noise = self.random.uniform(-0.02, 0.02)
		bid *= 1 + noise

		bid = math.floor(max(0.0, bid))
		return int(bid)

	def _pool_priority(self, pool: int, phase: str, effective_players: float) -> float:
		if pool <= 0:
			return 0.0

		pool_ev_gold_per = pool / max(1.0, effective_players)
		pool_ev_points = pool_ev_gold_per * self.GOLD_TO_POINT_RATIO

		pool_weight = self.pool_weights.get(phase, self.pool_weights[PHASE_LATE])
		pool_priority = pool_ev_points * pool_weight

		current_si = self.my_points + math.log1p(max(0.0, self.my_gold))
		new_si = self.my_points + math.log1p(max(0.0, self.my_gold + pool))
		pool_priority += max(0.0, new_si - current_si)
		return pool_priority

	def _decide_pool_purchase(self, pool_priority: float, phase: str) -> int:
		if self.my_points <= 0 or pool_priority <= 0:
			return 0

		phase_bias = {
			PHASE_EARLY: 0.05,
			PHASE_MID: 0.10,
			PHASE_LATE: 0.20,
			PHASE_FINISH: 0.25,
		}[phase]

		desired_points = pool_priority * phase_bias
		spend_cap = max(0, int(desired_points))
		spend = min(int(self.my_points), spend_cap)
		return spend

	def _ev_threshold_for_phase(self, phase: str, samples: Iterable[float]) -> float:
		sample_list = sorted(samples)
		if not sample_list:
			return 0.0

		percentile = self.ev_percentiles[phase]
		idx = percentile * (len(sample_list) - 1)
		lower = math.floor(idx)
		upper = math.ceil(idx)
		if lower == upper:
			return sample_list[lower]
		weight = idx - lower
		return sample_list[lower] * (1 - weight) + sample_list[upper] * weight

	# ------------------------------------------------------------------
	# State bookkeeping
	# ------------------------------------------------------------------
	def _ingest_prev_round(self, round_number: int, prev_auctions: Dict[str, Dict[str, Any]]) -> None:
		target_round = round_number - 1
		if target_round <= self.last_sync_round:
			return

		for auction in prev_auctions.values():
			bids_raw = auction.get("bids") or []
			bids = cast(List[Dict[str, Any]], bids_raw)
			if not bids:
				continue
			winning_bid = bids[0]
			bid_value = float(winning_bid.get("gold", 0))
			winner_id = str(winning_bid.get("a_id", ""))
			self.win_hist.push(bid_value)
			self._register_win(winner_id, bid_value, target_round)

		self.last_sync_round = target_round

	def _register_win(self, agent_id: str, bid_amount: float, round_number: int) -> None:
		if not agent_id:
			return
		record = self.player_stats.setdefault(agent_id, PlayerRecord())
		record.register_win(round_number, bid_amount)

	def _sync_player_snapshots(self, round_number: int, states: Dict[str, Dict[str, float]]) -> None:
		for pid, state in states.items():
			record = self.player_stats.setdefault(pid, PlayerRecord())
			record.register_snapshot(round_number, float(state.get("gold", 0)), float(state.get("points", 0)))

	def _estimate_effective_players(self) -> float:
		total = 0.0
		for pid, record in self.player_stats.items():
			activity_score = min(1.0, record.activity / self.ACTIVITY_NORM)
			wealth_score = min(1.0, record.gold / self.WEALTH_NORM)
			base = 0.4 if pid == self.agent_id else 0.6
			total += base + 0.3 * activity_score + 0.1 * wealth_score
		return max(total, 1.0)

	def _median_points(self, states: Iterable[Dict[str, float]]) -> float:
		points = [float(state.get("points", 0)) for state in states]
		if not points:
			return 0.0
		return statistics.median(points)

	def _determine_phase(
		self,
		round_number: int,
		effective_players: float,
		my_points: float,
		median_points: float,
	) -> str:
		if round_number < 200:
			phase = PHASE_EARLY
		elif round_number < 400:
			phase = PHASE_MID
		elif round_number < 800:
			phase = PHASE_LATE
		else:
			phase = PHASE_FINISH

		if effective_players <= self.SMALL_TABLE_THRESHOLD:
			phase = PHASE_LATE
		if my_points > median_points + self.POINT_LEAD_THRESHOLD:
			phase = PHASE_LATE

		return phase

	def _decay_inactive_players(self, round_number: int) -> None:
		for pid, record in self.player_stats.items():
			rounds_idle = max(0, round_number - record.last_active_round)
			if rounds_idle > 0:
				decay = math.exp(-self.ACTIVITY_DECAY * rounds_idle)
				record.activity *= decay
			if round_number - record.last_seen_round > 30:
				record.activity *= 0.5


AGENT = AdaptiveAuctionAgent()


def make_bid(
	agent_id: str,
	round: int,  # noqa: A002 - game interface uses this name
	states: Dict[str, Dict[str, float]],
	auctions: Dict[str, Dict[str, int]],
	prev_auctions: Dict[str, Dict[str, object]],
	pool: int,
	prev_pool_buys: Dict[str, int],
	bank_state: Dict[str, List[float]],
) -> Dict[str, Dict[str, int]]:
	"""Adapter function so the game can call into our agent instance."""

	return AGENT.make_bid(
		agent_id=agent_id,
		round_number=round,
		states=states,
		auctions=auctions,
		prev_auctions=prev_auctions,
		pool=pool,
		prev_pool_buys=prev_pool_buys,
		bank_state=bank_state,
	)

