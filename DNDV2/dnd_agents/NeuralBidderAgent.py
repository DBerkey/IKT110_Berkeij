"""NeuralBidderAgent

Uses a tiny feed-forward network trained from historic log files to estimate
competitive bids for every auction. The model is retrained automatically the
first time no cached weights are found, using streaming data from the log
folder to stay memory friendly.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TinyRegressor:
    """Very small MLP with ReLU activations and MSE loss."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Tuple[int, int] = (48, 24),
        lr: float = 1.5e-3,
        seed: int | None = None,
        grad_clip: float = 5.0,
    ) -> None:
        self.lr = lr
        self.grad_clip = grad_clip
        self.rng = np.random.default_rng(seed)
        dims = [input_dim, *hidden_sizes, 1]
        self.layers: List[Dict[str, np.ndarray]] = []
        for idx in range(len(dims) - 1):
            limit = math.sqrt(2.0 / dims[idx])
            weight = self.rng.uniform(-limit, limit, size=(dims[idx], dims[idx + 1])).astype(np.float32)
            bias = np.zeros((1, dims[idx + 1]), dtype=np.float32)
            self.layers.append({"W": weight, "b": bias})

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)

    def _clip(self, array: np.ndarray) -> np.ndarray:
        if self.grad_clip is None or self.grad_clip <= 0:
            return array
        np.clip(array, -self.grad_clip, self.grad_clip, out=array)
        return array

    def predict(self, batch: np.ndarray) -> np.ndarray:
        activations = batch.astype(np.float32)
        for layer in self.layers[:-1]:
            activations = self._relu(activations @ layer["W"] + layer["b"])
        return activations @ self.layers[-1]["W"] + self.layers[-1]["b"]

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 60, batch_size: int = 512) -> None:
        if len(X) == 0:
            return
        for _ in range(epochs):
            indices = self.rng.permutation(len(X))
            X = X[indices]
            y = y[indices]
            for start in range(0, len(X), batch_size):
                end = start + batch_size
                batch_x = X[start:end]
                batch_y = y[start:end]
                if len(batch_x) == 0:
                    continue
                activations = [batch_x]
                pre_acts = []
                a = batch_x
                for layer in self.layers[:-1]:
                    z = a @ layer["W"] + layer["b"]
                    pre_acts.append(z)
                    a = self._relu(z)
                    activations.append(a)
                z = a @ self.layers[-1]["W"] + self.layers[-1]["b"]
                pre_acts.append(z)
                activations.append(z)
                preds = z
                error = preds - batch_y
                grad = (2.0 / len(batch_x)) * error
                self._clip(grad)
                # backprop through output layer
                last_layer = self.layers[-1]
                grad_W = activations[-2].T @ grad
                grad_b = np.sum(grad, axis=0, keepdims=True)
                self._clip(grad_W)
                self._clip(grad_b)
                last_layer["W"] -= self.lr * grad_W
                last_layer["b"] -= self.lr * grad_b
                delta = grad @ last_layer["W"].T
                self._clip(delta)
                # hidden layers
                for layer_idx in range(len(self.layers) - 2, -1, -1):
                    layer = self.layers[layer_idx]
                    activation_input = activations[layer_idx]
                    relu_grad = self._relu_grad(pre_acts[layer_idx])
                    delta = delta * relu_grad
                    self._clip(delta)
                    grad_W = activation_input.T @ delta
                    grad_b = np.sum(delta, axis=0, keepdims=True)
                    self._clip(grad_W)
                    self._clip(grad_b)
                    layer["W"] -= self.lr * grad_W
                    layer["b"] -= self.lr * grad_b
                    if layer_idx > 0:
                        delta = delta @ layer["W"].T
                        self._clip(delta)

    def save(self, path: Path) -> None:
        payload = {f"W{idx}": layer["W"] for idx, layer in enumerate(self.layers)}
        payload.update({f"b{idx}": layer["b"] for idx, layer in enumerate(self.layers)})
        np.savez(path, **payload)

    def load(self, path: Path) -> None:
        data = np.load(path)
        for idx, layer in enumerate(self.layers):
            layer["W"] = data[f"W{idx}"]
            layer["b"] = data[f"b{idx}"]

    def weights_ok(self) -> bool:
        for layer in self.layers:
            if not np.all(np.isfinite(layer["W"])):
                return False
            if not np.all(np.isfinite(layer["b"])):
                return False
        return True


class NeuralBidderAgent:
    VALUE_SCALE = 1000.0
    REWARD_SCALE = 100.0
    POINT_SCALE = 1000.0
    TARGET_SCALE = 100.0
    MAX_SAMPLES = 6000  # fewer samples keep training lightweight
    TRAIN_EPOCHS = 24
    TRAIN_BATCH_SIZE = 256

    def __init__(self, agent_id: str, seed: int | None = None) -> None:
        self.agent_id = agent_id
        self.random = random.Random(seed)
        self._model_dir = Path(__file__).resolve().parent / "model_cache"
        self._model_dir.mkdir(exist_ok=True)
        self._weight_path = self._model_dir / "neural_bidder_weights.npz"
        self._log_dir = Path(__file__).resolve().parents[2] / "dnd_auction_agents" / "logs"
        self._model = TinyRegressor(
            input_dim=12,
            hidden_sizes=(48, 24),
            lr=1.5e-3,
            seed=seed,
            grad_clip=5.0,
        )
        self._ensure_model()

    # ------------------------------------------------------------------
    def make_bid(
        self,
        agent_id: str,
        _round_idx: int,
        states: Dict[str, Dict[str, Any]],
        auctions: Dict[str, Dict[str, Any]],
        _prev_auctions: Dict[str, Dict[str, Any]],
        pool: int,
        _prev_pool_buys: Dict[str, int],
        _bank_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        if agent_id not in states or not auctions:
            return {"bids": {}, "pool": 0}

        my_state = states[agent_id]
        stats = self._state_stats(states)
        spend_fraction = self._spend_fraction(my_state, stats)
        gold = float(my_state.get("gold", 0) or 0)
        reserve = gold * (1 - spend_fraction)
        reserve = max(reserve, gold * 0.2)
        budget = max(0.0, gold - reserve)

        ev_map: List[Tuple[str, float]] = []
        bid_candidates: Dict[str, float] = {}
        for auction_id, auction in auctions.items():
            features = self._encode_features(auction, my_state, stats, pool)
            prediction = float(self._model.predict(features[None, :])[0, 0]) * self.TARGET_SCALE
            prediction = max(1.0, prediction)
            ev = self._expected_value(auction)
            # favor high EV items by slightly scaling the prediction
            priority = prediction * (1.0 + ev / 150.0)
            bid_candidates[auction_id] = prediction
            ev_map.append((auction_id, priority))

        ev_map.sort(key=lambda item: item[1], reverse=True)
        bids: Dict[str, int] = {}
        remaining = budget
        for auction_id, _ in ev_map:
            if remaining <= 1:
                break
            bid_value = min(bid_candidates[auction_id], remaining)
            bid_value *= 1.0 + self.random.uniform(-0.05, 0.05)
            bid_value = max(1.0, bid_value)
            bids[auction_id] = int(bid_value)
            remaining -= bids[auction_id]

        if not bids and gold > 5 and ev_map:
            top_id, _ = ev_map[0]
            fallback_budget = max(5.0, gold * 0.1)
            bids[top_id] = min(int(fallback_budget), int(gold))
            if bids[top_id] <= 0:
                bids[top_id] = 1

        return {"bids": bids, "pool": 0}

    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._weight_path.exists():
            try:
                self._model.load(self._weight_path)
                if self._model.weights_ok():
                    return
                print("[NeuralBidder] Cached weights invalid; retraining.")
            except (OSError, ValueError, KeyError):
                pass
            try:
                self._weight_path.unlink()
            except OSError:
                pass
        dataset = self._build_dataset()
        if dataset is None:
            return
        X, y = dataset
        print(f"[NeuralBidder] Training on {len(X)} samples...")
        try:
            self._model.fit(X, y, epochs=self.TRAIN_EPOCHS, batch_size=self.TRAIN_BATCH_SIZE)
        except KeyboardInterrupt:
            print("[NeuralBidder] Training interrupted; continuing without cached weights.")
            return
        except MemoryError:
            print("[NeuralBidder] Training aborted due to memory pressure; continuing with current weights.")
            return
        if not self._model.weights_ok():
            print("[NeuralBidder] Training failed to converge to finite weights; skipping cache save.")
            return
        try:
            self._model.save(self._weight_path)
        except OSError:
            pass

    def _build_dataset(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self._log_dir.exists():
            return None
        samples: List[np.ndarray] = []
        targets: List[float] = []
        pending: Dict[str, np.ndarray] = {}
        log_files = sorted(self._log_dir.glob("*.jsonl"))
        for path in log_files:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        states = record.get("states") or {}
                        current_agent = record.get("current_agent")
                        my_state = states.get(current_agent) or (states[next(iter(states))] if states else {})
                        pool = record.get("pool", 0) or 0
                        features_for_round = self._feature_map(states, my_state, record.get("auctions") or {}, pool)
                        prev = record.get("prev_auctions") or {}
                        for auction_id, auction_info in prev.items():
                            feature_vec = pending.pop(auction_id, None)
                            if feature_vec is None:
                                continue
                            winning_bid = self._winning_bid(auction_info)
                            if winning_bid is None:
                                continue
                            samples.append(feature_vec)
                            targets.append(winning_bid / self.TARGET_SCALE)
                            if len(samples) >= self.MAX_SAMPLES:
                                return self._finalize_dataset(samples, targets)
                        pending = features_for_round
            except OSError:
                continue
        if not samples:
            return None
        return self._finalize_dataset(samples, targets)

    @staticmethod
    def _finalize_dataset(samples: List[np.ndarray], targets: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.vstack(samples).astype(np.float32)
        y = np.array(targets, dtype=np.float32).reshape(-1, 1)
        return X, y

    def _feature_map(
        self,
        states: Dict[str, Dict[str, Any]],
        my_state: Dict[str, Any],
        auctions: Dict[str, Dict[str, Any]],
        pool: int,
    ) -> Dict[str, np.ndarray]:
        stats = self._state_stats(states)
        return {
            auction_id: self._encode_features(auction, my_state, stats, pool)
            for auction_id, auction in auctions.items()
        }

    def _encode_features(
        self,
        auction: Dict[str, Any],
        agent_state: Dict[str, Any],
        stats: Dict[str, float],
        pool: int,
    ) -> np.ndarray:
        gold = float(agent_state.get("gold", 0) or 0)
        points = float(agent_state.get("points", 0) or 0)
        die = float(auction.get("die", 0) or 0)
        num = float(auction.get("num", 0) or 0)
        bonus = float(auction.get("bonus", 0) or 0)
        reward = float(auction.get("reward", 0) or 0)
        ev = self._expected_value(auction)
        feature = np.array(
            [
                gold / self.VALUE_SCALE,
                stats["mean_gold"] / self.VALUE_SCALE,
                stats["median_gold"] / self.VALUE_SCALE,
                stats["max_gold"] / self.VALUE_SCALE,
                stats["std_gold"] / self.VALUE_SCALE,
                pool / self.VALUE_SCALE,
                reward / self.REWARD_SCALE,
                ev / self.REWARD_SCALE,
                bonus / 50.0,
                die / 20.0,
                num / 10.0,
                points / self.POINT_SCALE,
            ],
            dtype=np.float32,
        )
        return feature

    def _state_stats(self, states: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        if not states:
            return {
                "mean_gold": 0.0,
                "median_gold": 0.0,
                "max_gold": 0.0,
                "std_gold": 0.0,
                "median_points": 0.0,
            }
        gold_values = np.array([float(s.get("gold", 0) or 0) for s in states.values()], dtype=np.float32)
        points_values = np.array([float(s.get("points", 0) or 0) for s in states.values()], dtype=np.float32)
        if len(gold_values) == 0:
            gold_values = np.array([0.0], dtype=np.float32)
        if len(points_values) == 0:
            points_values = np.array([0.0], dtype=np.float32)
        return {
            "mean_gold": float(np.mean(gold_values)),
            "median_gold": float(np.median(gold_values)),
            "max_gold": float(np.max(gold_values)),
            "std_gold": float(np.std(gold_values) + 1e-3),
            "median_points": float(np.median(points_values)),
        }

    def _spend_fraction(self, agent_state: Dict[str, Any], stats: Dict[str, float]) -> float:
        my_points = float(agent_state.get("points", 0) or 0)
        deficit = stats["median_points"] - my_points
        base = 0.36
        if deficit > 0:
            base += min(0.2, deficit / 2500.0)
        else:
            base -= min(0.15, abs(deficit) / 4000.0)
        return max(0.2, min(0.7, base))

    @staticmethod
    def _winning_bid(auction_info: Dict[str, Any]) -> Optional[float]:
        bids = auction_info.get("bids") or []
        if not bids:
            return None
        top_bid = bids[0]
        if isinstance(top_bid, dict):
            gold = top_bid.get("gold")
            if gold is None:
                return None
            return float(gold)
        if isinstance(top_bid, (int, float)):
            return float(top_bid)
        return None

    @staticmethod
    def _expected_value(auction: Dict[str, Any]) -> float:
        die = float(auction.get("die", 0) or 0)
        num = float(auction.get("num", 0) or 0)
        bonus = float(auction.get("bonus", 0) or 0)
        if die <= 0 or num <= 0:
            return bonus
        return ((die + 1.0) / 2.0) * num + bonus


__all__ = ["NeuralBidderAgent"]
