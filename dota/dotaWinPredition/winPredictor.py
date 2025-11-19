"""
Author: Douwe Berkeij
Date: 18-11-2025
Description: This module contains functions to predict the outcome of a Dota 2 match based on hero selections.

to run use & C:/Users/berke/anaconda3/envs/IKT110/python.exe -m dota.dotaWinPredition.winPredictor
"""

from typing import List, Tuple
import csv
from itertools import combinations
import numpy as np
import random
from .dataLoader import load_data
from .logistic_model import LogisticRegressionSGD


def load_lookup_matrix(filepath: str, n_heroes: int) -> np.ndarray:
    """Loads a hero lookup CSV into a dense matrix clipped to the hero pool size."""
    matrix = np.zeros((n_heroes, n_heroes), dtype=float)
    try:
        with open(filepath, newline='', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, None)
            if not header:
                return matrix

            column_ids: List[int | None] = []
            for value in header[1:]:
                try:
                    column_ids.append(int(value))
                except (TypeError, ValueError):
                    column_ids.append(None)

            for row in reader:
                if not row:
                    continue
                try:
                    hero_id = int(row[0])
                except ValueError:
                    continue
                if hero_id >= n_heroes:
                    continue

                for opponent_id, score in zip(column_ids, row[1:]):
                    if opponent_id is None or opponent_id >= n_heroes or score == '':
                        continue
                    try:
                        matrix[hero_id, opponent_id] = float(score)
                    except ValueError:
                        continue
    except FileNotFoundError:
        # Default to all zeros if the lookup does not exist.
        return matrix

    return matrix

class DotaWinPredictor:
    """Predicts the outcome of a Dota 2 match based on hero selections."""

    def __init__(
        self,
        n_heroes: int,
        counter_lookup: np.ndarray | None = None,
        synergy_lookup: np.ndarray | None = None,
    ):
        self.n_heroes = n_heroes
        self.counter_lookup = counter_lookup
        self.synergy_lookup = synergy_lookup
        self.extra_features = 0
        if self.synergy_lookup is not None:
            self.extra_features += 2  # radiant and dire synergy
        if self.counter_lookup is not None:
            self.extra_features += 2  # radiant over dire and vice versa counter pressure
        self.n_features = 2 * self.n_heroes + self.extra_features
        self.model = LogisticRegressionSGD(n_features=self.n_features, learning_rate=1e-3, l2=1e-4)

    def _team_synergy_score(self, heroes: List[int]) -> float:
        if self.synergy_lookup is None or len(heroes) < 2:
            return 0.0
        total = 0.0
        count = 0
        for hero_a, hero_b in combinations(heroes, 2):
            if hero_a >= self.synergy_lookup.shape[0] or hero_b >= self.synergy_lookup.shape[1]:
                continue
            total += 0.5 * (self.synergy_lookup[hero_a, hero_b] + self.synergy_lookup[hero_b, hero_a])
            count += 1
        return total / count if count else 0.0

    def _team_counter_score(self, team: List[int], opponents: List[int]) -> float:
        if self.counter_lookup is None or not team or not opponents:
            return 0.0
        total = 0.0
        count = 0
        for hero_a in team:
            if hero_a >= self.counter_lookup.shape[0]:
                continue
            for hero_b in opponents:
                if hero_b >= self.counter_lookup.shape[1]:
                    continue
                total += self.counter_lookup[hero_a, hero_b]
                count += 1
        return total / count if count else 0.0

    def _encode_match(self, radiant_heroes: List[int], dire_heroes: List[int]) -> np.ndarray:
        """Encodes the hero selections into a feature vector with optional synergy/counter stats."""
        x = np.zeros(self.n_features)
        for hero_id in radiant_heroes:
            x[hero_id] = 1
        for hero_id in dire_heroes:
            x[self.n_heroes + hero_id] = 1

        extra_idx = 2 * self.n_heroes
        if self.synergy_lookup is not None:
            x[extra_idx] = self._team_synergy_score(radiant_heroes)
            x[extra_idx + 1] = self._team_synergy_score(dire_heroes)
            extra_idx += 2
        if self.counter_lookup is not None:
            x[extra_idx] = self._team_counter_score(radiant_heroes, dire_heroes)
            x[extra_idx + 1] = self._team_counter_score(dire_heroes, radiant_heroes)
        return x

    def prepare_dataset(self, matches: List[Tuple[List[int], List[int], str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares the dataset from match data.

        Each match is a tuple of (radiant_heroes, dire_heroes, winner),
        where winner is 1 if Radiant won, 0 if Dire won.
        """
        X = []
        y = []
        for radiant_heroes, dire_heroes, winner in matches:
            if winner == "True":
                winner = 1
            else:
                winner = 0
            x = self._encode_match(radiant_heroes, dire_heroes)
            X.append(x)
            y.append(winner)
        return np.array(X), np.array(y)
    
    def split_dataset(self, matches: List[Tuple[List[int], List[int], str]], train_ratio: float = 0.8, random_state: int | None = 0) -> Tuple[List[Tuple[List[int], List[int], str]], List[Tuple[List[int], List[int], str]]]:
        """Splits the dataset into training and testing sets with shuffling."""
        rng = random.Random(random_state)
        matches_shuffled = matches.copy()
        rng.shuffle(matches_shuffled)

        n_train = int(len(matches_shuffled) * train_ratio)
        train_matches = matches_shuffled[:n_train]
        test_matches = matches_shuffled[n_train:]
        return train_matches, test_matches

    def train(self, matches: List[Tuple[List[int], List[int], int]], n_epochs: int = 100) -> None:
        """Trains the win predictor model."""
        X, y = self.prepare_dataset(matches)
        self.model.fit(X, y, n_epochs=n_epochs)

    def test(self, matches: List[Tuple[List[int], List[int], int]]) -> float:
        """Tests the win predictor model and returns accuracy."""
        X, y = self.prepare_dataset(matches)
        probs = self.model.predict_proba(X).reshape(-1)
        preds = (probs >= 0.5).astype(int).flatten()
        accuracy = float(np.mean(preds == y))
        # confusion matrix
        tp = np.sum((preds == 1) & (y == 1))
        tn = np.sum((preds == 0) & (y == 0))
        fp = np.sum((preds == 1) & (y == 0))
        fn = np.sum((preds == 0) & (y == 1))
        confusion_matrix = np.array([[tp, fp], [fn, tn]])
        # F1 score and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return accuracy, confusion_matrix, f1, recall

    def save_model(self, filepath: str) -> None:
        """Saves the model parameters to a file."""
        np.savez(filepath, w=self.model.w, b=self.model.b)

    def load_model(self, filepath: str) -> None:
        """Loads the model parameters from a file."""
        data = np.load(filepath)
        self.model.w = data['w']
        self.model.b = data['b']

    def predict(self, radiant_heroes: List[int], dire_heroes: List[int]) -> float:
        """Predicts the probability of Radiant winning given hero selections."""
        x = self._encode_match(radiant_heroes, dire_heroes).reshape(1, -1)
        prob = self.model.predict_proba(x)
        return float(prob.item())
    
def main() -> None:
    matchCSV = "C:\\Users\\berke\\Documents\\UiA\\IKT110_Berkeij\\dota\\new_ranked_matches.csv"
    playerCSV = "C:\\Users\\berke\\Documents\\UiA\\IKT110_Berkeij\\dota\\new_ranked_players.csv"
    heroCounters = "C:\\Users\\berke\\Documents\\UiA\\IKT110_Berkeij\\dota\\hero_counter_lookup.csv"
    heroSynergy = "C:\\Users\\berke\\Documents\\UiA\\IKT110_Berkeij\\dota\\hero_synergy_lookup.csv"
    matches, n_heroes = load_data(matchCSV, playerCSV)
    counter_lookup = load_lookup_matrix(heroCounters, n_heroes)
    synergy_lookup = load_lookup_matrix(heroSynergy, n_heroes)

    learning_rates = [0.05, 0.01, 0.005, 0.15, 0.1]
    l2_values = [0.0]
    epoch_values = [10, 20, 30, 40, 50]

    best_acc = 0.0
    best_params: Tuple[float, float, int] | None = None
    best_predictor: DotaWinPredictor | None = None
    best_test_matches: List[Tuple[List[int], List[int], int]] | None = None

    for lr in learning_rates:
        for l2 in l2_values:
            for n_epochs in epoch_values:
                predictor = DotaWinPredictor(
                    n_heroes=n_heroes,
                    counter_lookup=counter_lookup,
                    synergy_lookup=synergy_lookup,
                )
                predictor.model.learning_rate = lr
                predictor.model.l2 = l2

                train_matches, test_matches = predictor.split_dataset(matches, train_ratio=0.8, random_state=0)
                predictor.train(train_matches, n_epochs=n_epochs)
                accuracy, _, _, _ = predictor.test(test_matches)
                print(f"lr={lr}, l2={l2}, epochs={n_epochs} -> accuracy={accuracy:.4f}")

                if accuracy > best_acc:
                    best_acc = accuracy
                    best_params = (lr, l2, n_epochs)
                    best_predictor = predictor
                    best_test_matches = test_matches

    if best_params is not None and best_predictor is not None and best_test_matches is not None:
        print("================================")
        accuracy, confusion_matrix, f1, recall = best_predictor.test(best_test_matches)
        print(
            "Best params: lr={:.0e}, l2={:.1f}, epochs={} with accuracy={:.4f}".format(
                best_params[0], best_params[1], best_params[2], best_acc
            )
        )
        print(f"Confusion Matrix:\n{confusion_matrix}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")

        final_predictor = DotaWinPredictor(
            n_heroes=n_heroes,
            counter_lookup=counter_lookup,
            synergy_lookup=synergy_lookup,
        )
        final_predictor.model.learning_rate = best_params[0]
        final_predictor.model.l2 = best_params[1]
        final_predictor.train(matches, n_epochs=best_params[2])
        final_predictor.save_model("dota_win_predictor_model.npz")
        print("Final model trained on full data with best hyperparameters and saved.")


if __name__ == "__main__":
    main()
