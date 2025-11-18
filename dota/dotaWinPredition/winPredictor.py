"""
Author: Douwe Berkeij
Date: 18-11-2025
Description: This module contains functions to predict the outcome of a Dota 2 match based on hero selections.

to run use & C:/Users/berke/anaconda3/envs/IKT110/python.exe -m dota.dotaWinPredition.winPredictor
"""

from typing import List, Tuple
import numpy as np
# Support importing when running as a package or as a script: try absolute first, fall back to relative.
from .dataLoader import load_data
from .logistic_model import LogisticRegressionSGD

class DotaWinPredictor:
    """Predicts the outcome of a Dota 2 match based on hero selections."""

    def __init__(self, n_heroes: int):
        self.n_heroes = n_heroes
        self.model = LogisticRegressionSGD(n_features=2 * n_heroes, learning_rate=1e-3, l2=1e-4)

    def _encode_match(self, radiant_heroes: List[int], dire_heroes: List[int]) -> np.ndarray:
        """Encodes the hero selections into a feature vector."""
        x = np.zeros(2 * self.n_heroes)
        for hero_id in radiant_heroes:
            x[hero_id] = 1
        for hero_id in dire_heroes:
            x[self.n_heroes + hero_id] = 1
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
    
    def split_dataset(self, matches: List[Tuple[List[int], List[int], str]], train_ratio: float = 0.8) -> Tuple[List[Tuple[List[int], List[int], str]], List[Tuple[List[int], List[int], str]]]:
        """Splits the dataset into training and testing sets."""
        n_train = int(len(matches) * train_ratio)
        train_matches = matches[:n_train]
        test_matches = matches[n_train:]
        return train_matches, test_matches

    def train(self, matches: List[Tuple[List[int], List[int], int]], n_epochs: int = 100) -> None:
        """Trains the win predictor model."""
        X, y = self.prepare_dataset(matches)
        self.model.fit(X, y, n_epochs=n_epochs)

    def test(self, matches: List[Tuple[List[int], List[int], int]]) -> float:
        """Tests the win predictor model and returns accuracy."""
        X, y = self.prepare_dataset(matches)
        # compute predicted probabilities using the current model parameters
        logits = X @ self.model.w + self.model.b
        probs = self.model._sigmoid(logits)
        preds = (probs >= 0.5).astype(int).flatten()
        accuracy = float(np.mean(preds == y))
        # confusion matrix
        tp = np.sum((preds == 1) & (y == 1))
        tn = np.sum((preds == 0) & (y == 0))
        fp = np.sum((preds == 1) & (y == 0))
        fn = np.sum((preds == 0) & (y == 1))
        confusion_matrix = np.array([[tp, fp], [fn, tn]])
        return accuracy, confusion_matrix

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
        prob = self.model._sigmoid(x @ self.model.w + self.model.b)
        return prob.item()
    
if __name__ == "__main__":
    matchCSV = "C:\\Users\\berke\\Documents\\UiA\\IKT110_Berkeij\\dota\\new_ranked_matches.csv"
    playerCSV = "C:\\Users\\berke\\Documents\\UiA\\IKT110_Berkeij\\dota\\new_ranked_players.csv"
    matches, n_heroes = load_data(matchCSV, playerCSV)

    predictor = DotaWinPredictor(n_heroes=n_heroes)
    train_matches, test_matches = predictor.split_dataset(matches, train_ratio=0.8)
    predictor.train(train_matches, n_epochs=100)
    accuracy, confusion_matrix = predictor.test(test_matches)
    print(f"Test accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix)

    print("================================")
    predictor.save_model("dota_win_predictor_model.npz")
    print("Model trained and saved.")    
