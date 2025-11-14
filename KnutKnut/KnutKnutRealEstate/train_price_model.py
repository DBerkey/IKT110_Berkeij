import math
import pickle
from pathlib import Path

import numpy as np

from data_loader import build_dataset
from linear_model import LinearRegressionSGD
from logistic_model import LogisticRegressionSGD


def standardize_train(X: np.ndarray):
    """Standardize columns of X, returning X_scaled, mean, std.

    std values that are 0 or NaN are set to 1 (no scaling).
    Missing values (NaN) are imputed with column mean.
    """
    X = np.asarray(X, dtype=float)
    col_mean = np.nanmean(X, axis=0)
    # Impute NaNs
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])

    col_std = np.nanstd(X, axis=0)
    col_std[(col_std == 0) | np.isnan(col_std)] = 1.0

    X_scaled = (X - col_mean) / col_std
    return X_scaled, col_mean, col_std


def train_price_model():
    X, y_log, feature_names = build_dataset()

    # Shuffle and split into train / test
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.default_rng(0).shuffle(indices)

    split = int(0.8 * n_samples)
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_log[train_idx], y_log[test_idx]

    X_train_scaled, mean, std = standardize_train(X_train)

    # Apply same scaling to test (with NaN imputation using train means)
    X_test = np.asarray(X_test, dtype=float)
    inds_test = np.where(np.isnan(X_test))
    if inds_test[0].size > 0:
        X_test[inds_test] = np.take(mean, inds_test[1])
    std_safe = std.copy()
    std_safe[(std_safe == 0) | np.isnan(std_safe)] = 1.0
    X_test_scaled = (X_test - mean) / std_safe

    model = LinearRegressionSGD(
        n_features=X_train_scaled.shape[1],
        learning_rate=1e-3,
        l2=1e-4,
        batch_size=32,
        shuffle=True,
        random_state=0,
    )

    model.fit(X_train_scaled, y_train, n_epochs=50)

    # Evaluate
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.exp(y_pred_log)
    y_true = np.exp(y_test)

    rmse = math.sqrt(float(np.mean((y_pred - y_true) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100.0

    print(f"Test RMSE: {rmse:,.0f}")
    print(f"Test MAPE: {mape:.2f}%")

    # Show a few predictions
    for i in range(5):
        print(f"True: {y_true[i]:,.0f}, Pred: {y_pred[i]:,.0f}")

    # --- Train logistic model for sale-within-X-days probability ---
    # We already have X with NaNs; reuse it so indices match feature_names
    threshold_days = 30.0

    # We don't have access to raw houses list here easily, so we approximate
    # labels using the scaled train/test indices and the days_on_marked feature
    # which is included in X.
    try:
        days_idx = feature_names.index("days_on_marked")
        sold_flag_idx = feature_names.index("sold_flag")
    except ValueError:
        days_idx = None
        sold_flag_idx = None

    if (days_idx is not None) and (sold_flag_idx is not None):
        # Recreate labels: y_cls = 1 if sold and days_on_marked <= threshold
        # We don't have the original 'sold' column here, but we encoded
        # sold_flag as a numeric feature.
        sold_flag = X[:, sold_flag_idx]
        days = X[:, days_idx]
        y_cls = ((sold_flag > 0.5) & (days <= threshold_days)).astype(float)

        # Use same scaling statistics and impute NaNs as for training
        X_full = np.asarray(X, dtype=float)
        inds_full = np.where(np.isnan(X_full))
        if inds_full[0].size > 0:
            X_full[inds_full] = np.take(mean, inds_full[1])
        X_full_scaled = (X_full - mean) / std_safe

        clf = LogisticRegressionSGD(
            n_features=X_full_scaled.shape[1],
            learning_rate=1e-3,
            l2=1e-4,
            batch_size=32,
            shuffle=True,
            random_state=0,
        )
        clf.fit(X_full_scaled, y_cls, n_epochs=50)
        cls_w, cls_b = clf.parameters()
    else:
        cls_w = np.zeros_like(model.w)
        cls_b = 0.0
        threshold_days = 30.0

    # Save model and preprocessing metadata for dashboard use
    artifacts = {
        "weights": model.w,
        "bias": model.b,
        "mean": mean,
        "std": std_safe,
        "feature_names": feature_names,
        "cls_weights": cls_w,
        "cls_bias": cls_b,
        "cls_threshold_days": threshold_days,
    }

    out_path = Path(__file__).with_name("price_model.pkl")
    with out_path.open("wb") as f:
        pickle.dump(artifacts, f)
    print(f"Saved model artifacts (regression + classification) to {out_path}")

if __name__ == "__main__":
    train_price_model()
