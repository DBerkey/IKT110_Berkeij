"""
Author: Douwe Berkeij
Date: 23-09-2025
Clean route-specific travel time prediction models
"""

import sys
import os
# Add parent directory to path to import fortunaAlgorithm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fortunaAlgorithm import fortuna_algorithm
import pickle
import numpy as np
import json
import time
import pandas as pd




def preprocess(input_data):
    """Clean preprocessing with only essential features"""
    route_mapping = {"A->C->D": 0, "A->C->E": 1, "B->C->D": 2, "B->C->E": 3}

    for entry in input_data:
        # Route encoding
        entry["route"] = route_mapping[entry["road"]]

        # Parse times
        departure_time = time.strptime(entry["depature"], "%H:%M")
        arrival_time = time.strptime(entry["arrival"], "%H:%M")

        # Calculate travel time and departure minutes
        total_minutes = (arrival_time.tm_hour * 60 + arrival_time.tm_min) - \
            (departure_time.tm_hour * 60 + departure_time.tm_min)
        entry["travel_time"] = total_minutes
        entry["minutes_since_midnight"] = departure_time.tm_hour * \
            60 + departure_time.tm_min
        # Normalize departure time to hours for better model performance
        entry["departure_hours"] = entry["minutes_since_midnight"] / 60.0
        # Center around midday (12) for more stable quadratic behavior
        entry["departure_centered"] = entry["departure_hours"] - 12.0

        # Clean up
        entry.pop("road", None)
        entry.pop("depature", None)
        entry.pop("arrival", None)

    return input_data


def simple_polyfit(X, y, degree=2):
    """Simple polynomial fitting using numpy"""
    X_flat = X.flatten()
    if degree == 1:
        # Linear: y = a + bx
        A = np.column_stack([np.ones(len(X_flat)), X_flat])
    else:
        # Quadratic: y = a + bx + cx^2
        A = np.column_stack([np.ones(len(X_flat)), X_flat, X_flat**2])

    # Solve normal equations: A^T A x = A^T y
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    return coeffs


def train_route_model(route_data, route_name):
    """Train a single route model with pattern-specific approach"""

    if len(route_data) < 10:
        print(f"Insufficient data for {route_name}: {len(route_data)} samples")
        return None

    # Extract features and targets
    X = []
    y = []
    for entry in route_data:
        # Use centered departure time for better numerical stability
        t = entry["departure_centered"]
        X.append([t])
        y.append(entry["travel_time"])

    X = np.array(X)
    y = np.array(y)

    use_quadratic = False
    formula = ""
    description = ""

    # Route-specific model configuration
    if route_name in ["A->C->D", "B->C->D"]:
        # Parabolic pattern: quadratic function
        formula = "a0 + a1*x0 + a2*x0*x0"
        description = "Quadratic (parabolic)"
        use_quadratic = True

    elif route_name in ["A->C->E", "B->C->E"]:
        # Flat/low variance pattern: simple linear
        formula = "a0 + a1*x0"
        description = "Linear (flat trend)"
        use_quadratic = False

    # Use simple polynomial fitting instead of Fortuna for stability
    try:
        if use_quadratic:
            optimal_params = simple_polyfit(X, y, degree=2)
        else:
            optimal_params = simple_polyfit(X, y, degree=1)

        # Calculate RMSE for reporting
        y_pred = evaluate_model(X, optimal_params, formula)
        rmse = np.sqrt(np.mean((y - y_pred)**2))

        print(f"✓ Training complete. RMSE: {rmse:.2f} minutes")
        print(f"  Parameters: {optimal_params}")

        # Show sample predictions to verify reasonableness
        print(f"  Sample predictions:")
        for test_hour in [8, 12, 16]:
            centered_time = test_hour - 12
            if use_quadratic:
                pred = optimal_params[0] + optimal_params[1]*centered_time + \
                    optimal_params[2]*centered_time*centered_time
            else:
                pred = optimal_params[0] + optimal_params[1]*centered_time
            print(f"    {test_hour:02d}:00 -> {pred:.1f} min")

        return {
            "params": optimal_params,
            "formula": formula,
            "description": description,
            "rmse": rmse,
            "samples": len(route_data),
            "mean_travel_time": np.mean(y)
        }

    except Exception as e:
        print(f"✗ Training failed for {route_name}: {e}")
        return None


def evaluate_model(X, params, formula):
    """Evaluate model predictions"""
    predictions = []
    for x_row in X:
        x0 = x_row[0]  # x0 is now in hours (7-17 range)

        if "x0*x0" in formula:  # Quadratic
            pred = params[0] + params[1]*x0 + params[2]*x0*x0
        else:  # Linear (removed trigonometric)
            pred = params[0] + params[1]*x0

        predictions.append(pred)

    return np.array(predictions)


def predict_travel_time(dep_hour, dep_minute, route, models):
    """Predict travel time using route-specific model"""
    if route not in models or models[route] is None:
        return None

    model = models[route]
    # Convert to centered time (same as training data)
    hours = dep_hour + dep_minute / 60.0
    centered_hours = hours - 12.0

    # Apply route-specific formula
    params = model["params"]
    formula = model["formula"]

    if "x0*x0" in formula:  # Quadratic
        prediction = params[0] + params[1]*centered_hours + \
            params[2]*centered_hours*centered_hours
    else:  # Linear (no more trigonometric)
        prediction = params[0] + params[1]*centered_hours

    # Minimum 50 minutes (more realistic based on data)
    return max(prediction, 50)


def get_best_route(dep_hour, dep_minute, models):
    """Find best route at given time"""
    predictions = []

    for route_name in ["A->C->D", "A->C->E", "B->C->D", "B->C->E"]:
        pred_time = predict_travel_time(
            dep_hour, dep_minute, route_name, models)
        if pred_time is not None:
            predictions.append((route_name, pred_time))

    if not predictions:
        return None, None, []

    best_route, best_time = min(predictions, key=lambda x: x[1])
    return best_route, best_time, predictions


def train_all_models(processed_data):
    """Train models for all routes"""
    models = {}

    for route_name in ["A->C->D", "A->C->E", "B->C->D", "B->C->E"]:
        route_id = {"A->C->D": 0, "A->C->E": 1,
                    "B->C->D": 2, "B->C->E": 3}[route_name]
        route_data = [
            entry for entry in processed_data if entry["route"] == route_id]

        models[route_name] = train_route_model(route_data, route_name)

    return models


if __name__ == "__main__":
    # Load and preprocess data
    with open('KnutKnut/traffic.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]

    processed_data = preprocess(data)

    # Split data
    length = len(processed_data)
    test_data = processed_data[:length // 10]
    train_data = processed_data[length // 10:]

    print(
        f"Training on {len(train_data)} samples, testing on {len(test_data)} samples")

    # Train models
    models = train_all_models(train_data)

    # Save models
    with open('knut_knut_clean_models.pkl', 'wb') as f:
        pickle.dump(models, f)

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    successful_models = 0
    for route_name, model in models.items():
        if model is not None:
            print(f"{route_name}: {model['description']}")
            print(
                f"  RMSE: {model['rmse']:.2f} min, Mean: {model['mean_travel_time']:.1f} min")
            print(f"  Samples: {model['samples']}")
            successful_models += 1
        else:
            print(f"{route_name}: FAILED")

    print(f"\n✓ {successful_models}/4 models trained successfully")

    # Test predictions
    print("\n" + "="*60)
    print("PREDICTION TESTS")
    print("="*60)

    test_times = [(8, 30), (12, 0), (16, 30), (20, 0)]

    for hour, minute in test_times:
        print(f"\nTime: {hour:02d}:{minute:02d}")
        best_route, best_time, all_predictions = get_best_route(
            hour, minute, models)

        if best_route:
            print(f"🏆 Best: {best_route} ({best_time:.1f} min)")
            for route, pred_time in sorted(all_predictions, key=lambda x: x[1]):
                print(f"  {route}: {pred_time:.1f} min")
        else:
            print("No predictions available")

    print(f"\n✅ Models saved as 'knut_knut_clean_models.pkl'")
