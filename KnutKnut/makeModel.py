"""
Author: Douwe Berkeij
Date: 23-09-2025
Clean route-specific travel time prediction models
"""

from queue import Queue
import concurrent.futures
import threading
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
import numpy as np
import pickle
import sys
import os
# Add parent directory to path to import fortunaAlgorithm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fortunaAlgorithm import fortuna_algorithm


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


def train_single_model_instance(X, y, formula, use_quadratic, use_trendline, route_name, run_id):
    """Train a single instance of a model - used for multithreading"""
    try:
        # Create completely isolated random state for this thread
        import os
        import random

        # Set multiple levels of random seeds to ensure isolation
        thread_seed = run_id * 1234 + int(time.time() * 1000000) % 1000000
        np.random.seed(thread_seed)
        random.seed(thread_seed)
        os.environ['PYTHONHASHSEED'] = str(thread_seed)

        # Add small random delay to further desynchronize threads
        time.sleep(np.random.uniform(0.001, 0.01))

        # Add randomness to Fortuna algorithm parameters to increase diversity
        random_factor = 0.1  # 10% variation

        if use_quadratic:
            # Randomize parameters slightly
            max_iter_rand = int(
                50000 * (1 + np.random.uniform(-random_factor, random_factor)))
            init_samples_rand = int(
                2000 * (1 + np.random.uniform(-random_factor, random_factor)))
            pop_size_rand = int(
                200 * (1 + np.random.uniform(-random_factor, random_factor)))
            offspring_rand = int(
                500 * (1 + np.random.uniform(-random_factor, random_factor)))
            evo_init_rand = 1.0 * \
                (1 + np.random.uniform(-random_factor, random_factor))

            optimal_params = fortuna_algorithm(X, y, formula_str=formula, loss_func="(y_true - y_pred)**2",
                                               max_iter=max_iter_rand, init_samples=init_samples_rand, pop_size=pop_size_rand,
                                               offspring_per_gen=offspring_rand, evo_str_init=evo_init_rand, evo_str_min=0.001,
                                               param_range=(-200, 200))
        elif use_trendline:
            # For trendline: a0=baseline, a1=amplitude, a2=period, a3=phase_shift
            max_iter_rand = int(
                40000 * (1 + np.random.uniform(-random_factor, random_factor)))
            init_samples_rand = int(
                2500 * (1 + np.random.uniform(-random_factor, random_factor)))
            pop_size_rand = int(
                250 * (1 + np.random.uniform(-random_factor, random_factor)))
            offspring_rand = int(
                400 * (1 + np.random.uniform(-random_factor, random_factor)))
            evo_init_rand = 1.2 * \
                (1 + np.random.uniform(-random_factor, random_factor))

            optimal_params = fortuna_algorithm(X, y, formula_str=formula, loss_func="(y_true - y_pred)**2",
                                               max_iter=max_iter_rand, init_samples=init_samples_rand, pop_size=pop_size_rand,
                                               offspring_per_gen=offspring_rand, evo_str_init=evo_init_rand, evo_str_min=0.001,
                                               param_range=(-300, 300))
        else:
            max_iter_rand = int(
                30000 * (1 + np.random.uniform(-random_factor, random_factor)))
            init_samples_rand = int(
                1500 * (1 + np.random.uniform(-random_factor, random_factor)))
            pop_size_rand = int(
                150 * (1 + np.random.uniform(-random_factor, random_factor)))
            offspring_rand = int(
                300 * (1 + np.random.uniform(-random_factor, random_factor)))
            evo_init_rand = 0.5 * \
                (1 + np.random.uniform(-random_factor, random_factor))

            optimal_params = fortuna_algorithm(X, y, formula_str=formula, loss_func="(y_true - y_pred)**2",
                                               max_iter=max_iter_rand, init_samples=init_samples_rand, pop_size=pop_size_rand,
                                               offspring_per_gen=offspring_rand, evo_str_init=evo_init_rand, evo_str_min=0.001,
                                               param_range=(-100, 100))

        # Calculate RMSE for this run
        y_pred = evaluate_model(X, optimal_params, formula)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        return {
            'params': optimal_params,
            'rmse': rmse,
            'run_id': run_id,
            'success': True
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'run_id': run_id
        }


def train_route_model(route_data, route_name, num_runs=100):
    """Train a single route model with pattern-specific approach"""

    if len(route_data) < 10:
        print(f"Insufficient data for {route_name}: {len(route_data)} samples")
        return None

    # Extract features and targets
    X = []
    y = []
    for entry in route_data:
        if route_name == "B->C->E":
            # For B->C->E, use minutes since midnight for trendline formula
            t = entry["minutes_since_midnight"]
        else:
            # Use centered departure time for better numerical stability
            t = entry["departure_centered"]
        X.append([t])
        y.append(entry["travel_time"])

    X = np.array(X)
    y = np.array(y)

    use_quadratic = False
    use_trendline = False
    formula = ""
    description = ""

    # Route-specific model configuration
    if route_name in ["A->C->D", "B->C->D"]:
        # Parabolic pattern: quadratic function
        formula = "a0 + a1*x0 + a2*x0*x0"
        description = "Quadratic (parabolic)"
        use_quadratic = True

    elif route_name == "B->C->E":
        # Trendline pattern: y = a - b * (((x - x_mean + phi) mod c) / c)
        # Use the original approach but fix in train_single_model_instance
        x_mean = np.mean(X[:, 0])  # Mean of minutes since midnight
        formula = f"a0 - a1 * (Mod(x0 - {x_mean} + a3, a2) / a2)"
        description = "Trendline (sawtooth with phase shift)"
        use_trendline = True

    elif route_name == "A->C->E":
        # Flat/low variance pattern: simple linear
        formula = "a0 + a1*x0"
        description = "Linear (flat trend)"
        use_quadratic = False

    # Use simple polynomial fitting instead of Fortuna for stability
    print(
        f"\nTraining {route_name} with {num_runs} runs using multithreading...")

    # Use multithreading to run multiple training sessions
    best_result = None
    best_rmse = float('inf')
    successful_runs = 0

    print(f"  Running {num_runs} training sessions sequentially...")

    # Sequential processing with proper random seeding
    for run_id in range(num_runs):
        # Set unique random seed for each run
        unique_seed = run_id * 7919 + int(time.time() * 1000000) % 1000000
        np.random.seed(unique_seed)

        try:
            result = train_single_model_instance(
                X, y, formula, use_quadratic, use_trendline, route_name, run_id)
            if result['success']:
                successful_runs += 1
                if result['rmse'] < best_rmse:
                    best_rmse = result['rmse']
                    best_result = result
                    print(
                        f"    Run {run_id+1}/{num_runs}: New best RMSE = {best_rmse:.3f}")

                # Progress update every 10 runs
                if (run_id + 1) % 10 == 0:
                    print(
                        f"    Completed {run_id+1}/{num_runs} runs, best RMSE so far: {best_rmse:.3f}")
            else:
                print(
                    f"    Run {run_id+1} failed: {result.get('error', 'Unknown error')}")
        except Exception as exc:
            print(f"    Run {run_id+1} generated an exception: {exc}")

    if best_result is None:
        print(f"✗ All training runs failed for {route_name}")
        return None

    # Use the best parameters found
    optimal_params = best_result['params']

    # Calculate final metrics using the best model
    y_pred = evaluate_model(X, optimal_params, formula)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    print(f"✓ Training complete for {route_name}")
    print(f"  Successful runs: {successful_runs}/{num_runs}")
    print(f"  Best RMSE: {rmse:.2f} minutes")
    print(f"  Best parameters: {optimal_params}")

    # Show sample predictions to verify reasonableness
    print(f"  Sample predictions:")
    for test_hour in [8, 12, 16]:
        if use_trendline:
            # For trendline, use minutes since midnight
            minutes = test_hour * 60
            x_mean = np.mean(X[:, 0])
            a0, a1, a2, a3 = optimal_params[0], optimal_params[1], abs(
                optimal_params[2]) + 1e-10, optimal_params[3]
            mod_term = ((minutes - x_mean + a3) % a2) / a2
            pred = a0 - a1 * mod_term
        else:
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
        "mean_travel_time": np.mean(y),
        "successful_runs": successful_runs,
        "total_runs": num_runs
    }


def evaluate_model(X, params, formula):
    """Evaluate model predictions"""
    predictions = []
    for x_row in X:
        x0 = x_row[0]  # x0 is now in hours for most, minutes for trendline

        if "x0*x0" in formula:  # Quadratic
            pred = params[0] + params[1]*x0 + params[2]*x0*x0
        elif "Mod(" in formula:  # Trendline (contains SymPy Mod function)
            # Trendline formula: a0 - a1 * (Mod(x0 - x_mean + a3, a2) / a2)
            # params: a0=baseline, a1=amplitude, a2=period, a3=phase_shift
            a0, a1, a2, a3 = params[0], params[1], abs(
                params[2]) + 1e-10, params[3]
            # Extract x_mean from the formula string (it's embedded in the formula)
            import re
            x_mean_match = re.search(r'x0 - ([\d.]+) \+ a3', formula)
            x_mean = float(x_mean_match.group(1)) if x_mean_match else 0

            mod_term = ((x0 - x_mean + a3) % a2) / a2
            pred = a0 - a1 * mod_term
        else:  # Linear (removed trigonometric)
            pred = params[0] + params[1]*x0

        predictions.append(pred)

    return np.array(predictions)


def predict_travel_time(dep_hour, dep_minute, route, models):
    """Predict travel time using route-specific model"""
    if route not in models or models[route] is None:
        return None

    model = models[route]
    params = model["params"]
    formula = model["formula"]

    if "Mod(" in formula:  # Trendline formula
        # Convert to minutes since midnight for trendline
        minutes = dep_hour * 60 + dep_minute

        # Extract x_mean from the formula string
        import re
        x_mean_match = re.search(r'x0 - ([\d.]+) \+ a3', formula)
        x_mean = float(x_mean_match.group(1)) if x_mean_match else 0

        # Apply trendline formula: a0 - a1 * (Mod(x0 - x_mean + a3, a2) / a2)
        a0, a1, a2, a3 = params[0], params[1], abs(
            params[2]) + 1e-10, params[3]
        mod_term = ((minutes - x_mean + a3) % a2) / a2
        prediction = a0 - a1 * mod_term

    else:
        # Convert to centered time (same as training data) for other models
        hours = dep_hour + dep_minute / 60.0
        centered_hours = hours - 12.0

        if "x0*x0" in formula:  # Quadratic
            prediction = params[0] + params[1]*centered_hours + \
                params[2]*centered_hours*centered_hours
        else:  # Linear
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


def plot_model_fits(processed_data, models):
    """Plot fitted models over actual data for all routes"""
    routes = ["A->C->D", "A->C->E", "B->C->D", "B->C->E"]
    colors = ['red', 'green', 'blue', 'magenta']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, route_name in enumerate(routes):
        route_id = {"A->C->D": 0, "A->C->E": 1,
                    "B->C->D": 2, "B->C->E": 3}[route_name]
        route_data = [
            entry for entry in processed_data if entry["route"] == route_id]

        if len(route_data) == 0 or models[route_name] is None:
            axes[i].text(0.5, 0.5, f'No data/model for {route_name}',
                         transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(f'{route_name} - No Model')
            continue

        # Extract data for plotting
        if route_name == "B->C->E":
            # Trendline uses minutes since midnight
            x_data = np.array([entry["minutes_since_midnight"]
                              for entry in route_data])
            x_label = 'Departure Time (minutes since midnight)'
            x_smooth = np.linspace(x_data.min(), x_data.max(), 500)
        else:
            # Other models use centered hours
            x_data = np.array([entry["departure_centered"]
                              for entry in route_data])
            x_label = 'Centered Departure Time (hours from noon)'
            x_smooth = np.linspace(x_data.min(), x_data.max(), 500)

        y_data = np.array([entry["travel_time"] for entry in route_data])

        # Plot actual data points
        axes[i].scatter(x_data, y_data, alpha=0.6, color=colors[i],
                        s=20, label='Actual Data')

        # Generate model predictions
        model = models[route_name]
        params = model["params"]
        formula = model["formula"]

        y_smooth = []
        for x in x_smooth:
            if "Mod(" in formula:  # Trendline formula
                # Extract x_mean from the formula string
                import re
                x_mean_match = re.search(r'x0 - ([\d.]+) \+ a3', formula)
                x_mean = float(x_mean_match.group(1)) if x_mean_match else 0

                a0, a1, a2, a3 = params[0], params[1], abs(
                    params[2]) + 1e-10, params[3]
                mod_term = ((x - x_mean + a3) % a2) / a2
                pred = a0 - a1 * mod_term
            elif "x0*x0" in formula:  # Quadratic
                pred = params[0] + params[1]*x + params[2]*x*x
            else:  # Linear
                pred = params[0] + params[1]*x

            y_smooth.append(pred)

        # Plot model fit
        axes[i].plot(x_smooth, y_smooth, color='darkred', linewidth=2,
                     linestyle='--', label=f'{model["description"]} Fit')

        # Add time labels for trendline (minutes since midnight)
        if route_name == "B->C->E":
            hour_ticks = np.arange(int(x_data.min()//60)*60,
                                   int(x_data.max()//60)*60 + 120, 120)
            hour_labels = [
                f"{int(h//60):02d}:{int(h % 60):02d}" for h in hour_ticks]
            axes[i].set_xticks(hour_ticks)
            axes[i].set_xticklabels(hour_labels, rotation=45)

        # Formatting
        axes[i].set_xlabel(x_label)
        axes[i].set_ylabel('Travel Time (minutes)')
        axes[i].set_title(f'{route_name} - {model["description"]}\n'
                          f'RMSE: {model["rmse"]:.1f} min, Samples: {model["samples"]}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        # Add formula text
        if "Mod(" in formula:  # Trendline
            formula_text = f'y = {params[0]:.1f} - {params[1]:.1f} × (Mod(x-{x_mean:.0f}+{params[3]:.1f}, {abs(params[2]):.1f})/{abs(params[2]):.1f})'
        elif "x0*x0" in formula:  # Quadratic
            formula_text = f'y = {params[0]:.1f} + {params[1]:.2f}×x + {params[2]:.3f}×x²'
        else:  # Linear
            formula_text = f'y = {params[0]:.1f} + {params[1]:.2f}×x'

        axes[i].text(0.02, 0.98, formula_text, transform=axes[i].transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.suptitle('Model Fits for All Routes', fontsize=16, y=1.02)
    plt.show()


if __name__ == "__main__":
    # Load and preprocess data
    with open('traffic.jsonl', 'r') as f:
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
            if 'successful_runs' in model and 'total_runs' in model:
                print(
                    f"  Training runs: {model['successful_runs']}/{model['total_runs']} successful")
            successful_models += 1
        else:
            print(f"{route_name}: FAILED")

    print(f"\n✓ {successful_models}/4 models trained successfully")

    # Plot model fits
    print("\nGenerating model fit plots...")
    plot_model_fits(processed_data, models)

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
            print(f"Best: {best_route} ({best_time:.1f} min)")
            for route, pred_time in sorted(all_predictions, key=lambda x: x[1]):
                print(f"  {route}: {pred_time:.1f} min")
        else:
            print("No predictions available")

    print(f"\nModels saved as 'knut_knut_clean_models.pkl'")
