"""
Author: Douwe Berkeij & Urs Pfrommer
Date: 23-09-2025
AI uses: For this code, GitHUb Copilot was used to assist writing by suggesting
code snippets and functions. No AI was used to sugest or write the final models or
formulas. The code was manually reviewed and edited to ensure correctness and
suitability for the task.
"""

import json
import re
import sys
import time
import pickle
import os
import random
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Navigate to the parent directory to have access to fortunaAlgorithm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fortunaAlgorithm import fortuna_algorithm


def preprocess(input_data):
    """
    params:
        input_data: list of dictonaries with keys:
            - road: str, one of "A->C->D", "A->C->E", "B->C->D", "B->C->E"
            - depature: str, "HH:MM"
            - arrival: str, "HH:MM"
    returns:
        processed_data: list of dictonaries with keys:
            - route: int, 0-3 encoding the route
            - travel_time: int, travel time in minutes
            - minutes_since_midnight: int, departure time in minutes since midnight
            - departure_hours: float, departure time in hours since midnight (for model input)
            - departure_centered: float, departure time centered around midday (for model input)
    """
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

        # Clean up the dictionary to keep only relevant keys
        entry.pop("road", None)
        entry.pop("depature", None)
        entry.pop("arrival", None)

    return input_data


def train_single_model_instance(x, y, formula, apply_quadratic_formula,
                                apply_piecewise_linear_periodic_formula, run_id):
    """
    params:
        x: np.array of shape (n_samples, n_features)
        y: np.array of shape (n_samples,)
        formula: str, formula string for fortunaAlgorithm
        apply_quadratic_formula: bool, whether to use quadratic formula
        apply_piecewise_linear_periodic_formula: bool, whether to use 
            piecewise linear periodic formula
        run_id: int, unique identifier for this training run
    returns:
        dict with keys:
            - "model": the trained model
            - "params": the parameters used for training
            - "run_id": the unique identifier for this training run
    """
    try:
        # Set multiple levels of random seeds to ensure isolation
        thread_seed = run_id * 1234 + int(time.time() * 1000000) % 1000000
        np.random.seed(thread_seed)
        random.seed(thread_seed)
        os.environ['PYTHONHASHSEED'] = str(thread_seed)

        # Add small random delay to desynchronize threads
        time.sleep(np.random.uniform(0.001, 0.01))

        # Add randomness to Fortuna algorithm parameters to increase diversity
        random_factor = 0.1

        if apply_quadratic_formula:
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

            optimal_params = fortuna_algorithm(x, y,
                                               formula_str=formula,
                                               loss_func="(y_true - y_pred)**2",
                                               max_iter=max_iter_rand,
                                               init_samples=init_samples_rand,
                                               pop_size=pop_size_rand,
                                               offspring_per_gen=offspring_rand,
                                               evo_str_init=evo_init_rand,
                                               evo_str_min=0.001,
                                               param_range=(-200, 200))
        elif apply_piecewise_linear_periodic_formula:
            # Randomize parameters slightly
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

            optimal_params = fortuna_algorithm(x, y, formula_str=formula,
                                               loss_func="(y_true - y_pred)**2",
                                               max_iter=max_iter_rand,
                                               init_samples=init_samples_rand,
                                               pop_size=pop_size_rand,
                                               offspring_per_gen=offspring_rand,
                                               evo_str_init=evo_init_rand, evo_str_min=0.001,
                                               param_range=(-300, 300))
        else:
            # Randomize parameters slightly
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

            optimal_params = fortuna_algorithm(x, y, formula_str=formula,
                                               loss_func="(y_true - y_pred)**2",
                                               max_iter=max_iter_rand,
                                               init_samples=init_samples_rand,
                                               pop_size=pop_size_rand,
                                               offspring_per_gen=offspring_rand,
                                               evo_str_init=evo_init_rand,
                                               evo_str_min=0.001,
                                               param_range=(-100, 100))

        # Calculate Root Mean Square Error (RMSE) for this run
        y_pred = evaluate_model(x, optimal_params, formula)
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


def train_route_model(route_data, route_str, num_runs=100):
    """
    params:
        route_data: list of dictonaries for a specific route with keys:
            - route: int, 0-3 encoding the route
            - travel_time: int, travel time in minutes
            - minutes_since_midnight: int, departure time in minutes since midnight
            - departure_hours: float, departure time in hours since midnight (for model input)
            - departure_centered: float, departure time centered around midday (for model input)
        route_str: str, one of "A->C->D", "A->C->E", "B->C->D", "B->C->E"
        num_runs: int, number of training runs to perform
    returns:
        dict with keys:
            - "params": the best parameters found
            - "formula": the formula used
            - "description": description of the model type
            - "rmse": RMSE of the best model
            - "samples": number of samples used for training
            - "successful_runs": number of successful training runs
            - "total_runs": total number of training runs attempted
    """

    if len(route_data) < 10:
        print(f"Insufficient data for {route_str}: {len(route_data)} samples")
        return None

    # Extract features and targets
    x_axis_data = []
    y_axis_data = []
    for entry in route_data:
        if route_str == "B->C->E":
            # For B->C->E, use minutes since midnight for piecewise linear, periodic function
            time_data = entry["minutes_since_midnight"]
        else:
            # Use centered departure time because of the centered nature of the
            # data used for quadratic and linear models
            time_data = entry["departure_centered"]
        x_axis_data.append([time_data])
        y_axis_data.append(entry["travel_time"])

    x_axis_data = np.array(x_axis_data)
    y_axis_data = np.array(y_axis_data)

    apply_quadratic_formula = False
    apply_piecewise_linear_periodic_formula = False
    formula = ""
    description = ""

    # Route-specific model configuration
    if route_str in ["A->C->D", "B->C->D"]:
        # Formula: y = a0 + a1*x0 + a2*x0^2
        formula = "a0 + a1*x0 + a2*x0*x0"
        description = "Quadratic (parabolic)"
        apply_quadratic_formula = True

    elif route_str == "B->C->E":
        # Formula: y = a - b * (((x - x_mean + phi) mod c) / c)
        x_mean = np.mean(x_axis_data[:, 0])
        formula = f"a0 - a1 * (Mod(x0 - {x_mean} + a3, a2) / a2)"
        description = "piecewise linear, periodic function (sawtooth)"
        apply_piecewise_linear_periodic_formula = True

    elif route_str == "A->C->E":
        # Formula: y = a0 + a1*x0
        formula = "a0 + a1*x0"
        description = "Linear (flat)"
        apply_quadratic_formula = False

    best_result = None
    best_rmse = float('inf')
    successful_runs = 0

    # Run multiple training instances to find the best model
    for run_id in range(num_runs):
        # Set unique random seed for each run
        unique_seed = run_id * 7919 + int(time_data * 1000000) % 1000000
        np.random.seed(unique_seed)

        try:
            result = train_single_model_instance(
                x_axis_data, y_axis_data, formula, apply_quadratic_formula,
                apply_piecewise_linear_periodic_formula, run_id)
            if result['success']:
                successful_runs += 1
                if result['rmse'] < best_rmse:
                    best_rmse = result['rmse']
                    best_result = result
            else:
                print(
                    f"    Run {run_id+1} failed: {result.get('error', 'Unknown error')}")
        except Exception as exc:
            print(f"    Run {run_id+1} generated an exception: {exc}")

    if best_result is None:
        print(f"All training runs failed for {route_str}")
        return None

    # Use the best parameters found
    optimal_training_parameters = best_result['params']

    # Calculate final metrics using the best model
    y_pred = evaluate_model(x_axis_data, optimal_training_parameters, formula)
    rmse = np.sqrt(np.mean((y_axis_data - y_pred) ** 2))

    print(f"  Training complete for {route_str}")
    print(f"  Successful runs: {successful_runs}/{num_runs}")
    print(f"  Best RMSE: {rmse:.2f} minutes")
    print(f"  Best parameters: {optimal_training_parameters}")

    return {
        "params": optimal_training_parameters,
        "formula": formula,
        "description": description,
        "rmse": rmse,
        "samples": len(route_data),
        "mean_travel_time": np.mean(y_axis_data),
        "successful_runs": successful_runs,
        "total_runs": num_runs
    }


def evaluate_model(x_axis_data, optimal_model_parameters, formula):
    """
    params:
        x_axis_data: np.array of shape (n_samples, n_features)
        optimal_model_parameters: list or np.array of model parameters
        formula: str, formula string used for training
    returns:
        np.array of shape (n_samples,) with predicted values
    """
    predictions = []
    for x_row in x_axis_data:
        x0 = x_row[0]
        if "x0*x0" in formula:
            # Quadratic: y = a0 + a1*x0 + a2*x0^2
            # params: a0=intercept, a1=slope, a2=quadratic_term
            pred = (optimal_model_parameters[0] + optimal_model_parameters[1]*x0 +
                    optimal_model_parameters[2]*x0*x0)
        elif "Mod(" in formula:
            # piecewise linear, periodic function: a0 - a1 * (Mod(x0 - x_mean + a3, a2) / a2)
            # params: a0=baseline, a1=amplitude, a2=period, a3=phase_shift
            a0, a1, a2, a3 = optimal_model_parameters[0], optimal_model_parameters[1], abs(
                optimal_model_parameters[2]) + 1e-10, optimal_model_parameters[3]
            x_mean_match = re.search(r'x0 - ([\d.]+) \+ a3', formula)
            x_mean = float(x_mean_match.group(1)) if x_mean_match else 0

            mod_term = ((x0 - x_mean + a3) % a2) / a2
            pred = a0 - a1 * mod_term
        else:
            # Linear: y = a0 + a1*x0
            # params: a0=intercept, a1=slope
            pred = optimal_model_parameters[0] + optimal_model_parameters[1]*x0

        predictions.append(pred)

    return np.array(predictions)


def predict_travel_time(dep_hour, dep_minute, route_str, models_dict):
    """
    params:
        dep_hour: int, hour of departure (0-23)
        dep_minute: int, minute of departure (0-59)
        route_str: str, one of "A->C->D", "A->C->E", "B->C->D", "B->C->E"
        models_dict: dict with trained models for each route
    returns:
        float, predicted travel time in minutes with an minimum of 50 minutes, 
            or None if model is not available
    """
    if route_str not in models_dict or models_dict[route_str] is None:
        return None

    predictive_model = models_dict[route_str]
    params = predictive_model["params"]
    formula = predictive_model["formula"]

    if "Mod(" in formula:
        # piecewise linear, periodic function: a0 - a1 * (Mod(x0 - x_mean + a3, a2) / a2)
        # params: a0=baseline, a1=amplitude, a2=period, a3=phase_shift
        minutes = dep_hour * 60 + dep_minute

        x_mean_match = re.search(r'x0 - ([\d.]+) \+ a3', formula)
        x_mean = float(x_mean_match.group(1)) if x_mean_match else 0

        a0, a1, a2, a3 = params[0], params[1], abs(
            params[2]) + 1e-10, params[3]
        mod_term = ((minutes - x_mean + a3) % a2) / a2
        prediction = a0 - a1 * mod_term

    else:
        hours = dep_hour + dep_minute / 60.0
        centered_hours = hours - 12.0

        if "x0*x0" in formula:
            # Quadratic: y = a0 + a1*x0 + a2*x0^2
            # params: a0=intercept, a1=slope, a2=quadratic_term
            prediction = params[0] + params[1]*centered_hours + \
                params[2]*centered_hours*centered_hours
        else:
            # Linear: y = a0 + a1*x0
            # params: a0=intercept, a1=slope
            prediction = params[0] + params[1]*centered_hours

    return max(prediction, 50) # Minimum travel time of 50 minutes


def get_best_route(dep_hour, dep_minute, models_dict):
    """
    params:
        dep_hour: int, hour of departure (0-23)
        dep_minute: int, minute of departure (0-59)
        models_dict: dict with trained models for each route
    returns:
        best_route: str, route with the shortest predicted travel time, or None if no predictions
        best_time: float, predicted travel time for the best route, or None if no predictions
        predictions: list of tuples (route_str, predicted_time) for all routes 
            with valid predictions
    """
    predictions = []

    for route_identifiers in ["A->C->D", "A->C->E", "B->C->D", "B->C->E"]:
        predicted_travel_time = predict_travel_time(
            dep_hour, dep_minute, route_identifiers, models_dict)
        if predicted_travel_time is not None:
            predictions.append((route_identifiers, predicted_travel_time))

    if not predictions:
        return None, None, []

    fastest_route, fastest_route_time = min(predictions, key=lambda x: x[1])
    return fastest_route, fastest_route_time, predictions


def train_all_models(preprocessed_data_list):
    """
    params:
        preprocessed_data_list: list of dictonaries with preprocessed data
    returns:
        models_dict: dict with keys
            - route names as keys ("A->C->D", etc.)
            - values are the trained model dicts or None if training failed
    """
    models_dict = {}

    for route_str in ["A->C->D", "A->C->E", "B->C->D", "B->C->E"]:
        route_id = {"A->C->D": 0, "A->C->E": 1,
                    "B->C->D": 2, "B->C->E": 3}[route_str]
        route_data = [
            entry for entry in preprocessed_data_list if entry["route"] == route_id]

        models_dict[route_str] = train_route_model(route_data, route_str)

    return models_dict


def plot_model_fits(processed_data_list, models_dict):
    """Plot fitted models over actual data for all routes"""
    routes = ["A->C->D", "A->C->E", "B->C->D", "B->C->E"]
    colors = ['red', 'green', 'blue', 'magenta']

    _, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, route_str in enumerate(routes):
        route_id = {"A->C->D": 0, "A->C->E": 1,
                    "B->C->D": 2, "B->C->E": 3}[route_str]
        route_data = [
            entry for entry in processed_data_list if entry["route"] == route_id]

        if len(route_data) == 0 or models_dict[route_str] is None:
            axes[i].text(0.5, 0.5, f'No data/model for {route_str}',
                         transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(f'{route_str} - No Model')
            continue

        if route_str == "B->C->E":
            # The piecewise linear, periodic function uses minutes since midnight
            x_data = np.array([entry["minutes_since_midnight"]
                              for entry in route_data])
            x_label = 'Departure Time (minutes since midnight)'
            x_smooth = np.linspace(x_data.min(), x_data.max(), 500)
        else:
            # The quadratic and linear models use centered departure time
            x_data = np.array([entry["departure_centered"]
                              for entry in route_data])
            x_label = 'Centered Departure Time (hours from noon)'
            x_smooth = np.linspace(x_data.min(), x_data.max(), 500)

        y_data = np.array([entry["travel_time"] for entry in route_data])

        # Plot the data points
        axes[i].scatter(x_data, y_data, alpha=0.6, color=colors[i],
                        s=20, label='Actual Data')

        # Generate model predictions
        prediction_model = models_dict[route_str]
        params = prediction_model["params"]
        formula = prediction_model["formula"]

        y_smooth = []
        for x in x_smooth:
            if "Mod(" in formula:  # Trendline formula
                # Extract x_mean from the formula string
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
                     linestyle='--', label=f'{prediction_model["description"]} Fit')

        # Add time labels for trendline (minutes since midnight)
        if route_str == "B->C->E":
            hour_ticks = np.arange(int(x_data.min()//60)*60,
                                   int(x_data.max()//60)*60 + 120, 120)
            hour_labels = [
                f"{int(h//60):02d}:{int(h % 60):02d}" for h in hour_ticks]
            axes[i].set_xticks(hour_ticks)
            axes[i].set_xticklabels(hour_labels, rotation=45)

        # Formatting
        axes[i].set_xlabel(x_label)
        axes[i].set_ylabel('Travel Time (minutes)')
        plot_title = f'{route_str} - {prediction_model["description"]}\n' \
                      f'RMSE: {prediction_model["rmse"]:.1f} min, ' \
                      f'Samples: {prediction_model["samples"]}'
        axes[i].set_title(plot_title)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        # Add formula text
        if "Mod(" in formula:
            # piecewise linear, periodic function: a0 - a1 * (Mod(x0 - x_mean + a3, a2) / a2)
            # params: a0=baseline, a1=amplitude, a2=period, a3=phase_shift
            formula_text = (f'y = {params[0]:.1f} - {params[1]:.1f} ' +
                f'× (Mod(x-{x_mean:.0f}+{params[3]:.1f}, ' +
                f'{abs(params[2]):.1f})/{abs(params[2]):.1f})')
        elif "x0*x0" in formula:
            # Quadratic: y = a0 + a1*x0 + a2*x0^2
            # params: a0=intercept, a1=slope, a2=quadratic_term
            formula_text = f'y = {params[0]:.1f} + {params[1]:.2f}×x + {params[2]:.3f}×x²'
        else:
            # Linear: y = a0 + a1*x0
            # params: a0=intercept, a1=slope
            formula_text = f'y = {params[0]:.1f} + {params[1]:.2f}×x'

        axes[i].text(0.02, 0.98, formula_text, transform=axes[i].transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.suptitle('Model Fits for All Routes', fontsize=16, y=1.02)
    plt.show()


if __name__ == "__main__":
    # Load and preprocess data
    with open('KnutKnut/traffic.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    processed_data = preprocess(data)

    # Split data
    length = len(processed_data)
    test_data = processed_data[:length // 10]
    train_data = processed_data[length // 10:]

    print(f"Training on {len(train_data)} samples,",
          f" testing on {len(test_data)} samples")

    # Train models
    models = train_all_models(train_data)

    # Save models in an pickle file
    FILENAME_PICKLE = f'knut_knut_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    with open(FILENAME_PICKLE, 'wb') as f:
        pickle.dump(models, f)

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    SUCCESSFUL_MODELS = 0
    for route_name, model in models.items():
        if model is not None:
            print(f"{route_name}: {model['description']}")
            print(
                f"  RMSE: {model['rmse']:.2f} min, Mean: {model['mean_travel_time']:.1f} min")
            print(f"  Samples: {model['samples']}")
            if 'successful_runs' in model and 'total_runs' in model:
                print(
                    f"  Training runs: {model['successful_runs']}/{model['total_runs']} successful")
            SUCCESSFUL_MODELS += 1
        else:
            print(f"{route_name}: FAILED")

    print(f"\n{SUCCESSFUL_MODELS} models trained successfully")

    # Calculate the average time saved by using the predictions instead of a random route
    # for all times between 06:00 and 22:00
    total_time_saved = 0
    all_departure_times = []
    for hour in range(6, 22):
        for minute in range(0, 60, 5):
            all_departure_times.append((hour, minute))
    for dep_hour, dep_minute in all_departure_times:
        fastest_route, fastest_time, predictions = get_best_route(
            dep_hour, dep_minute, models)
        if fastest_time is None:
            continue
        avg_random_time = np.mean([time for _, time in predictions])
        time_saved = avg_random_time - fastest_time
        total_time_saved += time_saved

    average_time_saved = total_time_saved / len(all_departure_times)
    print(f"\nAverage time saved by using predictions: {average_time_saved:.2f} minutes")

    # Plot model fits
    plot_model_fits(processed_data, models)

    print(f"\nModels saved as '{FILENAME_PICKLE}'")
