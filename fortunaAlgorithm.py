"""
Author: Douwe Berkeij
Date: 22-08-2025
Description: Fortuna Algorithm for curve fitting using sympy and numpy. It is a very bad and inefficient algorithm.
"""

import sympy as sp
import numpy as np
import argparse as ap
import pandas as pd
import re

def loss_func_eval(y_true_np, y_pred_np, loss_func_sympy):
    losses = loss_func_sympy(y_true_np, y_pred_np) 
    return np.mean(losses)

def fit_func(x, *param_values, func, variables):
    # Prepare the variable values in the correct order for the lambdified function
    if isinstance(x, np.ndarray) and x.ndim == 2:
        # x is (N, num_vars), map columns to variable names
        # variables is a list of sympy symbols in the correct order
        var_values = []
        for i, var in enumerate(variables):
            # If there are fewer columns than variables, fill with zeros
            if i < x.shape[1]:
                var_values.append(x[:, i])
            else:
                var_values.append(np.zeros(x.shape[0]))
        return func(*var_values, *param_values)
    else:
        # fallback for 1D or scalar x
        return func(x, *param_values)

def fortuna_algorithm(x_data, y_data, formula_str, loss_func, max_iter=100000, 
                     init_samples=1000, pop_size=100, offspring_per_gen=500,
                     evo_str_init=10, evo_str_min=0.02, param_range=(-10, 10)):
    """
    Enhanced Fortuna Algorithm with evolutionary strategy for curve fitting.
    
    Parameters:
    - x_data, y_data: input data arrays
    - formula_str: mathematical formula as string
    - loss_func: loss function as string
    - max_iter: maximum amount of repeated function evaluations
    - init_samples: initial random population size
    - pop_size: size of parent population
    - offspring_per_gen: number of offspring per generation
    - evo_str_init: initial mutation strength
    - evo_str_min: minimum mutation strength
    - param_range: parameter search range tuple
    """
    # Automatically detect all variables of the form 'x', 'x1', 'x2', ..., 'xN'
    # Find all variable names in the formula string
    var_names = set(re.findall(r'\bx\d*\b', formula_str))
    # Ensure 'x' is included if present
    if 'x' in formula_str:
        var_names.add('x')
    # Create sympy symbols for all detected variables
    variables = [sp.symbols(name) for name in sorted(var_names, key=lambda s: (len(s), s))]
    # Parse the formula string into a sympy expression
    expr = sp.sympify(formula_str, locals={name: var for name, var in zip(var_names, variables)})

    # Extract parameter symbols (all symbols except detected x variants)
    x_syms = set(variables)
    params = sorted(expr.free_symbols - x_syms, key=lambda s: s.name)
    param_names = [str(p) for p in params]
    n_params = len(params)

    # Create a lambda function for curve_fit, accepting all x variants and params
    func = sp.lambdify((*x_syms, *params), expr, modules=['numpy'])

    try: 
        # Parse the loss function formula
        y_true_sym, y_pred_sym = sp.symbols('y_true y_pred')
        if 'y_true' not in loss_func or 'y_pred' not in loss_func:
            print("Please use y_true and y_pred in the loss function.")
            return
        loss_expr = sp.sympify(loss_func)
        loss_func_sympy = sp.lambdify((y_true_sym, y_pred_sym), loss_expr, modules=['numpy'])
    except TypeError as e:
        print("Please use y_true and y_pred in the loss function.", e)
        return 

    def loss_for_param_batch(param_array, x_data, y_data):
        """
        Vectorized loss computation for multiple parameter sets.
        param_array: shape (n_candidates, n_params)
        returns: losses shape (n_candidates,)
        """
        n_candidates, _ = param_array.shape
        losses = np.zeros(n_candidates)
        
        for i in range(n_candidates):
            try:
                y_pred = fit_func(x_data, *param_array[i], func=func, variables=variables)
                losses[i] = loss_func_eval(y_data, y_pred, loss_func_sympy)
            except:
                losses[i] = float('inf')
        
        return losses

    train_data, test_data = splitData(x_data, y_data)
    x_train, y_train = train_data
    x_test, y_test = test_data

    # Initialize population
    param_low, param_high = param_range
    population = np.random.uniform(param_low, param_high, size=(init_samples, n_params))
    evo_strength = evo_str_init
    best_params = None
    best_loss = 10e10
    eval_count = 0
    iteration = 0
    while eval_count < max_iter:
        iteration += 1
        # Generate offspring
        offspring = population[np.random.randint(0, pop_size, size=(offspring_per_gen,))] + \
                    np.random.normal(0, evo_strength, size=(offspring_per_gen, n_params))
        # Clip offspring to parameter range
        offspring = np.clip(offspring, param_low, param_high)

        # Evaluate losses for offspring
        losses = loss_for_param_batch(offspring, x_train, y_train)
        # Update population based on losses
        population = offspring[np.argsort(losses)[:pop_size]]
        
        # Update best parameters if we found a better solution
        current_best_loss = losses.min()
        if current_best_loss < best_loss:
            best_loss = current_best_loss
            best_params = offspring[np.argmin(losses)]
        
        eval_count += offspring.shape[0]
        # Decay evolutionary strategy
        evo_strength = max(evo_strength * 0.99, evo_str_min)
        if iteration % 10 == 0 or eval_count >= max_iter:
            print(f"Iteration {iteration}, Evaluations: {eval_count}, Best Loss: {best_loss:.4f}, Evo Strength: {evo_strength:.4f}")

    print("Optimal parameters found (train set):")
    for name, value in zip(param_names, best_params):
        print(f"{name} = {value}")
    print(f"Train loss: {best_loss:.4f}")

    # Evaluate on test data
    y_test_pred = fit_func(x_test, *best_params, func=func, variables=variables)
    test_loss = loss_func_eval(y_test, y_test_pred, loss_func_sympy)
    print(f"Test loss: {test_loss:.4f}")
    return best_params

def splitData(x_data, y_data, split_ratio=0.8):
    # Split the data into training and testing sets
    split_index = int(len(x_data) * split_ratio)
    x_train, x_test = x_data[:split_index], x_data[split_index:]
    y_train, y_test = y_data[:split_index], y_data[split_index:]
    return (x_train, y_train), (x_test, y_test)

# Example usage:
if __name__ == "__main__":
    ap = ap.ArgumentParser(description="Enhanced Fortuna Algorithm with evolutionary strategy for curve fitting")
    ap.add_argument('--formula', type=str, required=True, help='Formula in terms of x (e.g., a*x**2 + b*x + c)')
    ap.add_argument('--loss', type=str, required=True, help='Loss function formula (e.g., mean((y_true-y_pred)**2)) ' \
    'You can use mean, sum, abs, y_true and y_pred. Example for MAE: mean(abs(y_true-y_pred))')
    ap.add_argument('--data', type=str, required=True, help='Path to the CSV file containing the data with columns x and y')
    ap.add_argument('--max-iter', type=int, default=10000, help='Maximum number of function evaluations (default: 10000)')
    ap.add_argument('--init-samples', type=int, default=1000, help='Initial random population size (default: 1000)')
    ap.add_argument('--pop-size', type=int, default=50, help='Population size for evolution (default: 50)')
    ap.add_argument('--offspring-per-gen', type=int, default=500, help='Number of offspring per generation (default: 500)')
    ap.add_argument('--sigma-init', type=float, default=0.8, help='Initial mutation strength (default: 0.8)')
    ap.add_argument('--sigma-min', type=float, default=0.02, help='Minimum mutation strength (default: 0.02)')
    ap.add_argument('--param-range', nargs=2, type=float, default=[-10, 10], help='Parameter search range (default: -10 10)')
    args = ap.parse_args()

    # Load data
    data = args.data
    df = pd.read_csv(data)
    x_data = df['x'].values
    y_data = df['y'].values

    formula = args.formula
    loss_formula = args.loss
    
    # Run enhanced Fortuna algorithm
    fortuna_algorithm(x_data, y_data, formula, loss_formula, 
                     max_iter=args.max_iter,
                     init_samples=args.init_samples,
                     pop_size=args.pop_size,
                     offspring_per_gen=args.offspring_per_gen,
                     sigma_init=args.sigma_init,
                     sigma_min=args.sigma_min,
                     param_range=tuple(args.param_range))

