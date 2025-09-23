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

def fortuna_algorithm(x_data, y_data, formula_str, loss_func, max_iter=10000, 
                     init_samples=1000, pop_size=50, offspring_per_gen=500,
                     sigma_init=0.8, sigma_min=0.02, param_range=(-10, 10)):
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
    - sigma_init: initial mutation strength
    - sigma_min: minimum mutation strength
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
                y_pred = fit_func(x_data, *param_array[i], func, variables)
                losses[i] = loss_func_eval(y_data, y_pred, loss_func_sympy)
            except:
                losses[i] = float('inf')  # Handle numerical errors
        
        return losses

    train_data, test_data = splitData(x_data, y_data)
    x_train, y_train = train_data
    x_test, y_test = test_data

    # Evolutionary strategy implementation
    if init_samples >= max_iter:
        # Fallback to simple random search if no room for evolution
        best_params = None
        best_loss = float('inf')
        
        for _ in range(max_iter):
            params_try = np.random.uniform(param_range[0], param_range[1], n_params)
            y_pred = fit_func(x_train, *params_try, func, variables)
            loss = loss_func_eval(y_train, y_pred, loss_func_sympy)
            if loss < best_loss:
                best_loss = loss
                best_params = params_try
    else:
        # 1) Initial random population
        param_array = np.random.uniform(param_range[0], param_range[1], size=(init_samples, n_params))
        losses = loss_for_param_batch(param_array, x_train, y_train)
        
        # Select top performers as parents
        idxs = np.argsort(losses)[:pop_size]
        parents = param_array[idxs].copy()
        parent_losses = losses[idxs].copy()
        
        used = init_samples
        remaining = max_iter - used
        n_gens = max(1, remaining // offspring_per_gen)
        
        sigma = sigma_init
        best_idx = np.argmin(parent_losses)
        best_params = parents[best_idx].copy()
        best_loss = parent_losses[best_idx]
        
        print(f"Starting evolution with {n_gens} generations, pop_size={pop_size}")
        
        # 2) Evolution loop
        for g in range(int(n_gens)):
            # Linearly anneal sigma
            progress = g / max(1, n_gens - 1)
            sigma = max(sigma_min, sigma_init * (1 - progress) + sigma_min * progress)
            
            # Create offspring by mutating parents
            parent_choices = np.random.choice(pop_size, size=offspring_per_gen)
            offspring = parents[parent_choices] + np.random.normal(scale=sigma, size=(offspring_per_gen, n_params))
            
            # Clip to parameter range
            np.clip(offspring, param_range[0], param_range[1], out=offspring)
            
            # Evaluate offspring
            offspring_losses = loss_for_param_batch(offspring, x_train, y_train)
            
            # Merge parents and offspring, select best
            merged_params = np.vstack([parents, offspring])
            merged_losses = np.concatenate([parent_losses, offspring_losses])
            idxs = np.argsort(merged_losses)[:pop_size]
            
            parents = merged_params[idxs].copy()
            parent_losses = merged_losses[idxs].copy()
            
            # Update best solution
            if parent_losses[0] < best_loss:
                best_loss = float(parent_losses[0])
                best_params = parents[0].copy()
            
            if g % 10 == 0 or g == n_gens - 1:
                print(f"Generation {g}: best loss = {best_loss:.6f}, sigma = {sigma:.4f}")

    print("Optimal parameters found (train set):")
    for name, value in zip(param_names, best_params):
        print(f"{name} = {value}")
    print(f"Train loss: {best_loss:.4f}")

    # Evaluate on test data
    y_test_pred = fit_func(x_test, *best_params, func, variables)
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

