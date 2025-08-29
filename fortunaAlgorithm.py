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

def fortuna_algorithm(x_data, y_data, formula_str, loss_func, max_iter=10000):
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

    # Create a lambda function for curve_fit, accepting all x variants and params
    func = sp.lambdify((*x_syms, *params), expr, modules=['numpy'])

    # Parse the loss function formula
    y_true_sym, y_pred_sym = sp.symbols('y_true y_pred')    
    loss_expr = sp.sympify(loss_func)
    loss_func_sympy = sp.lambdify((y_true_sym, y_pred_sym), loss_expr, modules=['numpy'])

    def loss_func_eval(y_true_np, y_pred_np):
        losses = loss_func_sympy(y_true_np, y_pred_np) 
        return np.mean(losses) 

    train_data, test_data = splitData(x_data, y_data)
    x_train, y_train = train_data
    x_test, y_test = test_data

    def fit_func(x, *param_values):
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

    best_params = None
    best_loss = float('inf')
    np.random.seed(0)
    accuracy = 0.1
    improvement = float('inf')
    param_range = (-10, 10)
    iter_count = 0

    while improvement > accuracy and iter_count < max_iter:
        params_try = np.round(np.random.uniform(param_range[0], param_range[1], len(params)), 1)
        y_pred = fit_func(x_train, *params_try)
        loss = loss_func_eval(y_train, y_pred)
        if loss < best_loss:
            improvement = best_loss - loss
            best_loss = loss
            best_params = params_try
        iter_count += 1

    print("Optimal parameters found (train set):")
    for name, value in zip(param_names, best_params):
        print(f"{name} = {value}")
    print(f"Train loss: {best_loss:.4f}")

    # Evaluate on test data
    y_test_pred = fit_func(x_test, *best_params)
    test_loss = loss_func_eval(y_test, y_test_pred)
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
    ap = ap.ArgumentParser(description="Fortuna Algorithm for curve fitting")
    ap.add_argument('--formula', type=str, required=True, help='Formula in terms of x (e.g., a*x**2 + b*x + c)')
    ap.add_argument('--loss', type=str, required=True, help='Loss function formula (e.g., mean((y_true-y_pred)**2)) ' \
    'You can use mean, sum, abs, y_true and y_pred. Example for MAE: mean(abs(y_true-y_pred))')
    ap.add_argument('--data', type=str, required=True, help='Path to the CSV file containing the data with columns x and y')
    args = ap.parse_args()

    # Example data (quadratic)
    data = args.data
    df = pd.read_csv(data)
    x_data = df['x'].values
    y_data = df['y'].values

    formula = args.formula
    loss_formula = args.loss
    fortuna_algorithm(x_data, y_data, formula, loss_formula)

