"""
Author: Douwe Berkeij
Date: 14-10-2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from collections import defaultdict
import numpy as np
from datetime import datetime

def load_data(base_path):
    """Load all data files and return structured data"""
    
    # Load transactions data
    transactions = {}
    for i in range(5):  # 0-4 transaction files
        file_path = f"{base_path}/transactions/transactions_{i}.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
            transactions[i] = data
    
    # Load schedules data
    schedules = {}
    for i in range(5):  # 0-4 schedule files
        file_path = f"{base_path}/schedules/schedules_{i}.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
            schedules[i] = data
            
    # Load prices data
    prices = {}
    for i in range(5):  # 0-4 price files
        file_path = f"{base_path}/prices/prices_{i}.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
            prices[i] = data
            
    # Load amounts data
    amounts = {}
    for i in range(5):  # 0-4 amounts files
        file_path = f"{base_path}/amounts/amounts_{i}.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
            amounts[i] = data
    
    return transactions, schedules, prices, amounts

def load_worker_names(base_path):
    """Load worker names and create a mapping from worker_id to name"""
    worker_mapping = {}
    
    workers_file = f"{base_path}/workers/workers.jsonl"
    with open(workers_file, 'r') as f:
        for line in f:
            worker_data = json.loads(line.strip())
            worker_mapping[worker_data['worker_id']] = worker_data['name']
    
    return worker_mapping

def load_supplier_prices(base_path):
    """Load supplier prices (cost prices) for profit calculation"""
    supplier_file = f"{base_path}/supplier_prices.json"
    with open(supplier_file, 'r') as f:
        supplier_prices = json.load(f)
    return supplier_prices

def add_profit_data(df, prices, supplier_prices):
    """Add selling price, cost, and profit columns to the dataframe"""
    df_copy = df.copy()
    
    # Add selling price for each transaction
    def get_selling_price(row):
        week = row['week']
        product = row['product']
        if week in prices and product in prices[week]:
            return prices[week][product]
        return 0
    
    # Add cost price for each transaction
    def get_cost_price(row):
        product = row['product']
        return supplier_prices.get(product, 0)
    
    df_copy['selling_price'] = df_copy.apply(get_selling_price, axis=1)
    df_copy['cost_price'] = df_copy.apply(get_cost_price, axis=1)
    df_copy['profit_per_unit'] = df_copy['selling_price'] - df_copy['cost_price']
    df_copy['total_profit'] = df_copy['profit_per_unit'] * df_copy['amount']
    
    return df_copy

def add_worker_names(df, worker_mapping):
    """Add worker names to the dataframe"""
    df_copy = df.copy()
    df_copy['worker_name'] = df_copy['worker_id'].map(worker_mapping)
    # Fill any missing names with the ID (in case of missing data)
    df_copy['worker_name'] = df_copy['worker_name'].fillna(df_copy['worker_id'])
    return df_copy

def calculate_daily_inventory(transactions, amounts):
    """Calculate inventory levels for each day and detect sellouts"""
    print("Calculating daily inventory levels and detecting sellouts...")
    
    # Create inventory tracking structure
    inventory_data = []
    
    # Days of the week
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    num_weeks = len(amounts)
    for week in range(5):  # Use actual number of weeks
        # Get starting inventory for this week
        starting_inventory = amounts[week].copy()
        
        # Track daily inventory changes
        current_inventory = starting_inventory.copy()
        daily_sales = {day: {product: 0 for product in starting_inventory.keys()} for day in days}
        
        # Process all transactions for this week
        week_transactions = transactions[week]
        
        # Group transactions by day
        for day_num, day_transactions in week_transactions.items():
            day_name = days[int(day_num) - 1]  # Convert 1-7 to day names
            
            # Calculate total sales for each product on this day
            for transaction in day_transactions:
                if transaction.get('transaction_type') == 'customer_sale':
                    products = transaction.get('merch_types', [])
                    amounts_sold = transaction.get('merch_amounts', [])
                    
                    for product, amount in zip(products, amounts_sold):
                        if product in daily_sales[day_name]:
                            daily_sales[day_name][product] += amount
        
        # Calculate inventory levels for each day
        for day_idx, day_name in enumerate(days):
            day_inventory = starting_inventory.copy()
            
            # Subtract cumulative sales up to this day
            for prev_day_idx in range(day_idx + 1):
                prev_day = days[prev_day_idx]
                for product in day_inventory:
                    day_inventory[product] -= daily_sales[prev_day][product]
            
            # Record inventory data for this day
            for product, remaining in day_inventory.items():
                inventory_data.append({
                    'week': week,
                    'day_of_week': day_name,
                    'product': product,
                    'starting_inventory': starting_inventory[product],
                    'daily_sales': daily_sales[day_name][product],
                    'remaining_inventory': remaining,
                    'sold_out': remaining <= 0,
                    'day_index': day_idx
                })
    
    return pd.DataFrame(inventory_data)

def analyze_sellouts(inventory_df):
    """Analyze sellout patterns and create summary statistics"""
    print("Analyzing sellout patterns...")
    
    sellout_summary = []
    
    # Group by product and day to find sellout patterns
    for product in inventory_df['product'].unique():
        product_data = inventory_df[inventory_df['product'] == product]
        
        # Find sellout days
        sellout_days = product_data[product_data['sold_out']]['day_of_week'].unique()
        total_weeks = len(product_data['week'].unique())
        
        # Calculate average remaining inventory by day
        avg_inventory_by_day = product_data.groupby('day_of_week')['remaining_inventory'].mean()
        min_inventory_by_day = product_data.groupby('day_of_week')['remaining_inventory'].min()
        
        for day in inventory_df['day_of_week'].unique():
            day_data = product_data[product_data['day_of_week'] == day]
            sellouts_count = sum(day_data['sold_out'])
            
            sellout_summary.append({
                'product': product,
                'day_of_week': day,
                'sellouts_count': sellouts_count,
                'sellout_rate': sellouts_count / total_weeks * 100,
                'avg_remaining': avg_inventory_by_day.get(day, 0),
                'min_remaining': min_inventory_by_day.get(day, 0),
                'total_weeks': total_weeks
            })
    
    return pd.DataFrame(sellout_summary)

def create_output_directories(base_output_path):
    """Create directory structure for saving plots"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_path, f"sales_analysis_{timestamp}")
    
    directories = [
        output_dir,
        os.path.join(output_dir, "01_overall_analysis"),
        os.path.join(output_dir, "02_product_analysis"),
        os.path.join(output_dir, "02_product_analysis", "individual_products"),
        os.path.join(output_dir, "03_worker_analysis"), 
        os.path.join(output_dir, "03_worker_analysis", "individual_workers"),
        os.path.join(output_dir, "04_daily_analysis"),
        os.path.join(output_dir, "04_daily_analysis", "by_day"),
        os.path.join(output_dir, "05_combined_analysis"),
        os.path.join(output_dir, "05_combined_analysis", "product_worker_combinations"),
        os.path.join(output_dir, "05_combined_analysis", "product_day_combinations"),
        os.path.join(output_dir, "05_combined_analysis", "worker_day_combinations"),
        os.path.join(output_dir, "06_profit_analysis"),
        os.path.join(output_dir, "06_profit_analysis", "by_product"),
        os.path.join(output_dir, "06_profit_analysis", "by_week"),
        os.path.join(output_dir, "06_profit_analysis", "product_week_combinations"),
        os.path.join(output_dir, "07_inventory_analysis"),
        os.path.join(output_dir, "07_inventory_analysis", "sellout_details"),
        os.path.join(output_dir, "07_inventory_analysis", "by_product")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"Created output directory: {output_dir}")
    return output_dir

def safe_filename(name):
    """Convert a name to a safe filename"""
    # Replace problematic characters
    safe_name = str(name).replace(" ", "_").replace("/", "_").replace("\\", "_")
    safe_name = safe_name.replace(":", "_").replace("*", "_").replace("?", "_")
    safe_name = safe_name.replace('"', "_").replace("<", "_").replace(">", "_")
    safe_name = safe_name.replace("|", "_").replace(".", "_")
    return safe_name

def process_transactions_data(transactions):
    """Process transactions data into a structured DataFrame"""
    
    all_transactions = []
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for week, week_data in transactions.items():
        for day_num, day_transactions in week_data.items():
            # Days are numbered 1-7, so subtract 1 to get 0-6 index
            day_index = int(day_num) - 1
            day_name = day_names[day_index]
            
            for transaction in day_transactions:
                for product, amount in zip(transaction['merch_types'], transaction['merch_amounts']):
                    all_transactions.append({
                        'week': week,
                        'day_num': int(day_num),
                        'day_name': day_name,
                        'customer_id': transaction['customer_id'],
                        'product': product,
                        'amount': amount,
                        'worker_id': transaction['register_worker'],
                        'transaction_type': transaction['transaction_type']
                    })
    
    return pd.DataFrame(all_transactions)

def plot_sales_by_day_and_product(df, title="Sales by Day and Product", save_path=None):
    """Plot sales data by day and product"""
    
    # Group by day and product
    daily_product_sales = df.groupby(['day_name', 'product'])['amount'].sum().reset_index()
    
    # Create pivot table for heatmap
    pivot_data = daily_product_sales.pivot(index='day_name', columns='product', values='amount').fillna(0)
    
    # Reorder days properly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_data = pivot_data.reindex(day_order)
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.0f', cbar_kws={'label': 'Total Sales Amount'})
    plt.title(title)
    plt.xlabel('Product')
    plt.ylabel('Day of Week')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_sales_by_week_and_product(df, title="Sales by Week and Product"):
    """Plot sales data by week and product"""
    
    # Group by week and product
    weekly_product_sales = df.groupby(['week', 'product'])['amount'].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    sns.barplot(data=weekly_product_sales, x='week', y='amount', hue='product')
    plt.title(title)
    plt.xlabel('Week')
    plt.ylabel('Total Sales Amount')
    plt.legend(title='Product')
    plt.tight_layout()
    plt.show()

def plot_daily_sales_trends(df, title="Daily Sales Trends by Product"):
    """Plot daily sales trends for each product"""
    
    # Group by day and product
    daily_sales = df.groupby(['day_name', 'product'])['amount'].sum().reset_index()
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_sales['day_name'] = pd.Categorical(daily_sales['day_name'], categories=day_order, ordered=True)
    daily_sales = daily_sales.sort_values('day_name')
    
    plt.figure(figsize=(14, 8))
    
    # Create line plot
    sns.lineplot(data=daily_sales, x='day_name', y='amount', hue='product', marker='o', linewidth=2, markersize=8)
    plt.title(title)
    plt.xlabel('Day of Week')
    plt.ylabel('Total Sales Amount')
    plt.xticks(rotation=45)
    plt.legend(title='Product')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_product_distribution(df, title="Product Sales Distribution"):
    """Plot overall product distribution"""
    
    # Group by product
    product_sales = df.groupby('product')['amount'].sum().reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    sns.barplot(data=product_sales, x='product', y='amount', ax=ax1)
    ax1.set_title('Total Sales by Product')
    ax1.set_xlabel('Product')
    ax1.set_ylabel('Total Sales Amount')
    
    # Pie chart
    ax2.pie(product_sales['amount'], labels=product_sales['product'], autopct='%1.1f%%')
    ax2.set_title('Product Sales Share')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_weekly_comparison(df, title="Weekly Sales Comparison"):
    """Compare sales across different weeks"""
    
    # Group by week and day
    weekly_daily = df.groupby(['week', 'day_name'])['amount'].sum().reset_index()
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_daily['day_name'] = pd.Categorical(weekly_daily['day_name'], categories=day_order, ordered=True)
    weekly_daily = weekly_daily.sort_values(['week', 'day_name'])
    
    plt.figure(figsize=(14, 8))
    
    # Create line plot for each week
    sns.lineplot(data=weekly_daily, x='day_name', y='amount', hue='week', marker='o', linewidth=2, markersize=8)
    plt.title(title)
    plt.xlabel('Day of Week')
    plt.ylabel('Total Sales Amount')
    plt.xticks(rotation=45)
    plt.legend(title='Week')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_specific_product_trends(df, product_name):
    """Plot trends for a specific product across days and weeks"""
    
    product_data = df[df['product'] == product_name]
    
    if product_data.empty:
        print(f"No data found for product: {product_name}")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Daily trend
    daily = product_data.groupby('day_name')['amount'].sum().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily['day_name'] = pd.Categorical(daily['day_name'], categories=day_order, ordered=True)
    daily = daily.sort_values('day_name')
    
    sns.barplot(data=daily, x='day_name', y='amount', ax=ax1)
    ax1.set_title(f'{product_name.title()} - Daily Sales')
    ax1.tick_params(axis='x', rotation=45)
    
    # Weekly trend
    weekly = product_data.groupby('week')['amount'].sum().reset_index()
    sns.barplot(data=weekly, x='week', y='amount', ax=ax2)
    ax2.set_title(f'{product_name.title()} - Weekly Sales')
    
    # Day-Week heatmap
    day_week = product_data.groupby(['week', 'day_name'])['amount'].sum().reset_index()
    pivot = day_week.pivot(index='day_name', columns='week', values='amount').fillna(0)
    pivot = pivot.reindex(day_order)
    
    sns.heatmap(pivot, annot=True, fmt='.0f', ax=ax3, cmap='Blues')
    ax3.set_title(f'{product_name.title()} - Week vs Day Heatmap')
    
    # Transaction count vs amount
    transaction_stats = product_data.groupby(['week', 'day_name']).agg({
        'amount': ['sum', 'count']
    }).reset_index()
    transaction_stats.columns = ['week', 'day_name', 'total_amount', 'transaction_count']
    
    sns.scatterplot(data=transaction_stats, x='transaction_count', y='total_amount', 
                   hue='week', size='total_amount', ax=ax4)
    ax4.set_title(f'{product_name.title()} - Transactions vs Amount')
    
    plt.suptitle(f'Complete Analysis: {product_name.title()}', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_top_products_comparison(df, top_n=5):
    """Compare the top N products across different metrics"""
    
    # Get top products by total sales
    top_products = df.groupby('product')['amount'].sum().nlargest(top_n).index.tolist()
    top_data = df[df['product'].isin(top_products)]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Total sales comparison
    total_sales = top_data.groupby('product')['amount'].sum().sort_values(ascending=False)
    total_sales.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title(f'Top {top_n} Products - Total Sales')
    ax1.tick_params(axis='x', rotation=45)
    
    # Average transaction size
    avg_transaction = top_data.groupby('product')['amount'].mean().sort_values(ascending=False)
    avg_transaction.plot(kind='bar', ax=ax2, color='lightcoral')
    ax2.set_title(f'Top {top_n} Products - Average Transaction Size')
    ax2.tick_params(axis='x', rotation=45)
    
    # Daily distribution
    daily_dist = top_data.groupby(['product', 'day_name'])['amount'].sum().reset_index()
    sns.boxplot(data=daily_dist, x='product', y='amount', ax=ax3)
    ax3.set_title(f'Top {top_n} Products - Daily Sales Distribution')
    ax3.tick_params(axis='x', rotation=45)
    
    # Weekly trends
    weekly_trends = top_data.groupby(['product', 'week'])['amount'].sum().reset_index()
    sns.lineplot(data=weekly_trends, x='week', y='amount', hue='product', marker='o', ax=ax4)
    ax4.set_title(f'Top {top_n} Products - Weekly Trends')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_worker_performance(df):
    """Analyze worker performance and sales"""
    
    # Check if worker names are available
    if 'worker_name' in df.columns:
        worker_stats = df.groupby(['worker_id', 'worker_name']).agg({
            'amount': ['sum', 'count', 'mean'],
            'customer_id': 'nunique'
        }).round(2)
        
        worker_stats.columns = ['total_sales', 'transactions', 'avg_transaction', 'unique_customers']
        worker_stats = worker_stats.reset_index()
        name_col = 'worker_name'
    else:
        worker_stats = df.groupby('worker_id').agg({
            'amount': ['sum', 'count', 'mean'],
            'customer_id': 'nunique'
        }).round(2)
        
        worker_stats.columns = ['total_sales', 'transactions', 'avg_transaction', 'unique_customers']
        worker_stats = worker_stats.reset_index()
        name_col = 'worker_id'
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Total sales by worker
    top_workers = worker_stats.nlargest(10, 'total_sales')
    sns.barplot(data=top_workers, x=name_col, y='total_sales', ax=ax1)
    ax1.set_title('Top 10 Workers - Total Sales')
    ax1.tick_params(axis='x', rotation=45)
    
    # Transaction count by worker
    sns.barplot(data=top_workers, x=name_col, y='transactions', ax=ax2)
    ax2.set_title('Top 10 Workers - Transaction Count')
    ax2.tick_params(axis='x', rotation=45)
    
    # Average transaction size
    sns.scatterplot(data=worker_stats, x='transactions', y='avg_transaction', 
                   size='total_sales', alpha=0.7, ax=ax3)
    ax3.set_title('Worker Performance - Transactions vs Average Sale')
    
    # Add worker names/IDs to scatter plot (only top 15 to avoid clutter)
    for idx, row in worker_stats.head(15).iterrows():
        ax3.annotate(row[name_col], (row['transactions'], row['avg_transaction']), 
                    fontsize=8, alpha=0.7)
    
    # Unique customers served
    sns.scatterplot(data=worker_stats, x='unique_customers', y='total_sales', 
                   size='transactions', alpha=0.7, ax=ax4)
    ax4.set_title('Worker Performance - Customers vs Sales')
    
    # Add worker names/IDs to scatter plot (only top 15 to avoid clutter)
    for idx, row in worker_stats.head(15).iterrows():
        ax4.annotate(row[name_col], (row['unique_customers'], row['total_sales']), 
                    fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def plot_worker_performance_detailed(df):
    """Create a detailed worker performance analysis with names"""
    
    # Check if worker names are available
    if 'worker_name' in df.columns:
        worker_stats = df.groupby(['worker_id', 'worker_name']).agg({
            'amount': ['sum', 'count', 'mean', 'std'],
            'customer_id': 'nunique'
        }).round(2)
        
        worker_stats.columns = ['total_sales', 'transactions', 'avg_transaction', 'std_transaction', 'unique_customers']
        worker_stats = worker_stats.reset_index()
        name_col = 'worker_name'
    else:
        worker_stats = df.groupby('worker_id').agg({
            'amount': ['sum', 'count', 'mean', 'std'],
            'customer_id': 'nunique'
        }).round(2)
        
        worker_stats.columns = ['total_sales', 'transactions', 'avg_transaction', 'std_transaction', 'unique_customers']
        worker_stats = worker_stats.reset_index()
        name_col = 'worker_id'
    
    # Sort by total sales
    worker_stats = worker_stats.sort_values('total_sales', ascending=False)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Top 15 workers by sales
    top_15 = worker_stats.head(15)
    sns.barplot(data=top_15, x=name_col, y='total_sales', ax=ax1, hue=name_col, legend=False)
    ax1.set_title('Top 15 Workers by Total Sales')
    ax1.tick_params(axis='x', rotation=60)
    ax1.set_xlabel('Worker')
    
    # Transactions per worker
    sns.barplot(data=top_15, x=name_col, y='transactions', ax=ax2, hue=name_col, legend=False)
    ax2.set_title('Top 15 Workers by Transaction Count')
    ax2.tick_params(axis='x', rotation=60)
    ax2.set_xlabel('Worker')
    
    # Performance efficiency (Sales per transaction)
    worker_stats['efficiency'] = worker_stats['total_sales'] / worker_stats['transactions']
    top_efficient = worker_stats.nlargest(15, 'efficiency')
    sns.barplot(data=top_efficient, x=name_col, y='efficiency', ax=ax3, hue=name_col, legend=False)
    ax3.set_title('Top 15 Most Efficient Workers (Sales per Transaction)')
    ax3.tick_params(axis='x', rotation=60)
    ax3.set_xlabel('Worker')
    
    # Customer reach vs sales
    sns.scatterplot(data=worker_stats, x='unique_customers', y='total_sales', 
                   size='transactions', sizes=(50, 300), alpha=0.7, ax=ax4)
    ax4.set_title('Worker Performance: Customer Reach vs Total Sales')
    ax4.set_xlabel('Unique Customers Served')
    ax4.set_ylabel('Total Sales')
    
    plt.tight_layout()
    plt.show()
    
    # Print top performers summary
    print("\n=== TOP WORKER PERFORMANCE SUMMARY ===")
    print("Top 10 by Total Sales:")
    for i, (_, row) in enumerate(worker_stats.head(10).iterrows(), 1):
        worker_name = row[name_col] if 'worker_name' in df.columns else f"Worker {row['worker_id']}"
        print(f"{i:2d}. {worker_name:<20} - ${row['total_sales']:>8,.2f} ({row['transactions']:>3} transactions)")
    
    print(f"\nTop 5 Most Efficient (highest sales per transaction):")
    top_eff = worker_stats.nlargest(5, 'efficiency')
    for i, (_, row) in enumerate(top_eff.iterrows(), 1):
        worker_name = row[name_col] if 'worker_name' in df.columns else f"Worker {row['worker_id']}"
        print(f"{i}. {worker_name:<20} - ${row['efficiency']:>6.2f} per transaction")

def plot_sales_patterns(df):
    """Analyze sales patterns and trends"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sales by day of week (across all weeks)
    day_sales = df.groupby('day_name')['amount'].sum()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_sales = day_sales.reindex(day_order)
    
    day_sales.plot(kind='bar', ax=ax1, color='lightgreen')
    ax1.set_title('Total Sales by Day of Week')
    ax1.tick_params(axis='x', rotation=45)
    
    # Sales distribution by amount
    df['amount'].hist(bins=20, ax=ax2, alpha=0.7, color='orange')
    ax2.set_title('Distribution of Transaction Amounts')
    ax2.set_xlabel('Transaction Amount')
    ax2.set_ylabel('Frequency')
    
    # Product count vs total sales
    product_analysis = df.groupby('product').agg({
        'amount': ['sum', 'count']
    }).round(2)
    product_analysis.columns = ['total_sales', 'transaction_count']
    product_analysis = product_analysis.reset_index()
    
    sns.scatterplot(data=product_analysis, x='transaction_count', y='total_sales', 
                   s=100, alpha=0.7, ax=ax3)
    ax3.set_title('Product Analysis - Count vs Sales')
    
    # Add product labels to scatter plot
    for idx, row in product_analysis.iterrows():
        ax3.annotate(row['product'], (row['transaction_count'], row['total_sales']), 
                    fontsize=8, alpha=0.7)
    
    # Weekly sales progression
    weekly_sales = df.groupby('week')['amount'].sum()
    weekly_sales.plot(kind='line', marker='o', ax=ax4, color='purple', linewidth=3)
    ax4.set_title('Weekly Sales Progression')
    ax4.set_xlabel('Week')
    ax4.set_ylabel('Total Sales')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_inventory_levels(inventory_df, save_path=None):
    """Create plots showing inventory levels and sellouts by day of week"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Inventory Analysis - Stock Levels and Sellouts', fontsize=16, fontweight='bold')
    
    # 1. Average remaining inventory by day and product (heatmap)
    pivot_avg = inventory_df.pivot_table(values='remaining_inventory', 
                                         index='product', 
                                         columns='day_of_week', 
                                         aggfunc='mean')
    
    # Reorder columns by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_avg = pivot_avg.reindex(columns=day_order)
    
    sns.heatmap(pivot_avg, annot=True, fmt='.0f', cmap='RdYlGn', 
                ax=axes[0,0], cbar_kws={'label': 'Avg Remaining Inventory'})
    axes[0,0].set_title('Average Remaining Inventory by Day & Product')
    axes[0,0].set_xlabel('Day of Week')
    axes[0,0].set_ylabel('Product')
    
    # 2. Sellout frequency by day of week
    sellout_by_day = inventory_df[inventory_df['sold_out']].groupby('day_of_week').size()
    day_counts = inventory_df.groupby('day_of_week').size()
    sellout_rates = (sellout_by_day / day_counts * 100).reindex(day_order)
    
    axes[0,1].bar(sellout_rates.index, sellout_rates.values, color='red', alpha=0.7)
    axes[0,1].set_title('Sellout Rate by Day of Week')
    axes[0,1].set_xlabel('Day of Week')
    axes[0,1].set_ylabel('Sellout Rate (%)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Products with most sellouts
    sellout_by_product = inventory_df[inventory_df['sold_out']].groupby('product').size().sort_values(ascending=False)
    top_sellouts = sellout_by_product.head(10)
    
    axes[1,0].barh(range(len(top_sellouts)), top_sellouts.values, color='orange', alpha=0.7)
    axes[1,0].set_yticks(range(len(top_sellouts)))
    axes[1,0].set_yticklabels(top_sellouts.index)
    axes[1,0].set_title('Products with Most Sellouts')
    axes[1,0].set_xlabel('Number of Sellouts')
    
    # 4. Weekly sellout comparison
    weekly_sellouts = inventory_df[inventory_df['sold_out']].groupby('week').size()
    axes[1,1].bar(weekly_sellouts.index, weekly_sellouts.values, color='purple', alpha=0.7)
    axes[1,1].set_title('Total Sellouts by Week')
    axes[1,1].set_xlabel('Week')
    axes[1,1].set_ylabel('Number of Sellouts')
    axes[1,1].set_xticks(weekly_sellouts.index)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Inventory levels plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_sellout_details(inventory_df, save_path=None):
    """Create detailed sellout analysis plots"""
    # Filter to only products that have had sellouts
    sellout_products = inventory_df[inventory_df['sold_out']]['product'].unique()
    
    if len(sellout_products) == 0:
        print("No sellouts detected in the data.")
        return
    
    # Create subplots for each product with sellouts
    n_products = len(sellout_products)
    cols = 3
    rows = (n_products + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    fig.suptitle('Detailed Sellout Analysis by Product', fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = axes.reshape(1, -1) if n_products > 1 else [axes]
    
    for i, product in enumerate(sellout_products):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col] if n_products > 1 else axes
        
        product_data = inventory_df[inventory_df['product'] == product]
        
        # Plot remaining inventory over days for each week
        for week in product_data['week'].unique():
            week_data = product_data[product_data['week'] == week]
            week_data_sorted = week_data.sort_values('day_index')
            
            ax.plot(week_data_sorted['day_index'], week_data_sorted['remaining_inventory'], 
                   marker='o', label=f'Week {week}', alpha=0.7)
        
        # Highlight sellout threshold
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Sellout Threshold')
        
        ax.set_title(f'{product}')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Remaining Inventory')
        ax.set_xticks(range(7))
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=45)
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_products, rows * cols):
        row = i // cols
        col = i % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        elif n_products > 1:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Sellout details plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def save_weekly_sellout_plots(inventory_df, output_dir):
    """Generate and save a sellout plot for every week, showing all products in one plot"""
    sellout_dir = os.path.join(output_dir, "07_inventory_analysis", "weekly_sellouts")
    os.makedirs(sellout_dir, exist_ok=True)
    weeks = sorted(inventory_df['week'].unique())
    products = sorted(inventory_df['product'].unique())
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    cmap = plt.get_cmap('tab20')
    color_list = [cmap(i % cmap.N) for i in range(len(products))]
    for week in weeks:
        week_data = inventory_df[inventory_df['week'] == week]
        plt.figure(figsize=(14, 8))
        for idx, product in enumerate(products):
            product_data = week_data[week_data['product'] == product]
            # Get remaining inventory for each day
            inventory_by_day = product_data.set_index('day_of_week').reindex(days)['remaining_inventory'].fillna(0)
            plt.plot(days, inventory_by_day, marker='o', label=product, color=color_list[idx])
        plt.title(f'Remaining Inventory by Day - Week {week}')
        plt.xlabel('Day of Week')
        plt.ylabel('Remaining Inventory')
        plt.xticks(rotation=45)
        plt.legend(title='Product', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
        plt.tight_layout()
        save_path = os.path.join(sellout_dir, f'week_{week}_inventory.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved inventory decrease plot for week {week}: {save_path}")

def create_inventory_report(inventory_df, sellout_summary):
    """Create a text report of inventory analysis"""
    print("\n" + "="*60)
    print("📦 INVENTORY & SELLOUT ANALYSIS REPORT")
    print("="*60)
    
    # Overall statistics
    total_product_days = len(inventory_df)
    total_sellouts = len(inventory_df[inventory_df['sold_out']])
    sellout_rate = total_sellouts / total_product_days * 100
    
    print(f"Overall Statistics:")
    print(f"• Total product-day combinations analyzed: {total_product_days}")
    print(f"• Total sellouts occurred: {total_sellouts}")
    print(f"• Overall sellout rate: {sellout_rate:.1f}%")
    print()
    
    # Sellouts by day of week
    print("Sellouts by Day of Week:")
    sellout_by_day = inventory_df[inventory_df['sold_out']].groupby('day_of_week').size()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for day in day_order:
        count = sellout_by_day.get(day, 0)
        print(f"• {day}: {count} sellouts")
    print()
    
    # Most problematic products
    print("Products with Most Sellouts:")
    sellout_by_product = inventory_df[inventory_df['sold_out']].groupby('product').size().sort_values(ascending=False)
    for product, count in sellout_by_product.head(5).items():
        print(f"• {product}: {count} sellouts")
    print()
    
    # Products that never sold out
    all_products = set(inventory_df['product'].unique())
    sellout_products = set(inventory_df[inventory_df['sold_out']]['product'].unique())
    no_sellout_products = all_products - sellout_products
    
    if no_sellout_products:
        print("Products that Never Sold Out:")
        for product in sorted(no_sellout_products):
            print(f"• {product}")
        print()
    
    # Average inventory levels
    print("Average Remaining Inventory by Product:")
    avg_inventory = inventory_df.groupby('product')['remaining_inventory'].mean().sort_values(ascending=False)
    for product, avg in avg_inventory.items():
        print(f"• {product}: {avg:.0f} units")

def generate_summary_statistics(df):
    """Generate and print summary statistics"""
    
    print("=== SALES SUMMARY STATISTICS ===\n")
    
    # Overall statistics
    print("Overall Statistics:")
    print(f"Total transactions: {len(df)}")
    print(f"Total sales amount: {df['amount'].sum()}")
    print(f"Average sale amount: {df['amount'].mean():.2f}")
    print(f"Unique customers: {df['customer_id'].nunique()}")
    print(f"Unique workers: {df['worker_id'].nunique()}")
    print()
    
    # Product statistics
    print("Product Statistics:")
    product_stats = df.groupby('product')['amount'].agg(['count', 'sum', 'mean']).round(2)
    print(product_stats)
    print()
    
    # Daily statistics
    print("Daily Statistics:")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_stats = df.groupby('day_name')['amount'].agg(['count', 'sum', 'mean']).reindex(day_order).round(2)
    print(daily_stats)
    print()
    
    # Weekly statistics
    print("Weekly Statistics:")
    weekly_stats = df.groupby('week')['amount'].agg(['count', 'sum', 'mean']).round(2)
    print(weekly_stats)

def create_comprehensive_analysis(df, save_plots=False):
    """Create a comprehensive analysis including advanced plotting functions"""
    
    print("=== CREATING COMPREHENSIVE ANALYSIS ===")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Get available products
    products = df['product'].unique()
    print(f"Available products: {', '.join(products)}")
    
    # Get worker statistics
    if 'worker_name' in df.columns:
        unique_workers = df['worker_name'].nunique()
        print(f"Workers in dataset: {unique_workers} unique workers (with names)")
    else:
        unique_workers = df['worker_id'].nunique()
        print(f"Workers in dataset: {unique_workers} unique workers (IDs only)")
    
    # 1. Original analysis plots
    print("\n1. Basic Sales Analysis...")
    plot_sales_by_day_and_product(df, "Sales Heatmap: Day vs Product")
    plot_sales_by_week_and_product(df, "Sales by Week and Product")
    plot_daily_sales_trends(df, "Daily Sales Trends by Product")
    plot_product_distribution(df, "Product Sales Distribution")
    plot_weekly_comparison(df, "Weekly Sales Comparison")
    
    # 2. Advanced product analysis
    print("2. Top Products Comparison...")
    plot_top_products_comparison(df, top_n=6)
    
    # 3. Worker performance analysis
    print("3. Worker Performance Analysis...")
    plot_worker_performance(df)
    
    print("3b. Detailed Worker Performance Analysis...")
    plot_worker_performance_detailed(df)
    
    # 4. Sales patterns
    print("4. Sales Patterns Analysis...")
    plot_sales_patterns(df)
    
    # 5. Specific product analysis (top 3)
    top_3_products = df.groupby('product')['amount'].sum().nlargest(3).index.tolist()
    for product in top_3_products:
        print(f"5. Specific analysis for {product}...")
        plot_specific_product_trends(df, product)
    
    # 6. Summary statistics
    print("6. Summary Statistics...")
    generate_summary_statistics(df)
    
    print("\nComprehensive analysis complete!")

def save_individual_product_analysis(df, output_dir):
    """Generate and save analysis for each individual product"""
    
    print("Generating individual product analyses...")
    products = df['product'].unique()
    product_dir = os.path.join(output_dir, "02_product_analysis", "individual_products")
    
    for product in products:
        print(f"  Analyzing product: {product}")
        product_data = df[df['product'] == product]
        safe_product_name = safe_filename(product)
        
        # 1. Daily sales for this product
        plt.figure(figsize=(10, 6))
        daily_sales = product_data.groupby('day_name')['amount'].sum().reset_index()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_sales['day_name'] = pd.Categorical(daily_sales['day_name'], categories=day_order, ordered=True)
        daily_sales = daily_sales.sort_values('day_name')
        
        sns.barplot(data=daily_sales, x='day_name', y='amount')
        plt.title(f'{product.title()} - Daily Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(product_dir, f"{safe_product_name}_daily_sales.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Weekly sales for this product
        plt.figure(figsize=(8, 6))
        weekly_sales = product_data.groupby('week')['amount'].sum().reset_index()
        sns.barplot(data=weekly_sales, x='week', y='amount')
        plt.title(f'{product.title()} - Weekly Sales')
        plt.tight_layout()
        plt.savefig(os.path.join(product_dir, f"{safe_product_name}_weekly_sales.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Worker performance for this product
        if 'worker_name' in df.columns:
            plt.figure(figsize=(12, 8))
            worker_product_sales = product_data.groupby('worker_name')['amount'].sum().sort_values(ascending=False).head(10)
            worker_product_sales.plot(kind='bar')
            plt.title(f'{product.title()} - Top 10 Workers')
            plt.xticks(rotation=45)
            plt.ylabel('Total Sales')
            plt.tight_layout()
            plt.savefig(os.path.join(product_dir, f"{safe_product_name}_top_workers.png"), dpi=300, bbox_inches='tight')
            plt.close()

def save_individual_worker_analysis(df, output_dir):
    """Generate and save analysis for each individual worker"""
    
    if 'worker_name' not in df.columns:
        print("Worker names not available. Skipping individual worker analysis.")
        return
    
    print("Generating individual worker analyses...")
    workers = df['worker_name'].unique()
    worker_dir = os.path.join(output_dir, "03_worker_analysis", "individual_workers")
    
    for worker in workers:
        print(f"  Analyzing worker: {worker}")
        worker_data = df[df['worker_name'] == worker]
        safe_worker_name = safe_filename(worker)
        
        # 1. Daily performance for this worker
        plt.figure(figsize=(10, 6))
        daily_worker = worker_data.groupby('day_name')['amount'].sum().reset_index()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_worker['day_name'] = pd.Categorical(daily_worker['day_name'], categories=day_order, ordered=True)
        daily_worker = daily_worker.sort_values('day_name')
        
        sns.barplot(data=daily_worker, x='day_name', y='amount')
        plt.title(f'{worker} - Daily Sales Performance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(worker_dir, f"{safe_worker_name}_daily_performance.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Product sales by this worker
        plt.figure(figsize=(12, 6))
        product_worker = worker_data.groupby('product')['amount'].sum().sort_values(ascending=False)
        product_worker.plot(kind='bar')
        plt.title(f'{worker} - Product Sales')
        plt.xticks(rotation=45)
        plt.ylabel('Total Sales')
        plt.tight_layout()
        plt.savefig(os.path.join(worker_dir, f"{safe_worker_name}_product_sales.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Weekly performance for this worker
        plt.figure(figsize=(8, 6))
        weekly_worker = worker_data.groupby('week')['amount'].sum().reset_index()
        sns.lineplot(data=weekly_worker, x='week', y='amount', marker='o')
        plt.title(f'{worker} - Weekly Performance')
        plt.tight_layout()
        plt.savefig(os.path.join(worker_dir, f"{safe_worker_name}_weekly_performance.png"), dpi=300, bbox_inches='tight')
        plt.close()

def save_daily_analysis(df, output_dir):
    """Generate and save analysis for each day of the week"""
    
    print("Generating daily analyses...")
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_dir = os.path.join(output_dir, "04_daily_analysis", "by_day")
    
    for day in days:
        day_data = df[df['day_name'] == day]
        if day_data.empty:
            continue
            
        print(f"  Analyzing {day}")
        safe_day_name = safe_filename(day)
        
        # 1. Product sales on this day
        plt.figure(figsize=(12, 6))
        product_day = day_data.groupby('product')['amount'].sum().sort_values(ascending=False)
        product_day.plot(kind='bar')
        plt.title(f'{day} - Product Sales')
        plt.xticks(rotation=45)
        plt.ylabel('Total Sales')
        plt.tight_layout()
        plt.savefig(os.path.join(daily_dir, f"{safe_day_name}_product_sales.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Worker performance on this day
        if 'worker_name' in df.columns:
            plt.figure(figsize=(12, 8))
            worker_day = day_data.groupby('worker_name')['amount'].sum().sort_values(ascending=False).head(15)
            worker_day.plot(kind='bar')
            plt.title(f'{day} - Top 15 Worker Performance')
            plt.xticks(rotation=45)
            plt.ylabel('Total Sales')
            plt.tight_layout()
            plt.savefig(os.path.join(daily_dir, f"{safe_day_name}_worker_performance.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Weekly comparison for this day
        plt.figure(figsize=(8, 6))
        weekly_day = day_data.groupby('week')['amount'].sum().reset_index()
        sns.barplot(data=weekly_day, x='week', y='amount')
        plt.title(f'{day} - Weekly Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(daily_dir, f"{safe_day_name}_weekly_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

def save_combined_analysis(df, output_dir):
    """Generate and save combined analyses (product-worker, product-day, worker-day combinations)"""
    
    print("Generating combined analyses...")
    combined_dir = os.path.join(output_dir, "05_combined_analysis")
    
    # 1. Product-Worker combinations (top combinations)
    print("  Generating product-worker combinations...")
    if 'worker_name' in df.columns:
        pw_dir = os.path.join(combined_dir, "product_worker_combinations")
        
        # Get top product-worker combinations
        product_worker_sales = df.groupby(['product', 'worker_name'])['amount'].sum().reset_index()
        product_worker_sales = product_worker_sales.sort_values('amount', ascending=False).head(50)
        
        # Create pivot table for heatmap
        plt.figure(figsize=(20, 12))
        pivot_pw = product_worker_sales.pivot(index='worker_name', columns='product', values='amount').fillna(0)
        sns.heatmap(pivot_pw, annot=True, fmt='.0f', cmap='Blues')
        plt.title('Product-Worker Sales Heatmap (Top 50 Combinations)')
        plt.tight_layout()
        plt.savefig(os.path.join(pw_dir, "product_worker_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Product-Day combinations
    print("  Generating product-day combinations...")
    pd_dir = os.path.join(combined_dir, "product_day_combinations")
    
    products = df['product'].unique()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for product in products[:5]:  # Top 5 products to avoid too many files
        product_data = df[df['product'] == product]
        safe_product_name = safe_filename(product)
        
        plt.figure(figsize=(12, 8))
        daily_product = product_data.groupby(['day_name', 'week'])['amount'].sum().reset_index()
        daily_product['day_name'] = pd.Categorical(daily_product['day_name'], categories=days, ordered=True)
        
        sns.lineplot(data=daily_product, x='day_name', y='amount', hue='week', marker='o')
        plt.title(f'{product.title()} - Daily Sales by Week')
        plt.xticks(rotation=45)
        plt.legend(title='Week')
        plt.tight_layout()
        plt.savefig(os.path.join(pd_dir, f"{safe_product_name}_daily_by_week.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Worker-Day combinations
    print("  Generating worker-day combinations...")
    if 'worker_name' in df.columns:
        wd_dir = os.path.join(combined_dir, "worker_day_combinations")
        
        # Overall worker-day heatmap
        worker_day_sales = df.groupby(['worker_name', 'day_name'])['amount'].sum().reset_index()
        pivot_wd = worker_day_sales.pivot(index='worker_name', columns='day_name', values='amount').fillna(0)
        pivot_wd = pivot_wd.reindex(columns=days)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_wd, annot=True, fmt='.0f', cmap='Greens')
        plt.title('Worker-Day Sales Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(wd_dir, "worker_day_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()

def save_profit_analysis(df, output_dir):
    """Generate and save comprehensive profit analysis"""
    
    if 'total_profit' not in df.columns:
        print("❌ Profit data not available. Make sure profit calculations are added to dataframe.")
        return
    
    print("Generating profit analyses...")
    profit_dir = os.path.join(output_dir, "06_profit_analysis")
    by_product_dir = os.path.join(profit_dir, "by_product")
    by_week_dir = os.path.join(profit_dir, "by_week")
    combinations_dir = os.path.join(profit_dir, "product_week_combinations")
    
    # 1. Overall profit overview
    plt.figure(figsize=(15, 10))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Total profit by product
    product_profit = df.groupby('product')['total_profit'].sum().sort_values(ascending=False)
    product_profit.plot(kind='bar', ax=ax1, color='green', alpha=0.7)
    ax1.set_title('Total Profit by Product')
    ax1.set_ylabel('Total Profit ($)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Total profit by week
    weekly_profit = df.groupby('week')['total_profit'].sum()
    weekly_profit.plot(kind='bar', ax=ax2, color='blue', alpha=0.7)
    ax2.set_title('Total Profit by Week')
    ax2.set_ylabel('Total Profit ($)')
    
    # Profit margin by product
    product_stats = df.groupby('product').agg({
        'selling_price': 'mean',
        'cost_price': 'mean',
        'profit_per_unit': 'mean'
    }).round(2)
    product_stats['profit_margin'] = (product_stats['profit_per_unit'] / product_stats['selling_price'] * 100).round(1)
    product_stats = product_stats.sort_values('profit_margin', ascending=False)
    
    product_stats['profit_margin'].plot(kind='bar', ax=ax3, color='orange', alpha=0.7)
    ax3.set_title('Profit Margin by Product (%)')
    ax3.set_ylabel('Profit Margin (%)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Profit per unit vs quantity sold
    product_summary = df.groupby('product').agg({
        'profit_per_unit': 'mean',
        'amount': 'sum',
        'total_profit': 'sum'
    }).round(2)
    
    ax4.scatter(product_summary['amount'], product_summary['profit_per_unit'], 
               s=product_summary['total_profit']/10, alpha=0.6, c='purple')
    ax4.set_xlabel('Total Quantity Sold')
    ax4.set_ylabel('Profit per Unit ($)')
    ax4.set_title('Profit per Unit vs Quantity (bubble size = total profit)')
    
    # Add product labels
    for idx, row in product_summary.iterrows():
        ax4.annotate(idx, (row['amount'], row['profit_per_unit']), 
                    fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(profit_dir, "overall_profit_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Product-Week profit heatmap
    plt.figure(figsize=(12, 8))
    product_week_profit = df.groupby(['product', 'week'])['total_profit'].sum().reset_index()
    pivot_profit = product_week_profit.pivot(index='product', columns='week', values='total_profit').fillna(0)
    
    sns.heatmap(pivot_profit, annot=True, fmt='.0f', cmap='RdYlGn', center=0)
    plt.title('Profit by Product and Week')
    plt.ylabel('Product')
    plt.xlabel('Week')
    plt.tight_layout()
    plt.savefig(os.path.join(profit_dir, "product_week_profit_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Individual product profit analysis
    print("  Generating individual product profit analyses...")
    products = df['product'].unique()
    
    for product in products:
        product_data = df[df['product'] == product]
        safe_product_name = safe_filename(product)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Weekly profit for this product
        weekly_product_profit = product_data.groupby('week')['total_profit'].sum()
        weekly_product_profit.plot(kind='bar', ax=ax1, color='green')
        ax1.set_title(f'{product.title()} - Weekly Profit')
        ax1.set_ylabel('Total Profit ($)')
        
        # Daily profit for this product
        daily_product_profit = product_data.groupby('day_name')['total_profit'].sum()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_product_profit = daily_product_profit.reindex(day_order, fill_value=0)
        daily_product_profit.plot(kind='bar', ax=ax2, color='blue')
        ax2.set_title(f'{product.title()} - Daily Profit')
        ax2.set_ylabel('Total Profit ($)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Profit vs sales volume
        daily_stats = product_data.groupby('day_name').agg({
            'total_profit': 'sum',
            'amount': 'sum'
        })
        ax3.scatter(daily_stats['amount'], daily_stats['total_profit'])
        ax3.set_xlabel('Quantity Sold')
        ax3.set_ylabel('Total Profit ($)')
        ax3.set_title(f'{product.title()} - Profit vs Volume')
        
        # Add day labels
        for day, row in daily_stats.iterrows():
            ax3.annotate(day[:3], (row['amount'], row['total_profit']), fontsize=8)
        
        # Price and cost comparison
        price_cost_data = product_data.groupby('week').agg({
            'selling_price': 'mean',
            'cost_price': 'mean',
            'profit_per_unit': 'mean'
        })
        
        x = range(len(price_cost_data))
        width = 0.35
        ax4.bar([i - width/2 for i in x], price_cost_data['selling_price'], width, 
               label='Selling Price', alpha=0.8, color='green')
        ax4.bar([i + width/2 for i in x], price_cost_data['cost_price'], width, 
               label='Cost Price', alpha=0.8, color='red')
        ax4.set_xlabel('Week')
        ax4.set_ylabel('Price ($)')
        ax4.set_title(f'{product.title()} - Price vs Cost by Week')
        ax4.set_xticks(x)
        ax4.set_xticklabels(price_cost_data.index)
        ax4.legend()
        
        plt.suptitle(f'Profit Analysis: {product.title()}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(by_product_dir, f"{safe_product_name}_profit_analysis.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Weekly profit analysis
    print("  Generating weekly profit analyses...")
    weeks = sorted(df['week'].unique())
    
    for week in weeks:
        week_data = df[df['week'] == week]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Product profit ranking for this week
        product_week_profit = week_data.groupby('product')['total_profit'].sum().sort_values(ascending=False)
        product_week_profit.plot(kind='bar', ax=ax1, color='green')
        ax1.set_title(f'Week {week} - Profit by Product')
        ax1.set_ylabel('Total Profit ($)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Daily profit for this week
        daily_week_profit = week_data.groupby('day_name')['total_profit'].sum()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_week_profit = daily_week_profit.reindex(day_order, fill_value=0)
        daily_week_profit.plot(kind='bar', ax=ax2, color='blue')
        ax2.set_title(f'Week {week} - Daily Profit')
        ax2.set_ylabel('Total Profit ($)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Worker profit contribution for this week
        if 'worker_name' in df.columns:
            worker_week_profit = week_data.groupby('worker_name')['total_profit'].sum().sort_values(ascending=False).head(10)
            worker_week_profit.plot(kind='bar', ax=ax3, color='orange')
            ax3.set_title(f'Week {week} - Top 10 Workers by Profit')
            ax3.set_ylabel('Total Profit ($)')
            ax3.tick_params(axis='x', rotation=45)
        
        # Profit margin distribution for this week
        week_data['profit_margin'] = (week_data['profit_per_unit'] / week_data['selling_price'] * 100)
        week_data['profit_margin'].hist(bins=15, ax=ax4, alpha=0.7, color='purple')
        ax4.set_title(f'Week {week} - Profit Margin Distribution')
        ax4.set_xlabel('Profit Margin (%)')
        ax4.set_ylabel('Frequency')
        
        plt.suptitle(f'Week {week} Profit Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(by_week_dir, f"week_{week}_profit_analysis.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Product-Week combination analysis
    print("  Generating product-week combination analysis...")
    
    # Create detailed profit table for each product across weeks
    for product in products:  # All products instead of just top 5
        product_data = df[df['product'] == product]
        safe_product_name = safe_filename(product)
        
        print(f"    Creating product-week analysis for {product}...")
        
        # Weekly breakdown for this product
        weekly_breakdown = product_data.groupby('week').agg({
            'amount': 'sum',
            'total_profit': 'sum',
            'selling_price': 'mean',
            'cost_price': 'mean',
            'profit_per_unit': 'mean'
        }).round(2)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Profit and quantity trend
        ax1_twin = ax1.twinx()
        ax1.bar(weekly_breakdown.index, weekly_breakdown['total_profit'], alpha=0.7, color='green', label='Total Profit')
        ax1_twin.plot(weekly_breakdown.index, weekly_breakdown['amount'], color='blue', marker='o', linewidth=2, label='Quantity Sold')
        
        ax1.set_xlabel('Week')
        ax1.set_ylabel('Total Profit ($)', color='green')
        ax1_twin.set_ylabel('Quantity Sold', color='blue')
        ax1.set_title(f'{product.title()} - Weekly Profit & Quantity Trend')
        
        # Price evolution
        ax2.plot(weekly_breakdown.index, weekly_breakdown['selling_price'], marker='o', label='Selling Price', linewidth=2)
        ax2.plot(weekly_breakdown.index, weekly_breakdown['cost_price'], marker='s', label='Cost Price', linewidth=2)
        ax2.fill_between(weekly_breakdown.index, weekly_breakdown['cost_price'], weekly_breakdown['selling_price'], 
                        alpha=0.3, label='Profit Margin')
        ax2.set_xlabel('Week')
        ax2.set_ylabel('Price ($)')
        ax2.set_title(f'{product.title()} - Price Evolution')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(combinations_dir, f"{safe_product_name}_weekly_profit_trend.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✅ Profit analysis complete!")

def save_overall_analysis(df, output_dir):
    """Generate and save overall analysis plots"""
    
    print("Generating overall analysis...")
    overall_dir = os.path.join(output_dir, "01_overall_analysis")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Sales by day and product heatmap
    plot_sales_by_day_and_product(df, "Sales Heatmap: Day vs Product", 
                                  os.path.join(overall_dir, "sales_day_product_heatmap.png"))
    
    # 2. Product distribution
    plt.figure(figsize=(15, 6))
    product_sales = df.groupby('product')['amount'].sum().reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=product_sales, x='product', y='amount', ax=ax1)
    ax1.set_title('Total Sales by Product')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.pie(product_sales['amount'], labels=product_sales['product'], autopct='%1.1f%%')
    ax2.set_title('Product Sales Share')
    
    plt.tight_layout()
    plt.savefig(os.path.join(overall_dir, "product_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Daily sales trends
    plt.figure(figsize=(14, 8))
    daily_sales = df.groupby(['day_name', 'product'])['amount'].sum().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_sales['day_name'] = pd.Categorical(daily_sales['day_name'], categories=day_order, ordered=True)
    daily_sales = daily_sales.sort_values('day_name')
    
    sns.lineplot(data=daily_sales, x='day_name', y='amount', hue='product', marker='o', linewidth=2, markersize=8)
    plt.title('Daily Sales Trends by Product')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(overall_dir, "daily_sales_trends.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Worker performance overview
    if 'worker_name' in df.columns:
        worker_stats = df.groupby('worker_name').agg({
            'amount': ['sum', 'count', 'mean'],
            'customer_id': 'nunique'
        }).round(2)
        
        worker_stats.columns = ['total_sales', 'transactions', 'avg_transaction', 'unique_customers']
        worker_stats = worker_stats.reset_index().sort_values('total_sales', ascending=False)
        
        plt.figure(figsize=(15, 8))
        top_workers = worker_stats.head(15)
        sns.barplot(data=top_workers, x='worker_name', y='total_sales')
        plt.title('Top 15 Workers - Total Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(overall_dir, "top_workers_overview.png"), dpi=300, bbox_inches='tight')
        plt.close()

def save_inventory_analysis(inventory_df, sellout_summary, output_dir):
    """Save inventory and sellout analysis plots and reports"""
    print("Creating inventory analysis...")
    
    # Create inventory analysis directories
    inventory_dir = os.path.join(output_dir, "07_inventory_analysis")
    sellout_dir = os.path.join(inventory_dir, "sellout_details")
    product_dir = os.path.join(inventory_dir, "by_product")
    
    # 1. Overall inventory levels plot
    save_path = os.path.join(inventory_dir, "overall_inventory_analysis.png")
    plot_inventory_levels(inventory_df, save_path)
    
    # 2. Detailed sellout analysis
    save_path = os.path.join(sellout_dir, "sellout_details_by_product.png")
    plot_sellout_details(inventory_df, save_path)
    
    # 3. Create individual product inventory plots
    for product in inventory_df['product'].unique():
        product_data = inventory_df[inventory_df['product'] == product]
        
        plt.figure(figsize=(12, 8))
        
        # Plot inventory levels by day and week
        for week in product_data['week'].unique():
            week_data = product_data[product_data['week'] == week].sort_values('day_index')
            plt.plot(week_data['day_index'], week_data['remaining_inventory'], 
                    marker='o', label=f'Week {week}', linewidth=2, markersize=6)
        
        # Highlight sellout threshold
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Sellout Threshold')
        
        # Mark actual sellouts
        sellouts = product_data[product_data['sold_out']]
        if len(sellouts) > 0:
            plt.scatter(sellouts['day_index'], sellouts['remaining_inventory'], 
                       color='red', s=100, marker='X', label='Sellouts', zorder=5)
        
        plt.title(f'Inventory Levels - {product}', fontsize=14, fontweight='bold')
        plt.xlabel('Day of Week')
        plt.ylabel('Remaining Inventory')
        plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save individual product plot
        safe_filename = product.replace(' ', '_').replace('/', '_')
        save_path = os.path.join(product_dir, f"{safe_filename}_inventory.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📦 {product} inventory plot saved")
    
    # 4. Save inventory data as CSV
    inventory_csv = os.path.join(inventory_dir, "inventory_data.csv")
    inventory_df.to_csv(inventory_csv, index=False)
    
    sellout_csv = os.path.join(inventory_dir, "sellout_summary.csv")
    sellout_summary.to_csv(sellout_csv, index=False)
    
    # 5. Create inventory report
    report_file = os.path.join(inventory_dir, "inventory_report.txt")
    with open(report_file, 'w') as f:
        f.write("INVENTORY & SELLOUT ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        # Overall statistics
        total_sellouts = len(inventory_df[inventory_df['sold_out']])
        total_product_days = len(inventory_df)
        sellout_rate = total_sellouts / total_product_days * 100
        
        f.write(f"Overall Statistics:\n")
        f.write(f"• Total product-day combinations: {total_product_days}\n")
        f.write(f"• Total sellouts: {total_sellouts}\n")
        f.write(f"• Sellout rate: {sellout_rate:.1f}%\n\n")
        
        # Sellouts by day
        f.write("Sellouts by Day of Week:\n")
        sellout_by_day = inventory_df[inventory_df['sold_out']].groupby('day_of_week').size()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in day_order:
            count = sellout_by_day.get(day, 0)
            f.write(f"• {day}: {count} sellouts\n")
        f.write("\n")
        
        # Most problematic products
        f.write("Products with Most Sellouts:\n")
        sellout_by_product = inventory_df[inventory_df['sold_out']].groupby('product').size().sort_values(ascending=False)
        for product, count in sellout_by_product.head(10).items():
            f.write(f"• {product}: {count} sellouts\n")
        f.write("\n")
        
        # Products that never sold out
        all_products = set(inventory_df['product'].unique())
        sellout_products = set(inventory_df[inventory_df['sold_out']]['product'].unique())
        no_sellout_products = all_products - sellout_products
        
        if no_sellout_products:
            f.write("Products that Never Sold Out:\n")
            for product in sorted(no_sellout_products):
                f.write(f"• {product}\n")
        
        f.write(f"\nAnalysis completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"📦 Inventory analysis saved to: {inventory_dir}")
    print(f"📋 Inventory report: {report_file}")

def generate_all_saved_analyses(df, prices, supplier_prices, transactions, amounts, output_base_path="./analysis_output"):
    """Generate and save all possible analyses including profit and inventory analysis"""
    
    print("=== GENERATING COMPREHENSIVE SAVED ANALYSIS ===")
    
    # Add profit data to dataframe
    print("Adding profit calculations...")
    df_with_profit = add_profit_data(df, prices, supplier_prices)
    
    # Create output directory structure
    output_dir = create_output_directories(output_base_path)
    
    # Generate sales analyses
    save_overall_analysis(df_with_profit, output_dir)
    save_individual_product_analysis(df_with_profit, output_dir)
    save_individual_worker_analysis(df_with_profit, output_dir)
    save_daily_analysis(df_with_profit, output_dir)
    save_combined_analysis(df_with_profit, output_dir)
    save_profit_analysis(df_with_profit, output_dir)
    
    # Generate inventory analysis
    print("Generating inventory and sellout analysis...")
    inventory_df = calculate_daily_inventory(transactions, amounts)
    sellout_summary = analyze_sellouts(inventory_df)
    save_inventory_analysis(inventory_df, sellout_summary, output_dir)
    # Generate weekly sellout plots for all products
    save_weekly_sellout_plots(inventory_df, output_dir)
    
    # Calculate profit statistics for summary
    total_profit = df_with_profit['total_profit'].sum()
    avg_profit_per_transaction = df_with_profit['total_profit'].mean()
    total_revenue = (df_with_profit['selling_price'] * df_with_profit['amount']).sum()
    total_cost = (df_with_profit['cost_price'] * df_with_profit['amount']).sum()
    overall_profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
    
    # Calculate inventory statistics
    total_sellouts = len(inventory_df[inventory_df['sold_out']])
    total_product_days = len(inventory_df)
    sellout_rate = total_sellouts / total_product_days * 100
    
    # Generate summary report
    summary_file = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("=== SALES, PROFIT & INVENTORY ANALYSIS SUMMARY ===\n\n")
        f.write(f"Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("SALES STATISTICS:\n")
        f.write(f"Total transactions: {len(df_with_profit)}\n")
        f.write(f"Total sales amount: ${df_with_profit['amount'].sum():,.2f}\n")
        f.write(f"Average sale amount: ${df_with_profit['amount'].mean():.2f}\n")
        f.write(f"Products analyzed: {df_with_profit['product'].nunique()}\n")
        f.write(f"Workers analyzed: {df_with_profit['worker_id'].nunique()}\n")
        f.write(f"Days covered: {sorted(df_with_profit['day_name'].unique())}\n")
        f.write(f"Weeks covered: {sorted(df_with_profit['week'].unique())}\n\n")
        
        f.write("PROFIT STATISTICS:\n")
        f.write(f"Total revenue: ${total_revenue:,.2f}\n")
        f.write(f"Total cost: ${total_cost:,.2f}\n")
        f.write(f"Total profit: ${total_profit:,.2f}\n")
        f.write(f"Overall profit margin: {overall_profit_margin:.1f}%\n")
        f.write(f"Average profit per transaction: ${avg_profit_per_transaction:.2f}\n\n")
        
        f.write("INVENTORY STATISTICS:\n")
        f.write(f"Total product-day combinations: {total_product_days}\n")
        f.write(f"Total sellouts occurred: {total_sellouts}\n")
        f.write(f"Overall sellout rate: {sellout_rate:.1f}%\n")
        
        # Most problematic products
        sellout_by_product = inventory_df[inventory_df['sold_out']].groupby('product').size().sort_values(ascending=False)
        if len(sellout_by_product) > 0:
            f.write("Products with most sellouts:\n")
            for product, count in sellout_by_product.head(5).items():
                f.write(f"  {product}: {count} sellouts\n")
        
        # Days with most sellouts
        sellout_by_day = inventory_df[inventory_df['sold_out']].groupby('day_of_week').size().sort_values(ascending=False)
        if len(sellout_by_day) > 0:
            f.write("Days with most sellouts:\n")
            for day, count in sellout_by_day.head(3).items():
                f.write(f"  {day}: {count} sellouts\n")
        f.write("\n")
        
        # Top profitable products
        top_products = df_with_profit.groupby('product')['total_profit'].sum().sort_values(ascending=False).head(5)
        f.write("TOP 5 MOST PROFITABLE PRODUCTS:\n")
        for i, (product, profit) in enumerate(top_products.items(), 1):
            f.write(f"{i}. {product}: ${profit:,.2f}\n")
        f.write("\n")
        
        # Weekly profit breakdown
        weekly_profits = df_with_profit.groupby('week')['total_profit'].sum()
        f.write("WEEKLY PROFIT BREAKDOWN:\n")
        for week, profit in weekly_profits.items():
            f.write(f"Week {week}: ${profit:,.2f}\n")
        f.write("\n")
        
        if 'worker_name' in df_with_profit.columns:
            f.write("Worker names were successfully loaded and used in analysis.\n")
        else:
            f.write("Worker names were not available - using worker IDs.\n")
        
        f.write("Profit calculations include selling prices, cost prices, and profit margins.\n")
        f.write("Inventory analysis includes daily stock levels and sellout detection.\n")
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"All analyses saved to: {output_dir}")
    print(f"Summary report: {summary_file}")
    print(f"💰 Total profit analyzed: ${total_profit:,.2f}")
    print(f"📊 Profit margin: {overall_profit_margin:.1f}%")
    print(f"📦 Sellouts detected: {total_sellouts} ({sellout_rate:.1f}% of product-days)")
    
    return output_dir

def plot_all_analyses(df):
    """Run all plotting functions"""
    
    print("Generating all sales analysis plots...\n")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Generate all plots
    plot_sales_by_day_and_product(df, "Sales Heatmap: Day vs Product")
    plot_sales_by_week_and_product(df, "Sales by Week and Product")
    plot_daily_sales_trends(df, "Daily Sales Trends by Product")
    plot_product_distribution(df, "Product Sales Distribution")
    plot_weekly_comparison(df, "Weekly Sales Comparison")
    
    # Generate summary statistics
    generate_summary_statistics(df)

if __name__ == "__main__":
    BASE_PATH = "C:/Users/berke/Downloads/torgets_butik_id/torgets_butik_id"
    
    print("Loading data...")
    transactions, schedules, prices, amounts = load_data(BASE_PATH)
    
    print("Processing transactions data...")
    df = process_transactions_data(transactions)
    
    print("Loading worker names...")
    try:
        worker_mapping = load_worker_names(BASE_PATH)
        df = add_worker_names(df, worker_mapping)
        print(f"Successfully loaded {len(worker_mapping)} worker names")
        print("Sample worker names:", list(worker_mapping.values())[:5])
    except FileNotFoundError:
        print("Worker names file not found. Using worker IDs only.")
        worker_mapping = {}
    
    print("Loading supplier prices for profit calculation...")
    try:
        supplier_prices = load_supplier_prices(BASE_PATH)
        print(f"Successfully loaded supplier prices for {len(supplier_prices)} products")
    except FileNotFoundError:
        print("Supplier prices file not found. Profit analysis will be skipped.")
        supplier_prices = {}
    
    print(f"Loaded {len(df)} transaction records from {len(transactions)} weeks")
    print(f"Products available: {df['product'].unique()}")
    print(f"Days covered: {sorted(df['day_name'].unique())}")
    print()
    
    # Choose analysis type
    print("Choose analysis type:")
    print("1. Original analysis (basic plots - display only)")
    print("2. Comprehensive analysis (all plots - display only)")
    print("3. Save all analyses to files (comprehensive + individual analyses)")
    
    # For automatic execution, use save all analyses
    # You can uncomment the input line below for interactive selection
    # choice = input("Enter choice (1, 2, or 3): ").strip()
    choice = "3"  # Default to save all
    
    if choice == "1":
        print("\nRunning original analysis...")
        plot_all_analyses(df)
    elif choice == "2":
        print("\nRunning comprehensive analysis...")
        create_comprehensive_analysis(df)
    elif choice == "3":
        print("\nGenerating and saving all analyses...")
        output_dir = generate_all_saved_analyses(df, prices, supplier_prices, transactions, amounts, "./analysis_output")
        print(f"All analyses saved to: {output_dir}")
    else:
        print("Invalid choice. Running comprehensive analysis...")
        create_comprehensive_analysis(df)
    
    print("\nData analysis complete!")
