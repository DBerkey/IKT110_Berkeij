"""
Script to analyze weekly supply costs - how much money spent on buying supplies each week
"""

from data_analyses import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_weekly_supply_costs():
    """Calculate and visualize weekly supply costs"""
    
    BASE_PATH = "C:/Users/berke/Downloads/torgets_butik_id/torgets_butik_id"
    
    print("💰 WEEKLY SUPPLY COST ANALYSIS")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    transactions, schedules, prices, amounts = load_data(BASE_PATH)
    df = process_transactions_data(transactions)
    
    # Load supplier prices
    try:
        supplier_prices = load_supplier_prices(BASE_PATH)
        print(f"✅ Loaded supplier prices for {len(supplier_prices)} products")
    except FileNotFoundError:
        print("❌ Supplier prices file not found!")
        return
    
    print(f"✅ Analyzed {len(df)} transactions across {len(df['week'].unique())} weeks")
    print()
    
    # Calculate supply costs for each transaction (what you paid to suppliers)
    def get_supply_cost(row):
        product = row['product']
        amount = row['amount']
        cost_per_unit = supplier_prices.get(product, 0)
        return cost_per_unit * amount
    
    df['supply_cost'] = df.apply(get_supply_cost, axis=1)
    
    # Group by week to get weekly supply costs
    weekly_costs = df.groupby('week').agg({
        'supply_cost': 'sum',
        'amount': 'sum'  # total units sold
    }).round(2)
    
    weekly_costs.columns = ['total_supply_cost', 'total_units_sold']
    weekly_costs['avg_cost_per_unit'] = (weekly_costs['total_supply_cost'] / 
                                        weekly_costs['total_units_sold']).round(2)
    
    # Display results
    print("📊 WEEKLY SUPPLY COSTS:")
    print("-" * 30)
    total_supply_cost = 0
    for week in sorted(weekly_costs.index):
        cost = weekly_costs.loc[week, 'total_supply_cost']
        units = weekly_costs.loc[week, 'total_units_sold']
        avg_cost = weekly_costs.loc[week, 'avg_cost_per_unit']
        total_supply_cost += cost
        print(f"Week {week}: ${cost:,.2f} (sold {units:,.0f} units, avg ${avg_cost:.2f}/unit)")
    
    print(f"\n💵 TOTAL SUPPLY COSTS: ${total_supply_cost:,.2f}")
    print(f"📈 AVERAGE WEEKLY COST: ${total_supply_cost/len(weekly_costs):,.2f}")
    
    # Calculate supply costs by product
    print("\n📦 SUPPLY COSTS BY PRODUCT:")
    print("-" * 35)
    
    product_costs = df.groupby('product').agg({
        'supply_cost': 'sum',
        'amount': 'sum'
    }).round(2)
    
    product_costs.columns = ['total_supply_cost', 'total_units_sold']
    product_costs = product_costs.sort_values('total_supply_cost', ascending=False)
    
    for product in product_costs.index:
        cost = product_costs.loc[product, 'total_supply_cost']
        units = product_costs.loc[product, 'total_units_sold']
        unit_cost = supplier_prices.get(product, 0)
        print(f"{product}: ${cost:,.2f} ({units:,.0f} units × ${unit_cost:.2f})")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Weekly Supply Cost Analysis', fontsize=16, fontweight='bold')
    
    # 1. Weekly supply costs bar chart
    ax1 = axes[0, 0]
    weeks = weekly_costs.index
    costs = weekly_costs['total_supply_cost']
    bars1 = ax1.bar(weeks, costs, color='lightcoral', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, cost in zip(bars1, costs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(costs)*0.01,
                f'${cost:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('Weekly Supply Costs')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Supply Cost ($)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Supply costs by product (top 10)
    ax2 = axes[0, 1]
    top_products = product_costs.head(10)
    bars2 = ax2.barh(range(len(top_products)), top_products['total_supply_cost'], 
                     color='lightblue', alpha=0.8, edgecolor='black')
    
    ax2.set_yticks(range(len(top_products)))
    ax2.set_yticklabels(top_products.index)
    ax2.set_title('Supply Costs by Product (Top 10)')
    ax2.set_xlabel('Total Supply Cost ($)')
    
    # Add value labels
    for i, (bar, cost) in enumerate(zip(bars2, top_products['total_supply_cost'])):
        ax2.text(bar.get_width() + max(top_products['total_supply_cost'])*0.01, 
                bar.get_y() + bar.get_height()/2,
                f'${cost:,.0f}', ha='left', va='center', fontsize=9)
    
    # 3. Weekly units sold vs supply cost
    ax3 = axes[1, 0]
    ax3.scatter(weekly_costs['total_units_sold'], weekly_costs['total_supply_cost'], 
               s=100, color='green', alpha=0.7, edgecolor='black')
    
    # Add week labels
    for week in weekly_costs.index:
        units = weekly_costs.loc[week, 'total_units_sold']
        cost = weekly_costs.loc[week, 'total_supply_cost']
        ax3.annotate(f'Week {week}', (units, cost), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
    
    ax3.set_title('Units Sold vs Supply Cost')
    ax3.set_xlabel('Total Units Sold')
    ax3.set_ylabel('Supply Cost ($)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Average cost per unit by week
    ax4 = axes[1, 1]
    bars4 = ax4.bar(weeks, weekly_costs['avg_cost_per_unit'], 
                    color='orange', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, avg_cost in zip(bars4, weekly_costs['avg_cost_per_unit']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(weekly_costs['avg_cost_per_unit'])*0.01,
                f'${avg_cost:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_title('Average Supply Cost per Unit by Week')
    ax4.set_xlabel('Week')
    ax4.set_ylabel('Average Cost per Unit ($)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "./weekly_supply_costs_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Analysis saved: {output_path}")
    plt.show()
    
    # Create a detailed breakdown table
    print("\n📋 DETAILED WEEKLY BREAKDOWN:")
    print("-" * 60)
    print(f"{'Week':<6} {'Supply Cost':<12} {'Units Sold':<12} {'Avg $/Unit':<12}")
    print("-" * 60)
    
    for week in sorted(weekly_costs.index):
        cost = weekly_costs.loc[week, 'total_supply_cost']
        units = weekly_costs.loc[week, 'total_units_sold']
        avg_cost = weekly_costs.loc[week, 'avg_cost_per_unit']
        print(f"{week:<6} ${cost:<11,.2f} {units:<12,.0f} ${avg_cost:<11.2f}")
    
    print("-" * 60)
    print(f"{'TOTAL':<6} ${total_supply_cost:<11,.2f} {weekly_costs['total_units_sold'].sum():<12,.0f} ${weekly_costs['total_supply_cost'].sum()/weekly_costs['total_units_sold'].sum():<11.2f}")
    
    # Calculate and show supplier cost breakdown
    print(f"\n💡 INSIGHTS:")
    print(f"• Most expensive week: Week {weekly_costs['total_supply_cost'].idxmax()} (${weekly_costs['total_supply_cost'].max():,.2f})")
    print(f"• Least expensive week: Week {weekly_costs['total_supply_cost'].idxmin()} (${weekly_costs['total_supply_cost'].min():,.2f})")
    print(f"• Most expensive product to source: {product_costs.index[0]} (${product_costs.iloc[0]['total_supply_cost']:,.2f} total)")
    print(f"• Average weekly supply cost: ${weekly_costs['total_supply_cost'].mean():,.2f}")
    
    return weekly_costs, product_costs

if __name__ == "__main__":
    weekly_data, product_data = analyze_weekly_supply_costs()