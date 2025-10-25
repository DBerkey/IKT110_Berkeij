"""
Quick profit analysis summary
Shows key profit insights from the data
"""

from data_analyses import *

def show_profit_summary():
    """Show key profit insights"""
    
    BASE_PATH = "C:/Users/berke/Downloads/torgets_butik_id/torgets_butik_id"
    
    print("=== PROFIT ANALYSIS SUMMARY ===")
    print("Loading data...")
    
    # Load all data
    transactions, schedules, prices, amounts = load_data(BASE_PATH)
    df = process_transactions_data(transactions)
    
    # Load worker names and supplier prices
    worker_mapping = load_worker_names(BASE_PATH)
    df = add_worker_names(df, worker_mapping)
    supplier_prices = load_supplier_prices(BASE_PATH)
    
    # Add profit calculations
    df = add_profit_data(df, prices, supplier_prices)
    
    print("\n💰 OVERALL PROFIT SUMMARY")
    total_profit = df['total_profit'].sum()
    total_revenue = (df['selling_price'] * df['amount']).sum()
    total_cost = (df['cost_price'] * df['amount']).sum()
    profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
    
    print(f"Total Revenue: ${total_revenue:,.2f}")
    print(f"Total Cost: ${total_cost:,.2f}")
    print(f"Total Profit: ${total_profit:,.2f}")
    print(f"Profit Margin: {profit_margin:.1f}%")
    
    print("\n📦 PROFIT BY PRODUCT")
    product_profit = df.groupby('product').agg({
        'total_profit': 'sum',
        'amount': 'sum',
        'selling_price': 'mean',
        'cost_price': 'mean'
    }).round(2)
    product_profit['profit_margin'] = ((product_profit['selling_price'] - product_profit['cost_price']) / product_profit['selling_price'] * 100).round(1)
    product_profit = product_profit.sort_values('total_profit', ascending=False)
    
    print(f"{'Product':<15} {'Total Profit':<12} {'Quantity':<8} {'Margin %':<8}")
    print("-" * 50)
    for product, row in product_profit.head(10).iterrows():
        print(f"{product:<15} ${row['total_profit']:>9,.0f} {row['amount']:>7.0f} {row['profit_margin']:>6.1f}%")
    
    print("\n📅 PROFIT BY WEEK")
    weekly_profit = df.groupby('week').agg({
        'total_profit': 'sum',
        'amount': 'sum'
    }).round(2)
    
    print(f"{'Week':<6} {'Total Profit':<12} {'Quantity':<8}")
    print("-" * 30)
    for week, row in weekly_profit.iterrows():
        print(f"{week:<6} ${row['total_profit']:>9,.0f} {row['amount']:>7.0f}")
    
    print("\n📈 PRODUCT-WEEK PROFIT BREAKDOWN")
    print("Top profit-generating product-week combinations:")
    product_week_profit = df.groupby(['product', 'week'])['total_profit'].sum().sort_values(ascending=False)
    
    print(f"{'Product':<15} {'Week':<6} {'Profit':<10}")
    print("-" * 35)
    for (product, week), profit in product_week_profit.head(10).items():
        print(f"{product:<15} {week:<6} ${profit:>8,.0f}")
    
    print("\n👥 TOP PROFIT-GENERATING WORKERS")
    worker_profit = df.groupby('worker_name')['total_profit'].sum().sort_values(ascending=False)
    
    print(f"{'Worker':<20} {'Total Profit':<12}")
    print("-" * 35)
    for worker, profit in worker_profit.head(10).items():
        print(f"{worker:<20} ${profit:>9,.0f}")
    
    print(f"\n🎯 KEY INSIGHTS:")
    best_product = product_profit.index[0]
    best_week = weekly_profit['total_profit'].idxmax()
    best_margin_product = product_profit['profit_margin'].idxmax()
    
    print(f"• Most profitable product: {best_product} (${product_profit.loc[best_product, 'total_profit']:,.0f})")
    print(f"• Most profitable week: Week {best_week} (${weekly_profit.loc[best_week, 'total_profit']:,.0f})")
    print(f"• Highest profit margin: {best_margin_product} ({product_profit.loc[best_margin_product, 'profit_margin']:.1f}%)")
    print(f"• Average profit per transaction: ${df['total_profit'].mean():.2f}")

if __name__ == "__main__":
    show_profit_summary()