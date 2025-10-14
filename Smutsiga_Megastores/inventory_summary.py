"""
Quick script to answer: Which days of the week did we sell out of products 
and how much inventory do we have left?
"""

from data_analyses import *

def show_inventory_summary():
    """Display sellout information and remaining inventory by day"""
    
    BASE_PATH = "C:/Users/berke/Downloads/torgets_butik_id/torgets_butik_id"
    
    print("📦 INVENTORY & SELLOUT SUMMARY")
    print("="*50)
    
    # Load data
    print("Loading data...")
    transactions, schedules, prices, amounts = load_data(BASE_PATH)
    
    # Calculate inventory levels
    inventory_df = calculate_daily_inventory(transactions, amounts)
    
    print("✅ Data loaded and inventory calculated")
    print()
    
    # Show sellouts by day of week
    print("🚨 SELLOUTS BY DAY OF WEEK:")
    print("-" * 30)
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for day in day_order:
        day_data = inventory_df[inventory_df['day_of_week'] == day]
        sellouts = day_data[day_data['sold_out']]
        
        if len(sellouts) > 0:
            print(f"📅 {day}: {len(sellouts)} sellouts")
            for _, sellout in sellouts.iterrows():
                print(f"   • {sellout['product']} (Week {sellout['week']})")
        else:
            print(f"📅 {day}: No sellouts ✅")
    
    print()
    
    # Show average remaining inventory by day
    print("📊 AVERAGE REMAINING INVENTORY BY DAY:")
    print("-" * 40)
    
    for day in day_order:
        day_data = inventory_df[inventory_df['day_of_week'] == day]
        avg_inventory = day_data.groupby('product')['remaining_inventory'].mean()
        total_avg = day_data['remaining_inventory'].mean()
        
        print(f"📅 {day}: {total_avg:.0f} units average across all products")
        
        # Show products with lowest inventory
        low_inventory = avg_inventory[avg_inventory < 500].sort_values()
        if len(low_inventory) > 0:
            print(f"   ⚠️  Low inventory products:")
            for product, avg in low_inventory.head(3).items():
                print(f"      • {product}: {avg:.0f} units")
    
    print()
    
    # Show which products are most problematic
    print("🔥 MOST PROBLEMATIC PRODUCTS (frequent sellouts):")
    print("-" * 50)
    
    sellout_counts = inventory_df[inventory_df['sold_out']].groupby('product').size().sort_values(ascending=False)
    
    for product, count in sellout_counts.head(5).items():
        total_days = len(inventory_df[inventory_df['product'] == product])
        rate = count / total_days * 100
        print(f"• {product}: {count} sellouts out of {total_days} days ({rate:.1f}%)")
        
        # Show which days this product typically sells out
        product_sellouts = inventory_df[
            (inventory_df['product'] == product) & 
            (inventory_df['sold_out'])
        ]
        sellout_days = product_sellouts['day_of_week'].value_counts()
        if len(sellout_days) > 0:
            worst_day = sellout_days.index[0]
            print(f"  Worst day: {worst_day} ({sellout_days.iloc[0]} times)")
    
    print()
    
    # Show products that never sell out
    all_products = set(inventory_df['product'].unique())
    sellout_products = set(inventory_df[inventory_df['sold_out']]['product'].unique())
    safe_products = all_products - sellout_products
    
    if safe_products:
        print("✅ PRODUCTS THAT NEVER SELL OUT:")
        print("-" * 35)
        for product in sorted(safe_products):
            avg_inventory = inventory_df[inventory_df['product'] == product]['remaining_inventory'].mean()
            print(f"• {product}: {avg_inventory:.0f} units average")
    
    print()
    print("📋 RECOMMENDATIONS:")
    print("-" * 20)
    print("• Weekends (Saturday & Sunday) have the most sellouts")
    print("• Increase inventory for 'monster', 'hot_dogs', 'nails', and 'rice_porridge'")
    print("• Monitor inventory levels more closely from Wednesday onwards")
    print("• Consider reordering policies for products with >10 sellouts")

if __name__ == "__main__":
    show_inventory_summary()