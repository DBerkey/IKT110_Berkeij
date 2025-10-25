"""
Simple script to generate and save all analyses without displaying plots
Run this when you just want to save all plots to files
"""

from data_analyses import *

def run_analysis_save_only():
    """Run complete analysis and save all plots to files"""
    
    BASE_PATH = "C:/Users/berke/Downloads/torgets_butik_id/torgets_butik_id"
    
    print("=== SALES ANALYSIS - SAVE TO FILES ===")
    print("Loading data...")
    
    # Load and process data
    transactions, schedules, prices, amounts = load_data(BASE_PATH)
    df = process_transactions_data(transactions)
    
    # Load worker names
    try:
        worker_mapping = load_worker_names(BASE_PATH)
        df = add_worker_names(df, worker_mapping)
        print(f"✅ Loaded {len(worker_mapping)} worker names")
    except FileNotFoundError:
        print("⚠️ Worker names file not found. Using worker IDs only.")
    
    # Load supplier prices
    try:
        supplier_prices = load_supplier_prices(BASE_PATH)
        print(f"✅ Loaded supplier prices for {len(supplier_prices)} products")
    except FileNotFoundError:
        print("⚠️ Supplier prices file not found. Profit analysis will be limited.")
        supplier_prices = {}
    
    print(f"✅ Loaded {len(df)} transaction records")
    print(f"📦 Products: {df['product'].nunique()}")
    print(f"👥 Workers: {df['worker_id'].nunique()}")
    print()
    
    # Generate and save all analyses
    output_dir = generate_all_saved_analyses(df, prices, supplier_prices, transactions, amounts, "./analysis_output")
    
    # Show quick summary
    print(f"\n🎉 SUCCESS!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Generated comprehensive analysis with individual reports for:")
    print(f"   • Every product ({df['product'].nunique()} products)")
    print(f"   • Every worker ({df['worker_name'].nunique() if 'worker_name' in df.columns else df['worker_id'].nunique()} workers)")
    print(f"   • Every day of the week (7 days)")
    print(f"   • Combined analyses and overall summaries")
    print(f"   • 💰 Profit analysis for every product every week")
    print(f"   • 📦 Inventory & sellout analysis by product and day")
    print()
    print("💡 Run 'python show_analysis_results.py' to see detailed structure")

if __name__ == "__main__":
    run_analysis_save_only()