"""
Utility to show the contents of generated analysis folders
"""

import os
from datetime import datetime

def show_analysis_structure(analysis_dir):
    """Show the structure of generated analysis files"""
    
    if not os.path.exists(analysis_dir):
        print(f"Analysis directory not found: {analysis_dir}")
        return
    
    print(f"=== ANALYSIS STRUCTURE ===")
    print(f"Directory: {analysis_dir}")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Count files by category
    file_counts = {
        'overall': 0,
        'products': 0,
        'workers': 0,
        'daily': 0,
        'combined': 0,
        'total': 0
    }
    
    for root, dirs, files in os.walk(analysis_dir):
        for file in files:
            if file.endswith('.png'):
                file_counts['total'] += 1
                if '01_overall' in root:
                    file_counts['overall'] += 1
                elif '02_product' in root:
                    file_counts['products'] += 1
                elif '03_worker' in root:
                    file_counts['workers'] += 1
                elif '04_daily' in root:
                    file_counts['daily'] += 1
                elif '05_combined' in root:
                    file_counts['combined'] += 1
    
    print("FILE SUMMARY:")
    print(f"📊 Overall Analysis: {file_counts['overall']} plots")
    print(f"📦 Product Analysis: {file_counts['products']} plots")
    print(f"👥 Worker Analysis: {file_counts['workers']} plots")
    print(f"📅 Daily Analysis: {file_counts['daily']} plots")
    print(f"🔗 Combined Analysis: {file_counts['combined']} plots")
    print(f"📈 Total Plots Generated: {file_counts['total']}")
    print()
    
    # Show directory structure
    print("DIRECTORY STRUCTURE:")
    for root, dirs, files in os.walk(analysis_dir):
        level = root.replace(analysis_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        folder_name = os.path.basename(root)
        if level == 0:
            folder_name = "📁 " + os.path.basename(analysis_dir)
        else:
            folder_name = "📂 " + folder_name
        print(f'{indent}{folder_name}')
        
        subindent = ' ' * 2 * (level + 1)
        png_files = [f for f in files if f.endswith('.png')]
        other_files = [f for f in files if not f.endswith('.png')]
        
        for file in other_files:
            if file.endswith('.txt'):
                print(f'{subindent}📄 {file}')
        
        if png_files:
            print(f'{subindent}🖼️  {len(png_files)} plot files (.png)')

def find_latest_analysis():
    """Find the most recent analysis directory"""
    
    analysis_base = "./analysis_output"
    if not os.path.exists(analysis_base):
        print("No analysis output directory found.")
        return None
    
    # Find all analysis directories
    analysis_dirs = []
    for item in os.listdir(analysis_base):
        item_path = os.path.join(analysis_base, item)
        if os.path.isdir(item_path) and item.startswith("sales_analysis_"):
            analysis_dirs.append(item_path)
    
    if not analysis_dirs:
        print("No analysis directories found.")
        return None
    
    # Return the most recent one
    return max(analysis_dirs, key=os.path.getctime)

if __name__ == "__main__":
    # Find and show the latest analysis
    latest_dir = find_latest_analysis()
    if latest_dir:
        show_analysis_structure(latest_dir)
        
        # Show summary file if it exists
        summary_file = os.path.join(latest_dir, "analysis_summary.txt")
        if os.path.exists(summary_file):
            print("\nANALYSIS SUMMARY:")
            with open(summary_file, 'r') as f:
                print(f.read())
    else:
        print("No analysis found. Run the data analysis first.")