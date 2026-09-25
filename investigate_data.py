import csv
import os
import re
import sys

def analyze_dataset():
    data_dir = "dataset/train"
    files = {
        "source1": "train_source1.tsv",
        "source2": "train_source2.tsv",
        "source3": "train_source3.tsv",
        "ground_truth": "train_ground_truth.tsv"
    }

    results = {}
    
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        expected_cols = 2 if name == "ground_truth" else 4
        
        wrong_col_count = 0
        non_ascii_name_count = 0
        total_rows = 0
        
        sample_wrong_cols = []
        sample_non_ascii = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
                header = next(reader, None)
                
                for row in reader:
                    total_rows += 1
                    if len(row) != expected_cols:
                        wrong_col_count += 1
                        if len(sample_wrong_cols) < 20:
                            sample_wrong_cols.append(row)
                    
                    if name != "ground_truth" and len(row) >= 2:
                        b_name = row[1]
                        if any(ord(c) >= 128 for c in b_name):
                            non_ascii_name_count += 1
                            if len(sample_non_ascii) < 20:
                                sample_non_ascii.append(b_name)
                                
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
        results[name] = {
            "total_rows": total_rows,
            "wrong_col_count": wrong_col_count,
            "non_ascii_name_count": non_ascii_name_count,
            "sample_wrong_cols": sample_wrong_cols,
            "sample_non_ascii": sample_non_ascii
        }

    for name, stat in results.items():
        print(f"=== {name} ===")
        print(f"Total Rows: {stat['total_rows']}")
        print(f"Wrong Column Count Rows: {stat['wrong_col_count']} ({stat['wrong_col_count'] / max(1, stat['total_rows']) * 100:.4f}%)")
        print(f"Non-ASCII Business Names: {stat['non_ascii_name_count']} ({stat['non_ascii_name_count'] / max(1, stat['total_rows']) * 100:.4f}%)")
        print("Sample Wrong Column Rows (up to 5):")
        for r in stat['sample_wrong_cols'][:5]:
            print("  ", r)
        print("Sample Non-ASCII Names (up to 5):")
        for n in stat['sample_non_ascii'][:5]:
            print("  ", n)
        print("\n")

if __name__ == "__main__":
    analyze_dataset()
