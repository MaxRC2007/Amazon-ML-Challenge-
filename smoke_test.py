import sys
sys.path.insert(0, 'code/business_entity_resolution')
import pandas as pd
from src.normalization import add_normalized_columns
from src.blocking import run_stage_a, measure_recall

# Tiny smoke test with a few rows per source
s1 = pd.DataFrame([
    ['S1-1', 'Custom Wealth Services LLC', '5559 Orville Ave, Columbus OH', 'US'],
    ['S1-2', 'Ram Marketing Pvt Ltd', '23 MG Road, Mumbai 400001', 'India'],
    ['S1-3', 'Nexus Anchor Rain', '1111 Church St, Nashville TN', 'US'],
], columns=['entity_id','business_name','business_address','country'])

s2 = pd.DataFrame([
    ['S2-1', 'Custom Wealth Service LLC', '5559 Orville Columbus', 'US'],
    ['S2-2', 'Rain Anchor Nexus', '1111 Church Street Nashville', 'US'],
], columns=['entity_id','business_name','business_address','country'])

s3 = pd.DataFrame([
    ['S3-1', 'राम मार्केटिंग प्राइवेट लिमिटेड', 'MG Road Mumbai 400001', 'India'],
    ['S3-2', 'Wealth Custom Services', 'Columbus OH 43204', 'US'],
], columns=['entity_id','business_name','business_address','country'])

add_normalized_columns(s1)
add_normalized_columns(s2)
add_normalized_columns(s3)

print("=== Normalized S1 names ===")
print(s1[['entity_id','norm_name','norm_name_sorted']].to_string())

cands = run_stage_a(s1, s2, s3)
print("\n=== Candidate pairs ===")
print(cands[['source1_entity_id','candidate_entity_ids','n_candidates']].to_string())

gt = pd.DataFrame([
    ['S1-1', 'S2-1,S3-2'],
    ['S1-2', 'S3-1'],
    ['S1-3', 'S2-2'],
], columns=['source1_entity_id','matched_entity_ids'])

stats = measure_recall(cands, gt)
recall_val = stats["recall"]
recovered  = stats["n_recovered"]
total      = stats["n_true_matches"]
print(f"\nRecall: {recall_val:.4f}  ({recovered}/{total} matches recovered)")
