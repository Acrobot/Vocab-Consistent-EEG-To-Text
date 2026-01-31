import os
import glob
import re
import ast
import csv
import argparse
import sys

def parse_score_file(filepath):
    """
    Parses a single score text file and extracts metrics.
    Returns a dictionary of metric_name -> value.
    """
    scores = {}
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        content = f.read()

        # Regex patterns for parsing content
        patterns = {
            'WER': (r'wer_scores = ([\d\.]+)', float),
            'CER': (r'cer_scores = ([\d\.]+)', float),
            'WER (TF)': (r'wer_scores_with_tf = ([\d\.]+)', float),
            'CER (TF)': (r'cer_scores_with_tf = ([\d\.]+)', float),
            'SacreBLEU': (r"sacre_blue_score = \{'score': ([\d\.]+)", float),
            'SacreBLEU (TF)': (r"sacre_blue_score_with_tf = \{'score': ([\d\.]+)", float),
            'BLEU List': (r'corpus_bleu_score = (\[.*?\])', list),
            'BLEU List (TF)': (r'corpus_bleu_score_with_tf = (\[.*?\])', list),
            'ROUGE Dict': (r'rouge_scores = (\{.*?\})(?=\n|$)', dict),
            'ROUGE Dict (TF)': (r'rouge_scores_with_tf = (\{.*?\})(?=\n|$)', dict)
        }

        for key, (pat, type_func) in patterns.items():
            match = re.search(pat, content, re.DOTALL)
            if match:
                val_str = match.group(1)
                try:
                    if type_func in [list, dict]:
                        val = ast.literal_eval(val_str)
                    else:
                        val = type_func(val_str)
                    scores[key] = val
                except Exception as e:
                    print(f"[WARN] Error parsing {key} in {filepath}: {e}")
            else:
                scores[key] = None

    # Flatten complex structures
    flat_scores = {}
    
    # 1. Scalars
    scalar_keys = ['WER', 'CER', 'WER (TF)', 'CER (TF)', 'SacreBLEU', 'SacreBLEU (TF)']
    for k in scalar_keys:
        if scores.get(k) is not None:
            flat_scores[k] = scores[k]

    # 2. BLEU Lists (1-4)
    if scores.get('BLEU List'):
        for i, val in enumerate(scores['BLEU List']):
            flat_scores[f'BLEU-{i+1}'] = val
    if scores.get('BLEU List (TF)'):
        for i, val in enumerate(scores['BLEU List (TF)']):
            flat_scores[f'BLEU-{i+1} (TF)'] = val

    # 3. ROUGE Dicts
    # Structure: {'rouge-1': {'f': ..., 'p': ..., 'r': ...}, ...}
    for base_key, prefix in [('ROUGE Dict', ''), ('ROUGE Dict (TF)', ' (TF)')]:
        if scores.get(base_key):
            try:
                r_dict = scores[base_key]
                flat_scores[f'ROUGE-1{prefix}'] = r_dict['rouge-1']['f']
                flat_scores[f'ROUGE-2{prefix}'] = r_dict['rouge-2']['f']
                flat_scores[f'ROUGE-L{prefix}'] = r_dict['rouge-l']['f']
            except KeyError:
                pass

    return flat_scores

def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results into a CSV file.")
    parser.add_argument('--input_dir', '-i', required=True, help="Root directory containing subject subfolders (e.g. ./score_results/optimal_fixed)")
    parser.add_argument('--output_csv', '-o', required=True, help="Path to output CSV file")
    args = parser.parse_args()

    base_dir = args.input_dir
    output_file = args.output_csv

    if not os.path.isdir(base_dir):
        print(f"Error: Input directory {base_dir} does not exist.")
        sys.exit(1)

    print(f"Scanning {base_dir}...")
    
    # Collect all data
    rows = []
    all_metrics = set()

    # Walk through directory
    # Expected structure: base_dir / Subject / InputType / *.txt
    subjects = sorted(os.listdir(base_dir))
    
    for subj in subjects:
        subj_dir = os.path.join(base_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
            
        # Look for subdirectories like 'EEG', 'noise', etc.
        input_types = os.listdir(subj_dir)
        for input_type in input_types:
            input_dir = os.path.join(subj_dir, input_type)
            if not os.path.isdir(input_dir):
                continue
            
            # Find result text files
            txt_files = glob.glob(os.path.join(input_dir, '*.txt'))
            if not txt_files:
                continue
            
            # Parse the first txt file found (assuming 1 per input type)
            scores = parse_score_file(txt_files[0])
            if not scores:
                continue
            
            row = {
                'Subject': subj,
                'Input': input_type,
                'File': os.path.basename(txt_files[0])
            }
            row.update(scores)
            
            rows.append(row)
            all_metrics.update(scores.keys())

    if not rows:
        print("No results found.")
        sys.exit(0)

    # Sort columns: Meta headers first, then metrics sorted alphabetically
    meta_headers = ['Subject', 'Input', 'File']
    metric_headers = sorted(list(all_metrics))
    
    # Custom sort for standard metrics logic (BLEU-1 before BLEU-2, etc.)
    # But simple alpha sort is okay for now, or we can enforce a specific order.
    # Enforcing specific order for readability
    preferred_order = [
        'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'WER', 'CER', 'ROUGE-1', 'ROUGE-L', 'SacreBLEU',
        'BLEU-1 (TF)', 'BLEU-2 (TF)', 'BLEU-3 (TF)', 'BLEU-4 (TF)', 'WER (TF)', 'CER (TF)', 'ROUGE-1 (TF)', 'ROUGE-L (TF)', 'SacreBLEU (TF)'
    ]
    
    final_headers = meta_headers + [h for h in preferred_order if h in metric_headers] + [h for h in metric_headers if h not in preferred_order]

    # Write to CSV
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_headers)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Successfully aggregated {len(rows)} records to {output_file}")

if __name__ == "__main__":
    main()
