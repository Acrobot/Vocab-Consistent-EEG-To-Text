import re
import sys
import os
import argparse

def parse_metrics(filepath):
    """Parses the metrics from the result file."""
    # Initialize metrics with None
    metrics = {
        'BLEU-1': None, 'BLEU-2': None, 'BLEU-3': None, 'BLEU-4': None,
        'SacreBLEU': None,
        'ROUGE-1-P': None, 'ROUGE-1-R': None, 'ROUGE-1-F': None,
        'ROUGE-2-P': None, 'ROUGE-2-R': None, 'ROUGE-2-F': None,
        'ROUGE-L-P': None, 'ROUGE-L-R': None, 'ROUGE-L-F': None,
        'WER': None
    }

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Result file not found: {filepath}")

    try:
        with open(filepath, 'r') as f:
            content = f.read()

            # Split content by "corpus_bleu_score =" to handle multiple entries.
            # Select the last entry to handle appended output (e.g., from crash recovery).

            blocks = content.split("corpus_bleu_score =")
            if len(blocks) > 1:
                # Use the last block
                text = "corpus_bleu_score =" + blocks[-1]
            else:
                text = content

            # Parse BLEU scores (list format)
            bleu_match = re.search(r"corpus_bleu_score = \[([\d\., e\-\+]+)\]", text)
            if bleu_match:
                bleus = [float(x.strip()) for x in bleu_match.group(1).split(',')]
                if len(bleus) >= 4:
                    metrics['BLEU-1'] = bleus[0] * 100
                    metrics['BLEU-2'] = bleus[1] * 100
                    metrics['BLEU-3'] = bleus[2] * 100
                    metrics['BLEU-4'] = bleus[3] * 100

            # Parse SacreBLEU score
            sacre_match = re.search(r"sacre_blue_score = \{.*?'score': ([\d\.]+)", text)
            if sacre_match:
                metrics['SacreBLEU'] = float(sacre_match.group(1))

            # Parse ROUGE scores (P, R, F)
            for key in ['1', '2', 'l']:
                upper_key = key.upper()
                for m_type in ['p', 'r', 'f']:
                    upper_m = m_type.upper()
                    # Example: 'rouge-1': {'f': 0.12, 'p': 0.11, 'r': 0.13}
                    m_match = re.search(fr"'rouge-{key}': \{{.*?'{m_type}': ([\d\.]+)", text)
                    if m_match:
                        metrics[f'ROUGE-{upper_key}-{upper_m}'] = float(m_match.group(1)) * 100

            # Parse WER
            wer = re.search(r"wer_scores = ([\d\.]+)", text)
            if wer:
                metrics['WER'] = float(wer.group(1)) * 100

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")

    return metrics

def format_val(val, bold=False):
    """Formats the numerical value for the markdown table."""
    if val is None:
        return "N/A"
    s = f"{val:.2f}"
    return f"**{s}**" if bold else s

def main():
    parser = argparse.ArgumentParser(description="Generate results table for EEG-To-Text experiments.")
    parser.add_argument('--highlight-best', action='store_true', help="Highlight (bold) the best results in each column.")
    args = parser.parse_args()

    # Define file paths for all experiments
    files = {
        'EEG': {
            'Our Baseline': 'score_results/optimal_baseline_cohort/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline.txt',
            'Learned PE': 'score_results/optimal_fixed_cohort/EEG/task1_task2_taskNRv2-BrainTranslator.txt',
            'V2': 'score_results/optimal_v2_cohort/EEG/task1_task2_taskNRv2-BrainTranslatorV2.txt',
            'V3 (Contrastive)': 'score_results/optimal_v3_cohort/EEG/task1_task2_taskNRv2-BrainTranslatorV2.txt',
            'V3 Sweep Best': 'score_results/sweep_v3_best/EEG/task1_task2_taskNRv2-BrainTranslatorV2.txt',
            'Baseline+Contr λ=0.2': 'score_results/baseline_contrastive/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline.txt',
            'Baseline+Contr λ=1.0': 'score_results/baseline_contrastive_strong/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline.txt',
            'Baseline+Contr λ=1.0 (Fixed)': 'score_results/baseline_contrastive_fixed/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline.txt',
            'V4 (Baseline+TwoStep+Contr)': 'score_results/v4_baseline_twostep_contrastive_eeg/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline.txt',
            'V5 Best (Baseline+OneStep+Sweep)': 'score_results/sweep/ihig0y4v/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline.txt',
        },
        'Noise': {
            'Our Baseline': 'score_results/optimal_baseline_cohort/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise.txt',
            'Learned PE': 'score_results/optimal_fixed_cohort/noise/task1_task2_taskNRv2-BrainTranslator-EEG_noise.txt',
            'V2': 'score_results/optimal_v2_cohort/noise/task1_task2_taskNRv2-BrainTranslatorV2-EEG_noise.txt',
            'V3 (Contrastive)': 'score_results/optimal_v3_cohort/noise/task1_task2_taskNRv2-BrainTranslatorV2-EEG_noise.txt',
            'V3 Sweep Best': 'score_results/sweep_v3_best/noise/task1_task2_taskNRv2-BrainTranslatorV2-EEG_noise.txt',
            'Baseline+Contr λ=0.2': 'score_results/baseline_contrastive/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise.txt',
            'Baseline+Contr λ=1.0': 'score_results/baseline_contrastive_strong/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise.txt',
            'Baseline+Contr λ=1.0 (Fixed)': 'score_results/baseline_contrastive_fixed/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise.txt',
            'V4 (Baseline+TwoStep+Contr)': 'score_results/v4_baseline_twostep_contrastive_eeg/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise.txt',
            'V5 Best (Baseline+OneStep+Sweep)': 'score_results/sweep/ihig0y4v/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise.txt',
        }
    }

    # BART Metrics (Sourced from the arXiv paper 'Are EEG-to-Text Models Working?')
    # Values as floats for comparison logic
    bart_eeg = {
        'BLEU-1': 11.58, 'BLEU-2': 3.40, 'BLEU-3': 1.33, 'BLEU-4': 0.54,
        'SacreBLEU': None,
        'ROUGE-1-P': 11.33, 'ROUGE-1-R': 15.44, 'ROUGE-1-F': 12.40,
        'ROUGE-2-P': None, 'ROUGE-2-R': None, 'ROUGE-2-F': None,
        'ROUGE-L-P': None, 'ROUGE-L-R': None, 'ROUGE-L-F': None,
        'WER': 99.68
    }
    bart_noise = {
        'BLEU-1': 11.26, 'BLEU-2': 3.19, 'BLEU-3': 1.28, 'BLEU-4': 0.57,
        'SacreBLEU': None,
        'ROUGE-1-P': 11.07, 'ROUGE-1-R': 15.08, 'ROUGE-1-F': 12.06,
        'ROUGE-2-P': None, 'ROUGE-2-R': None, 'ROUGE-2-F': None,
        'ROUGE-L-P': None, 'ROUGE-L-R': None, 'ROUGE-L-F': None,
        'WER': 100.03
    }

    # Order of metrics in the table
    metric_keys = [
        'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'SacreBLEU',
        'ROUGE-1-P', 'ROUGE-1-R', 'ROUGE-1-F',
        'ROUGE-2-P', 'ROUGE-2-R', 'ROUGE-2-F',
        'ROUGE-L-P', 'ROUGE-L-R', 'ROUGE-L-F',
        'WER'
    ]

    for section_name, section_files in files.items():
        print(f"## {section_name} Input Performance")

        # 1. Collect all rows for this section
        all_rows = []

        # Add BART as first row
        bart_results = bart_eeg if section_name == 'EEG' else bart_noise
        all_rows.append(("Original Article (BART)", bart_results))

        # Parse all result files
        for model_name, path in section_files.items():
            all_rows.append((model_name, parse_metrics(path)))

        # 2. Determine best value in each column for highlighting
        best_vals = {}
        if args.highlight_best:
            for k in metric_keys:
                # Filter out 'NR' or None values for numerical comparison
                numerical_vals = [row[1][k] for row in all_rows if isinstance(row[1][k], (int, float))]
                if numerical_vals:
                    if k == 'WER':
                        # Lower is better for WER
                        best_vals[k] = min(numerical_vals)
                    else:
                        # Higher is better for BLEU/ROUGE
                        best_vals[k] = max(numerical_vals)

        # 3. Print Table Header
        header_cols = ["Model"] + metric_keys
        print("| " + " | ".join(header_cols) + " |")
        print("| " + " | ".join([":---"] + [":---:"] * (len(header_cols) - 1)) + " |")

        # 4. Print Data Rows
        for model_name, row_metrics in all_rows:
            formatted_cells = [f"**{model_name}**"]
            for k in metric_keys:
                val = row_metrics[k]
                is_best = False
                if args.highlight_best and isinstance(val, (int, float)) and k in best_vals:
                    # Use a small epsilon for float comparison
                    if abs(val - best_vals[k]) < 1e-6:
                        is_best = True
                formatted_cells.append(format_val(val, bold=is_best))
            print("| " + " | ".join(formatted_cells) + " |")

        print("\n")

if __name__ == "__main__":
    main()
