import re
import sys
import os

def parse_metrics(filepath):
    """Parses the metrics from the result file."""
    metrics = {
        'BLEU-1': 'NR', 'BLEU-2': 'NR', 'BLEU-3': 'NR', 'BLEU-4': 'NR',
        'SacreBLEU': 'NR',
        'ROUGE-1 (F)': 'NR', 'ROUGE-2 (F)': 'NR', 'ROUGE-L (F)': 'NR',
        'WER': 'NR'
    }

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return metrics

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
                    metrics['BLEU-1'] = f"{bleus[0]*100:.2f}"
                    metrics['BLEU-2'] = f"{bleus[1]*100:.2f}"
                    metrics['BLEU-3'] = f"{bleus[2]*100:.2f}"
                    metrics['BLEU-4'] = f"{bleus[3]*100:.2f}"

            # Parse SacreBLEU score
            sacre_match = re.search(r"sacre_blue_score = \{.*?'score': ([\d\.]+)", text)
            if sacre_match:
                metrics['SacreBLEU'] = f"{float(sacre_match.group(1)):.2f}"

            # Parse ROUGE scores
            r1_f = re.search(r"'rouge-1': \{.*?'f': ([\d\.]+)", text)
            r2_f = re.search(r"'rouge-2': \{.*?'f': ([\d\.]+)", text)
            rl_f = re.search(r"'rouge-l': \{.*?'f': ([\d\.]+)", text)

            if r1_f: metrics['ROUGE-1 (F)'] = f"{float(r1_f.group(1))*100:.2f}"
            if r2_f: metrics['ROUGE-2 (F)'] = f"{float(r2_f.group(1))*100:.2f}"
            if rl_f: metrics['ROUGE-L (F)'] = f"{float(rl_f.group(1))*100:.2f}"

            # Parse WER
            wer = re.search(r"wer_scores = ([\d\.]+)", text)
            if wer:
                metrics['WER'] = f"{float(wer.group(1))*100:.2f}"

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")

    return metrics

def generate_row(model_name, metrics):
    return f"| **{model_name}** | {metrics['BLEU-1']} | {metrics['BLEU-2']} | {metrics['BLEU-3']} | {metrics['BLEU-4']} | {metrics['SacreBLEU']} | {metrics['ROUGE-1 (F)']} | {metrics['ROUGE-2 (F)']} | {metrics['ROUGE-L (F)']} | {metrics['WER']} |"

def main():
    # Define file paths
    files = {
        'Our Baseline (EEG)': 'score_results/optimal_baseline_cohort/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline.txt',
        'Our Baseline (Noise)': 'score_results/optimal_baseline_cohort/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise.txt',
        'Learned PE (EEG)': 'score_results/optimal_fixed_cohort/EEG/task1_task2_taskNRv2-BrainTranslator.txt',
        'Learned PE (Noise)': 'score_results/optimal_fixed_cohort/noise/task1_task2_taskNRv2-BrainTranslator-EEG_noise.txt',
        'V2 (EEG)': 'score_results/optimal_v2_cohort/EEG/task1_task2_taskNRv2-BrainTranslatorV2.txt',
        'V2 (Noise)': 'score_results/optimal_v2_cohort/noise/task1_task2_taskNRv2-BrainTranslatorV2-EEG_noise.txt',
        'V3 (EEG)': 'score_results/optimal_v3_cohort/EEG/task1_task2_taskNRv2-BrainTranslatorV2.txt',
        'V3 (Noise)': 'score_results/optimal_v3_cohort/noise/task1_task2_taskNRv2-BrainTranslatorV2-EEG_noise.txt',
        'V3 Sweep Best (EEG)': 'score_results/sweep_v3_best/EEG/task1_task2_taskNRv2-BrainTranslatorV2.txt',
        'V3 Sweep Best (Noise)': 'score_results/sweep_v3_best/noise/task1_task2_taskNRv2-BrainTranslatorV2-EEG_noise.txt',
        'Baseline+Contr λ=0.2 (EEG)': 'score_results/baseline_contrastive/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline.txt',
        'Baseline+Contr λ=0.2 (Noise)': 'score_results/baseline_contrastive/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise.txt',
        'Baseline+Contr λ=1.0 (EEG)': 'score_results/baseline_contrastive_strong/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline.txt',
        'Baseline+Contr λ=1.0 (Noise)': 'score_results/baseline_contrastive_strong/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise.txt',
        'Baseline+Contr λ=1.0 Fixed (EEG)': 'score_results/baseline_contrastive_fixed/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline.txt',
        'Baseline+Contr λ=1.0 Fixed (Noise)': 'score_results/baseline_contrastive_fixed/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise.txt',
        'V4 (EEG)': 'score_results/v4_baseline_twostep_contrastive_eeg/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline.txt',
        'V4 (Noise)': 'score_results/v4_baseline_twostep_contrastive_eeg/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise.txt',
        'V5 Sweep Best (EEG)': 'score_results/sweep/ihig0y4v/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline.txt',
        'V5 Sweep Best (Noise)': 'score_results/sweep/ihig0y4v/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise.txt',
    }

    # BART Metrics (Sourced from the arXiv paper 'Are EEG-to-Text Models Working?')
    bart_eeg = {
        'BLEU-1': '11.58', 'BLEU-2': '3.40', 'BLEU-3': '1.33', 'BLEU-4': '0.54',
        'SacreBLEU': 'N/A',
        'ROUGE-1 (F)': '12.40', 'ROUGE-2 (F)': 'N/A', 'ROUGE-L (F)': 'N/A',
        'WER': '99.68'
    }
    bart_noise = {
        'BLEU-1': '11.26', 'BLEU-2': '3.19', 'BLEU-3': '1.28', 'BLEU-4': '0.57',
        'SacreBLEU': 'N/A',
        'ROUGE-1 (F)': '12.06', 'ROUGE-2 (F)': 'N/A', 'ROUGE-L (F)': 'N/A',
        'WER': '100.03'
    }

    # Header
    header = "| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | SacreBLEU | ROUGE-1 (F) | ROUGE-2 (F) | ROUGE-L (F) | WER |\n"
    divider = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"

    print("## 1. EEG Input Performance")
    print(header + divider)
    print(generate_row("Original Article (BART)", bart_eeg))
    print(generate_row("Our Baseline", parse_metrics(files['Our Baseline (EEG)'])))
    print(generate_row("Learned PE", parse_metrics(files['Learned PE (EEG)'])))
    print(generate_row("V2", parse_metrics(files['V2 (EEG)'])))
    print(generate_row("V3 (Contrastive)", parse_metrics(files['V3 (EEG)'])))
    print(generate_row("V3 Sweep Best", parse_metrics(files['V3 Sweep Best (EEG)'])))
    print(generate_row("Baseline+Contr λ=0.2", parse_metrics(files['Baseline+Contr λ=0.2 (EEG)'])))
    print(generate_row("Baseline+Contr λ=1.0", parse_metrics(files['Baseline+Contr λ=1.0 (EEG)'])))
    print(generate_row("Baseline+Contr λ=1.0 (Fixed)", parse_metrics(files['Baseline+Contr λ=1.0 Fixed (EEG)'])))
    print(generate_row("V4 (Baseline+TwoStep+Contr)", parse_metrics(files['V4 (EEG)'])))
    print(generate_row("V5 Best (Baseline+OneStep+Sweep)", parse_metrics(files['V5 Sweep Best (EEG)'])))
    print("\n")

    print("## 2. Random Noise Input Performance")
    print(header + divider)
    print(generate_row("Original Article (BART)", bart_noise))
    print(generate_row("Our Baseline", parse_metrics(files['Our Baseline (Noise)'])))
    print(generate_row("Learned PE", parse_metrics(files['Learned PE (Noise)'])))
    print(generate_row("V2", parse_metrics(files['V2 (Noise)'])))
    print(generate_row("V3 (Contrastive)", parse_metrics(files['V3 (Noise)'])))
    print(generate_row("V3 Sweep Best", parse_metrics(files['V3 Sweep Best (Noise)'])))
    print(generate_row("Baseline+Contr λ=0.2", parse_metrics(files['Baseline+Contr λ=0.2 (Noise)'])))
    print(generate_row("Baseline+Contr λ=1.0", parse_metrics(files['Baseline+Contr λ=1.0 (Noise)'])))
    print(generate_row("Baseline+Contr λ=1.0 (Fixed)", parse_metrics(files['Baseline+Contr λ=1.0 Fixed (Noise)'])))
    print(generate_row("V4 (Baseline+TwoStep+Contr)", parse_metrics(files['V4 (Noise)'])))
    print(generate_row("V5 Best (Baseline+OneStep+Sweep)", parse_metrics(files['V5 Sweep Best (Noise)'])))

if __name__ == "__main__":
    main()

