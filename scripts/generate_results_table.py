import re
import sys
import os

def parse_metrics(filepath):
    """Parses the metrics from the result file."""
    metrics = {
        'BLEU-1': 'NR', 'BLEU-2': 'NR', 'BLEU-3': 'NR', 'BLEU-4': 'NR',
        'ROUGE-1 (F)': 'NR', 'ROUGE-1 (P)': 'NR', 'ROUGE-1 (R)': 'NR',
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
            # corpus_bleu_score = [0.157141716194837, 0.04715314572446555, ...]
            bleu_match = re.search(r"corpus_bleu_score = \[([\d\., e\-\+]+)\]", text)
            if bleu_match:
                bleus = [float(x.strip()) for x in bleu_match.group(1).split(',')]
                if len(bleus) >= 4:
                    metrics['BLEU-1'] = f"{bleus[0]*100:.2f}"
                    metrics['BLEU-2'] = f"{bleus[1]*100:.2f}"
                    metrics['BLEU-3'] = f"{bleus[2]*100:.2f}"
                    metrics['BLEU-4'] = f"{bleus[3]*100:.2f}"

            # Parse ROUGE scores
            # rouge_scores = {'rouge-1': {'f': 0.1039...
            # Use regex to robustly extract ROUGE scores, handling potential variations in quoting (single vs double)
            # or dict string formatting.
            r1_f = re.search(r"'rouge-1': \{.*?'f': ([\d\.]+)", text)
            r1_p = re.search(r"'rouge-1': \{.*?'p': ([\d\.]+)", text)
            r1_r = re.search(r"'rouge-1': \{.*?'r': ([\d\.]+)", text)
            
            if r1_f: metrics['ROUGE-1 (F)'] = f"{float(r1_f.group(1))*100:.2f}"
            if r1_p: metrics['ROUGE-1 (P)'] = f"{float(r1_p.group(1))*100:.2f}"
            if r1_r: metrics['ROUGE-1 (R)'] = f"{float(r1_r.group(1))*100:.2f}"

            # Parse WER
            # wer_scores = 1.068...
            wer = re.search(r"wer_scores = ([\d\.]+)", text)
            if wer:
                metrics['WER'] = f"{float(wer.group(1))*100:.2f}"

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")

    return metrics

def generate_row(model_name, metrics):
    return f"| **{model_name}** | {metrics['BLEU-1']} | {metrics['BLEU-2']} | {metrics['BLEU-3']} | {metrics['BLEU-4']} | {metrics['ROUGE-1 (P)']} | {metrics['ROUGE-1 (R)']} | {metrics['ROUGE-1 (F)']} | {metrics['WER']} |"

def main():
    # Define file paths
    # Note: Using the specific files identified in conversation
    files = {
        'Our Baseline (EEG)': 'score_results/optimal_baseline_cohort/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline.txt',
        'Our Baseline (Noise)': 'score_results/optimal_baseline_cohort/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise.txt',
        'Our Fixed Model (EEG)': 'score_results/optimal_fixed_cohort/EEG/task1_task2_taskNRv2-BrainTranslator.txt',
        'Our Fixed Model (Noise)': 'score_results/optimal_fixed_cohort/noise/task1_task2_taskNRv2-BrainTranslator-EEG_noise.txt'
    }
    
    # BART Metrics (Sourced from the arXiv paper 'Are EEG-to-Text Models Working?')
    bart_eeg = {
        'BLEU-1': '11.58', 'BLEU-2': '3.40', 'BLEU-3': '1.33', 'BLEU-4': '0.54',
        'ROUGE-1 (P)': '11.33', 'ROUGE-1 (R)': '15.44', 'ROUGE-1 (F)': '12.40',
        'WER': '99.68'
    }
    bart_noise = {
        'BLEU-1': '11.26', 'BLEU-2': '3.19', 'BLEU-3': '1.28', 'BLEU-4': '0.57',
        'ROUGE-1 (P)': '11.07', 'ROUGE-1 (R)': '15.08', 'ROUGE-1 (F)': '12.06',
        'WER': '100.03'
    }

    # Header
    header = "| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 (P) | ROUGE-1 (R) | ROUGE-1 (F) | WER |\n"
    divider = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"

    print("## 1. EEG Input Performance")
    print(header + divider)
    print(generate_row("Original Article (BART)", bart_eeg))
    print(generate_row("Our Baseline", parse_metrics(files['Our Baseline (EEG)'])))
    print(generate_row("Our Fixed Model", parse_metrics(files['Our Fixed Model (EEG)'])))
    print("\n")

    print("## 2. Random Noise Input Performance")
    print(header + divider)
    print(generate_row("Original Article (BART)", bart_noise))
    print(generate_row("Our Baseline", parse_metrics(files['Our Baseline (Noise)'])))
    print(generate_row("Our Fixed Model", parse_metrics(files['Our Fixed Model (Noise)'])))

if __name__ == "__main__":
    main()
