"""Confidence intervals and significance tests for the EEG-to-text results.

Covers every results table: free generation and teacher forcing, real EEG and
injected noise, the three cohort models and the single-subject (ZDM) model.

Three interval/test methods:

* **Wald proportion interval** (Ruben, *ExperimentsAndResults.pdf* sec. 4.2.2):
  ``CI = p +- 1.96 * sqrt(p (1 - p) / n)``.  Reported only for reference -- it is
  over-confident for n-gram metrics (sliding-window n-grams are not independent
  samples) and undefined for WER/CER above 100%.
* **Paired bootstrap over sentences** (Koehn, EMNLP 2004): resample whole
  sentences and recompute the corpus metric.  Gives the cell CIs in the tables.
* **Paired approximate randomisation** (Riezler & Maxwell, 2005): the stricter
  significance test, used for the difference tests (Section 5.3 table).

The pipeline runs in three stages as *separate processes*, because this project's
pinned conda stack segfaults non-deterministically when the heavy tokenisation and
scoring libraries are co-resident; each stage imports only what it needs:

    python confidence_intervals.py dump-bleu   # BART tokenizer -> BLEU counts
    python confidence_intervals.py dump-text   # rouge + jiwer  -> ROUGE/WER/CER stats
    python confidence_intervals.py dump-sacre  # sacrebleu      -> sacreBLEU stats
    python confidence_intervals.py analyse     # numpy only     -> CIs + significance
    python confidence_intervals.py latex       # numpy only     -> table rows / sig. table

Stat keys are ``model|condition|regime`` with model in {literature, vocab,
contrastive, zdm}, condition in {EEG, noise}, regime in {free, tf}.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent
BLEU_STATS = DOCS / "ci_stats.json"
TEXT_STATS = DOCS / "ci_stats2.json"
SACRE_STATS = DOCS / "ci_stats3.json"

# Prediction files per model and input condition. Teacher-forcing outputs live in
# the same files, so each file yields both the "free" and "tf" regimes.
MODELS: dict[str, dict[str, str]] = {
    "literature": {
        "EEG": "results/sweep_jo_baseline_btb_eval/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline-all_decoding_results.txt",
        "noise": "results/sweep_jo_baseline_btb_eval/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise-all_decoding_results.txt",
    },
    "vocab": {
        "EEG": "results/sweep_baseline_nocontrastive_eval/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline-all_decoding_results.txt",
        "noise": "results/sweep_baseline_nocontrastive_eval/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise-all_decoding_results.txt",
    },
    "contrastive": {
        "EEG": "results/sweep/ihig0y4v/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline-all_decoding_results.txt",
        "noise": "results/sweep/ihig0y4v/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise-all_decoding_results.txt",
    },
    "zdm": {
        "EEG": "results/single_subject_ZDM/ZDM/EEG/task1_task2_taskNRv2-BrainTranslatorBaseline-all_decoding_results.txt",
        "noise": "results/single_subject_ZDM/ZDM/noise/task1_task2_taskNRv2-BrainTranslatorBaseline-EEG_noise-all_decoding_results.txt",
    },
}
CONDITIONS = ("EEG", "noise")
REGIMES = ("free", "tf")
METRICS = [
    "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "sacreBLEU",
    "ROUGE-1", "ROUGE-2", "ROUGE-L", "WER", "CER",
]  # fmt: skip

# Published table values, for sanity-checking the recomputation (cells the paper
# actually prints; tf-noise is computed but never displayed, so it is omitted).
PAPER = {
    "literature|EEG|free": [14.52, 4.10, 1.76, 0.91, 1.03, 12.64, 1.42, 11.28, 102.54, 80.51],
    "vocab|EEG|free": [16.74, 5.87, 3.68, 2.90, 2.97, 11.85, 1.80, 10.54, 110.55, 88.11],
    "contrastive|EEG|free": [18.04, 7.44, 5.21, 4.36, 4.46, 12.48, 2.33, 11.07, 105.94, 85.08],
    "literature|noise|free": [16.18, 4.53, 1.87, 0.95, 1.08, 13.22, 1.43, 11.68, 107.46, 83.33],
    "vocab|noise|free": [11.39, 1.95, 0.66, 0.37, 0.37, 8.40, 0.30, 7.11, 133.79, 113.49],
    "contrastive|noise|free": [11.80, 1.53, 0.54, 0.32, 0.36, 8.88, 0.20, 7.74, 126.05, 101.31],
    "literature|EEG|tf": [41.95, 25.02, 14.80, 9.12, 8.52, 31.97, 9.21, 30.22, 75.10, 59.18],
    "vocab|EEG|tf": [93.29, 92.91, 92.68, 92.50, 91.73, 91.32, 89.68, 91.29, 8.52, 5.75],
    "contrastive|EEG|tf": [93.36, 93.02, 92.83, 92.69, 91.90, 91.41, 89.73, 91.41, 8.41, 5.66],
    "zdm|EEG|free": [15.38, 5.13, 2.51, 1.35, 1.53, 11.53, 1.68, 10.48, 115.36, 92.06],
    "zdm|EEG|tf": [43.56, 26.15, 15.07, 8.27, 7.38, 32.61, 9.04, 30.31, 73.93, 59.02],
    "zdm|noise|free": [5.98, 1.50, 0.00, 0.00, 0.23, 7.93, 1.05, 7.91, 96.43, 81.63],
}

Z_95 = 1.96


# --------------------------------------------------------------------------- #
# Shared parsing
# --------------------------------------------------------------------------- #


def parse_predictions(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (references, free-generation hypotheses, teacher-forcing hypotheses).

    Each record is::

        target string: <reference>
        predicted string with tf: <teacher-forcing output>
        predicted string: <free-generation output>
        ##############################
    """
    refs, free, tf = [], [], []
    record: dict[str, str] = {}
    for line in path.open(encoding="utf-8", errors="replace"):
        if line.startswith("target string:"):
            record["ref"] = line.split("target string:", 1)[1].strip()
        elif line.startswith("predicted string with tf:"):
            record["tf"] = line.split("predicted string with tf:", 1)[1].strip()
        elif line.startswith("predicted string:"):
            record["free"] = line.split("predicted string:", 1)[1].strip()
        elif line.startswith("####"):
            if {"ref", "free", "tf"} <= record.keys():
                refs.append(record["ref"])
                free.append(record["free"])
                tf.append(record["tf"])
            record = {}
    if {"ref", "free", "tf"} <= record.keys():
        refs.append(record["ref"])
        free.append(record["free"])
        tf.append(record["tf"])
    return refs, free, tf


def _iter_files():
    """Yield (key, references, hypotheses) for every model|condition|regime."""
    for model, paths in MODELS.items():
        for cond in CONDITIONS:
            refs, free, tf = parse_predictions(REPO / paths[cond])
            for regime, hyps in (("free", free), ("tf", tf)):
                yield f"{model}|{cond}|{regime}", refs, hyps


# --------------------------------------------------------------------------- #
# Stage 1: BLEU n-gram counts (needs the BART tokenizer)
# --------------------------------------------------------------------------- #


def dump_bleu_stats() -> None:
    """Dump per-sentence clipped/total n-gram counts for corpus BLEU.

    BLEU was scored by ``nltk.corpus_bleu`` over BART sub-word tokens, so counting
    n-grams over those tokens reproduces the paper's corpus BLEU exactly.

    Free generation re-tokenises the saved decoded string (the eval re-encodes its
    own free-gen output, so this matches). Teacher forcing instead reads the exact
    argmax token lists dumped by ``eval_tf_dump.py``: the eval scored TF BLEU on the
    raw predicted token IDs, which a string round-trip cannot reproduce (it drifts
    ~1.3 BLEU low), so the saved strings are not usable for TF BLEU.
    """
    from transformers import BartTokenizer

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    def tokens(text: str) -> list[str]:
        return tokenizer.convert_ids_to_tokens(tokenizer.encode(text), skip_special_tokens=True)

    def ngrams(seq: list[str], n: int) -> Counter:
        return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))

    def bleu_row(cand: list[str], gold: list[str]) -> dict[str, int]:
        row = {"clen": len(cand), "rlen": len(gold)}
        for n in range(1, 5):
            cand_ng, gold_ng = ngrams(cand, n), ngrams(gold, n)
            row[f"clip{n}"] = sum(min(c, gold_ng.get(g, 0)) for g, c in cand_ng.items())
            row[f"tot{n}"] = sum(cand_ng.values())
        return row

    out: dict[str, list[dict[str, int]]] = {}
    for model, paths in MODELS.items():
        for cond in CONDITIONS:
            refs, free, _ = parse_predictions(REPO / paths[cond])

            key = f"{model}|{cond}|free"
            out[key] = [bleu_row(tokens(h), tokens(r)) for r, h in zip(refs, free)]
            print(f"dump-bleu: {key} -> {len(out[key])} sentences", flush=True)

            key = f"{model}|{cond}|tf"
            tf_path = (REPO / paths[cond]).parent / "tf_tokens.json"
            tf_recs = json.loads(tf_path.read_text())
            assert len(tf_recs) == len(refs), f"{tf_path}: {len(tf_recs)} != {len(refs)}"
            out[key] = [bleu_row(rec["tf_tokens"], rec["target_tokens"]) for rec in tf_recs]
            print(f"dump-bleu: {key} -> {len(out[key])} sentences", flush=True)
    BLEU_STATS.write_text(json.dumps(out))
    print(f"dump-bleu: wrote {BLEU_STATS}", flush=True)


# --------------------------------------------------------------------------- #
# Stage 2: ROUGE (rouge lib) + WER/CER (jiwer) -- no BART tokenizer
# --------------------------------------------------------------------------- #


def dump_text_stats() -> None:
    """Dump per-sentence ROUGE-1/2/L F (rouge lib) and WER/CER edit counts (jiwer).

    Both libraries match the evaluation's; summing jiwer's edits/lengths gives the
    corpus (micro) WER/CER and averaging the rouge F values gives the corpus ROUGE.
    """
    import jiwer
    from rouge import Rouge

    rouge = Rouge()

    def edit_counts(ref: str, hyp: str, *, characters: bool) -> tuple[int, int]:
        if not hyp.strip():  # empty hypothesis -> all reference tokens are deletions
            length = len(ref) if characters else len(ref.split())
            return length, length
        proc = jiwer.process_characters if characters else jiwer.process_words
        result = proc(ref, hyp)
        return (
            result.substitutions + result.deletions + result.insertions,
            result.substitutions + result.deletions + result.hits,
        )

    out: dict[str, list[dict[str, float]]] = {}
    for key, refs, hyps in _iter_files():
        rows = []
        for ref, hyp in zip(refs, hyps):
            cand = hyp.split()
            scores = rouge.get_scores([hyp if hyp.strip() else "."], [ref])[0]
            word_edits, word_ref = edit_counts(ref, hyp, characters=False)
            char_edits, char_ref = edit_counts(ref, hyp, characters=True)
            rows.append(
                {
                    "r1f": scores["rouge-1"]["f"],
                    "r2f": scores["rouge-2"]["f"],
                    "rlf": scores["rouge-l"]["f"],
                    "r1_c": len(cand),  # token / n-gram counts, for the Wald sample size
                    "r2_c": max(0, len(cand) - 1),
                    "rl_c": len(cand),
                    "w_edits": word_edits,
                    "w_ref": word_ref,
                    "c_edits": char_edits,
                    "c_ref": char_ref,
                }
            )
        out[key] = rows
        print(f"dump-text: {key} -> {len(rows)} sentences", flush=True)
    TEXT_STATS.write_text(json.dumps(out))
    print(f"dump-text: wrote {TEXT_STATS}", flush=True)


# --------------------------------------------------------------------------- #
# Stage 2b: sacreBLEU sufficient statistics (sacrebleu's own tokenizer)
# --------------------------------------------------------------------------- #


def dump_sacre_stats() -> None:
    """Dump per-sentence sacreBLEU sufficient statistics.

    sacreBLEU is scored on the decoded strings (sacrebleu's 13a tokeniser, not the
    BART sub-words), and the teacher-forcing strings in the saved files reproduce
    exactly, so both regimes are computed straight from the saved text. Each record
    is ``[sys_len, ref_len, correct_1..4, total_1..4]``; summing those over a
    sentence resample and applying NIST 'exp' smoothing (see _sacrebleu_from_stats)
    reproduces corpus sacreBLEU, which lets the metric be bootstrapped honestly
    rather than approximated by BLEU-4.
    """
    from sacrebleu.metrics import BLEU

    bleu = BLEU()
    out: dict[str, list[list[int]]] = {}
    for key, refs, hyps in _iter_files():
        rows = [bleu._extract_corpus_statistics([h], [[r]])[0] for r, h in zip(refs, hyps)]
        out[key] = rows
        print(f"dump-sacre: {key} -> {len(rows)} sentences", flush=True)
    SACRE_STATS.write_text(json.dumps(out))
    print(f"dump-sacre: wrote {SACRE_STATS}", flush=True)


def _sacrebleu_from_stats(s) -> float:
    """Corpus sacreBLEU from summed sufficient stats, replicating sacrebleu's
    ``BLEU.compute_bleu`` with the default NIST 'exp' smoothing (max order 4)."""
    sys_len, ref_len = s[0], s[1]
    correct, total = s[2:6], s[6:10]
    if not any(c > 0 for c in correct):
        return 0.0
    bp = 1.0
    if sys_len < ref_len:
        bp = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0
    precisions = [0.0, 0.0, 0.0, 0.0]
    smooth = 1.0
    for n in range(4):
        if total[n] == 0:
            break
        if correct[n] == 0:
            smooth *= 2
            precisions[n] = 100.0 / (smooth * total[n])
        else:
            precisions[n] = 100.0 * correct[n] / total[n]
    logs = [math.log(p) if p > 0 else -9999999999.0 for p in precisions]
    return bp * math.exp(sum(logs) / 4)


# --------------------------------------------------------------------------- #
# Stage 3: confidence intervals + significance (numpy only)
# --------------------------------------------------------------------------- #


def _load_arrays():
    """Load both stat files into per-key dicts of numpy arrays."""
    import numpy as np

    if not (BLEU_STATS.exists() and TEXT_STATS.exists() and SACRE_STATS.exists()):
        raise SystemExit("Missing stats; run `dump-bleu`, `dump-text` and `dump-sacre` first.")
    bleu = json.loads(BLEU_STATS.read_text())
    text = json.loads(TEXT_STATS.read_text())
    sacre = json.loads(SACRE_STATS.read_text())

    def col(rows, k):
        return np.array([r[k] for r in rows], dtype=float)

    data = {}
    for key in bleu:
        a, b = bleu[key], text[key]
        data[key] = {
            "sacre": np.array(sacre[key], dtype=float),
            "clip": np.array([[r[f"clip{k}"] for k in range(1, 5)] for r in a], dtype=float),
            "tot": np.array([[r[f"tot{k}"] for k in range(1, 5)] for r in a], dtype=float),
            "clen": col(a, "clen"),
            "rlen": col(a, "rlen"),
            "r1": col(b, "r1f"),
            "r2": col(b, "r2f"),
            "rl": col(b, "rlf"),
            "we": col(b, "w_edits"),
            "wr": col(b, "w_ref"),
            "ce": col(b, "c_edits"),
            "cr": col(b, "c_ref"),
            "n_r1": col(b, "r1_c"),
            "n_r2": col(b, "r2_c"),
            "n_rl": col(b, "rl_c"),
        }
    return data


def _score_vector(arr: dict) -> list[float]:
    """The 10 metrics from a set of per-sentence arrays.

    BLEU is corpus ratios over BART sub-words; sacreBLEU is corpus BLEU over
    sacrebleu's own tokeniser (from its sufficient stats); ROUGE is the mean of
    per-sentence F; WER/CER are micro-averaged.
    """
    import numpy as np

    clip = arr["clip"].sum(0)
    tot = arr["tot"].sum(0)
    cand_len, ref_len = arr["clen"].sum(), arr["rlen"].sum()
    brevity = 1.0 if cand_len > ref_len else (math.exp(1 - ref_len / cand_len) if cand_len else 0.0)
    bleu = []
    for k in range(1, 5):
        if (tot[:k] == 0).any() or (clip[:k] == 0).any():
            bleu.append(0.0)
        else:
            bleu.append(brevity * float(np.prod(clip[:k] / tot[:k]) ** (1 / k)) * 100)
    return bleu + [
        _sacrebleu_from_stats(arr["sacre"].sum(0)),
        arr["r1"].mean() * 100,
        arr["r2"].mean() * 100,
        arr["rl"].mean() * 100,
        arr["we"].sum() / arr["wr"].sum() * 100,
        arr["ce"].sum() / arr["cr"].sum() * 100,
    ]


def _wald_halfwidth(value_pct: float, n: float) -> float | None:
    p = value_pct / 100
    if not 0 <= p <= 1 or n <= 0:
        return None
    return Z_95 * math.sqrt(p * (1 - p) / n) * 100


def _bootstrap_ci(arr, n_boot, rng):
    """95% percentile bootstrap CI per metric, resampling sentences."""
    import numpy as np

    n = arr["clen"].size
    samples = np.array(
        [
            _score_vector({k: v[idx] for k, v in arr.items()})
            for idx in (rng.integers(0, n, n) for _ in range(n_boot))
        ]
    )
    return np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


# Difference tests for Section 5.3 (label, system A, system B).
COMPARISONS = [
    ("Contrastive - Vocab (free gen)", "contrastive|EEG|free", "vocab|EEG|free"),
    ("Contrastive - Vocab (teacher forcing)", "contrastive|EEG|tf", "vocab|EEG|tf"),
    ("Real - Noise: Literature (free gen)", "literature|EEG|free", "literature|noise|free"),
    ("Real - Noise: Vocab (free gen)", "vocab|EEG|free", "vocab|noise|free"),
    ("Real - Noise: Contrastive (free gen)", "contrastive|EEG|free", "contrastive|noise|free"),
    ("Real - Noise: ZDM single-subject (free gen)", "zdm|EEG|free", "zdm|noise|free"),
    ("Real - Noise: Contrastive (teacher forcing)", "contrastive|EEG|tf", "contrastive|noise|tf"),
    ("Real - Noise: ZDM single-subject (teacher forcing)", "zdm|EEG|tf", "zdm|noise|tf"),
]


def _paired_tests(a, b, n_boot, n_ar, rng):
    """Paired bootstrap CI of the difference and approximate-randomisation p-value."""
    import numpy as np

    n = a["clen"].size
    observed = np.array(_score_vector(a)) - np.array(_score_vector(b))
    diffs = np.array(
        [
            np.array(_score_vector({k: v[idx] for k, v in a.items()}))
            - np.array(_score_vector({k: v[idx] for k, v in b.items()}))
            for idx in (rng.integers(0, n, n) for _ in range(n_boot))
        ]
    )
    lo, hi = np.percentile(diffs, 2.5, axis=0), np.percentile(diffs, 97.5, axis=0)
    exceed = np.zeros(len(METRICS))
    for _ in range(n_ar):
        mask = rng.integers(0, 2, n).astype(bool)
        a_sw = {k: np.where(mask[:, None] if v.ndim == 2 else mask, b[k], v) for k, v in a.items()}
        b_sw = {k: np.where(mask[:, None] if v.ndim == 2 else mask, v, b[k]) for k, v in a.items()}
        d = np.array(_score_vector(a_sw)) - np.array(_score_vector(b_sw))
        exceed += np.abs(d) >= np.abs(observed) - 1e-9
    return observed, lo, hi, (exceed + 1) / (n_ar + 1)


def analyse(n_boot: int, n_ar: int, seed: int) -> None:
    import numpy as np

    rng = np.random.default_rng(seed)
    data = _load_arrays()

    print("Recomputation vs. paper (should match within rounding):")
    for key in PAPER:
        got, want = _score_vector(data[key]), PAPER[key]
        flags = [
            f"{m}={got[i]:.2f}/{want[i]}"
            for i, m in zip((0, 3, 4, 5, 7, 8, 9), ["B1", "B4", "sB", "R1", "RL", "WER", "CER"])
        ]
        print(f"  {key:24s} " + "  ".join(flags))

    print("\nCell 95% CIs  ::  value | Wald +- | bootstrap [lo, hi]")
    for key in PAPER:
        arr = data[key]
        tot = arr["tot"].sum(0)
        wald_n = [
            tot[0],
            tot[1],
            tot[2],
            tot[3],
            tot[3],
            arr["n_r1"].sum(),
            arr["n_r2"].sum(),
            arr["n_rl"].sum(),
            arr["wr"].sum(),
            arr["cr"].sum(),
        ]
        lo, hi = _bootstrap_ci(arr, n_boot, rng)
        print(f"\n{key}:")
        for i, name in enumerate(METRICS):
            half = _wald_halfwidth(PAPER[key][i], wald_n[i])
            wald = f"+-{half:.2f}" if half is not None else "n/a(>100%)"
            print(f"   {name:10s} {PAPER[key][i]:7.2f} | {wald:>10s} | [{lo[i]:.2f}, {hi[i]:.2f}]")

    print("\nDifference tests (Section 5.3)  ::  bootstrap CI + approximate-randomisation p")
    for label, a_key, b_key in COMPARISONS:
        observed, lo, hi, p = _paired_tests(data[a_key], data[b_key], n_boot, n_ar, rng)
        print(f"\n{label}:")
        for i, name in enumerate(METRICS):
            sig = "significant" if p[i] < 0.05 else "n.s."
            print(
                f"   {name:10s} d={observed[i]:+7.2f}  boot[{lo[i]:+7.2f}, {hi[i]:+7.2f}]  AR p={p[i]:.4f}  {sig}"
            )


# --------------------------------------------------------------------------- #
# Stage 4: LaTeX emission
# --------------------------------------------------------------------------- #

LABELS = {
    "literature": "Literature (BART)",
    "vocab": "Vocab-consistent",
    "contrastive": "Contrastive (Ours)",
    "zdm": "Single-subject (ZDM)",
}


def emit_latex(n_boot: int, seed: int) -> None:
    """Emit two-row cell rows (value over bootstrap CI) per table, and the Section 5.3
    significance table rows (Delta with an approximate-randomisation marker)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    data = _load_arrays()
    higher_is_better = {m for m in METRICS if m not in ("WER", "CER")}

    def cell_rows(keys, *, bold, labels=None):
        labels = labels or {k: LABELS[k.split("|")[0]] for k in keys}
        best = [
            (max if METRICS[i] in higher_is_better else min)(
                PAPER[k][i] for k in keys if k in PAPER
            )
            for i in range(len(METRICS))
        ]
        for key in keys:
            lo, hi = _bootstrap_ci(data[key], n_boot, rng)
            vals = PAPER[key]
            cells = []
            for i in range(len(METRICS)):
                top = (
                    f"\\textbf{{{vals[i]:.2f}}}"
                    if bold and abs(vals[i] - best[i]) < 1e-9
                    else f"{vals[i]:.2f}"
                )
                cells.append(f"\\makecell{{{top}\\\\{{\\tiny[{lo[i]:.2f},\\,{hi[i]:.2f}]}}}}")
            print(f"{labels[key]:20s} & " + " & ".join(cells) + " \\\\")

    cohort = ["literature", "vocab", "contrastive"]
    print("% ===== Table 3 (free generation, real EEG) =====")
    cell_rows([f"{m}|EEG|free" for m in cohort], bold=True)
    print("% ===== Table 4 (teacher forcing, real EEG) =====")
    cell_rows([f"{m}|EEG|tf" for m in cohort], bold=True)
    print("% ===== Table 5 real-EEG block (no bold) =====")
    cell_rows([f"{m}|EEG|free" for m in cohort], bold=False)
    print("% ===== Table 5 noise block (no bold) =====")
    cell_rows([f"{m}|noise|free" for m in cohort], bold=False)
    print("% ===== Table 6 (cohort contrastive vs ZDM) =====")
    for regime, cond in (("free", "EEG"), ("tf", "EEG"), ("free", "noise")):
        print(f"% -- {regime}/{cond} --")
        cell_rows(
            [f"contrastive|{cond}|{regime}", f"zdm|{cond}|{regime}"],
            bold=False,
            labels={
                f"contrastive|{cond}|{regime}": "Multi-subject (cohort)",
                f"zdm|{cond}|{regime}": "Single-subject (ZDM)",
            },
        )

    print("\n% ===== Table 7: Section 5.3 significance (Delta; bold = AR p<0.05) =====")
    print("% columns: BLEU 1/2/3/4, sB, ROUGE 1/2/L, WER, CER (same order as the other tables)")
    for label, a_key, b_key in COMPARISONS:
        observed, _, _, p = _paired_tests(data[a_key], data[b_key], n_boot, 10000, rng)
        cells = " & ".join(
            (f"{{\\boldmath${observed[i]:+.2f}$}}" if p[i] < 0.05 else f"${observed[i]:+.2f}$")
            for i in range(len(METRICS))
        )
        print(f"{label} & {cells} \\\\")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("dump-bleu", help="stage 1: BART-token BLEU counts")
    sub.add_parser("dump-text", help="stage 2: ROUGE / WER / CER stats")
    sub.add_parser("dump-sacre", help="stage 2b: sacreBLEU sufficient stats")
    for name in ("analyse", "latex"):
        p = sub.add_parser(name)
        p.add_argument("--n-boot", type=int, default=2000)
        p.add_argument("--seed", type=int, default=42)
        if name == "analyse":
            p.add_argument("--n-ar", type=int, default=10000)
    args = parser.parse_args()

    if args.stage == "dump-bleu":
        dump_bleu_stats()
    elif args.stage == "dump-text":
        dump_text_stats()
    elif args.stage == "dump-sacre":
        dump_sacre_stats()
    elif args.stage == "analyse":
        analyse(args.n_boot, args.n_ar, args.seed)
    elif args.stage == "latex":
        emit_latex(args.n_boot, args.seed)


if __name__ == "__main__":
    main()
