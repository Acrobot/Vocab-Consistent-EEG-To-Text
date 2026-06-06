"""Confidence intervals and significance tests for the EEG-to-text results.

This computes uncertainty for the free-generation numbers in Table 5 of the
paper (real EEG vs. injected noise, for the literature, vocabulary-consistent
and contrastive models).

Three interval/test methods are produced:

* **Wald proportion interval** (Ruben, *ExperimentsAndResults.pdf* sec. 4.2.2):
  ``CI = p +- 1.96 * sqrt(p (1 - p) / n)``.  Reported only for reference -- it is
  over-confident for n-gram metrics because the sliding-window n-grams are not
  independent samples, so ``n`` (the n-gram count) badly overstates the effective
  sample size.  It is also undefined for WER/CER when the rate exceeds 100%.
* **Paired bootstrap over sentences** (Koehn, EMNLP 2004): resample whole
  sentences with replacement and recompute the corpus metric.  The sentence is
  the independent unit, so this respects the n-gram dependence.
* **Paired approximate randomisation** (Riezler & Maxwell, 2005): the stricter,
  more conservative significance test (fewer type-I errors than the bootstrap).

The pipeline runs in three stages as *separate processes*:

    python confidence_intervals.py dump-bleu   # BART tokenizer -> BLEU / ROUGE-1 counts
    python confidence_intervals.py dump-text   # jiwer + stdlib -> ROUGE-2/L, WER, CER counts
    python confidence_intervals.py analyse     # numpy only     -> CIs + significance

The split is deliberate: this project's pinned conda stack (torch 1.9 /
transformers / nltk / sacrebleu / rouge) segfaults non-deterministically when the
heavy tokenisation and scoring libraries are co-resident, so each stage imports
only what it needs.  Stages 1-2 write per-sentence *sufficient statistics* to JSON;
stage 3 reads them and does every resample in pure numpy.
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
BLEU_STATS = DOCS / "ci_stats.json"  # stage 1 output
TEXT_STATS = DOCS / "ci_stats2.json"  # stage 2 output

# Prediction files for the three cohort models, free generation, both inputs.
# (contrastive = sweep run ihig0y4v, vocab = 00ksermq, literature = btb baseline.)
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
}
CONDITIONS = ("EEG", "noise")

METRICS = [
    "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "sacreBLEU",
    "ROUGE-1", "ROUGE-2", "ROUGE-L", "WER", "CER",
]  # fmt: skip

# Published Table 5 values, used purely to sanity-check the recomputation.
PAPER = {
    "literature|EEG": [14.52, 4.10, 1.76, 0.91, 1.03, 12.64, 1.42, 11.28, 102.54, 80.51],
    "vocab|EEG": [16.74, 5.87, 3.68, 2.90, 2.97, 11.85, 1.80, 10.54, 110.55, 88.11],
    "contrastive|EEG": [18.04, 7.44, 5.21, 4.36, 4.46, 12.48, 2.33, 11.07, 105.94, 85.08],
    "literature|noise": [16.18, 4.53, 1.87, 0.95, 1.08, 13.22, 1.43, 11.68, 107.46, 83.33],
    "vocab|noise": [11.39, 1.95, 0.66, 0.37, 0.37, 8.40, 0.30, 7.11, 133.79, 113.49],
    "contrastive|noise": [11.80, 1.53, 0.54, 0.32, 0.36, 8.88, 0.20, 7.74, 126.05, 101.31],
}

Z_95 = 1.96  # standard-normal critical value for a 95% interval


# --------------------------------------------------------------------------- #
# Shared parsing
# --------------------------------------------------------------------------- #


def parse_predictions(path: Path) -> tuple[list[str], list[str]]:
    """Read a decoding-results file into (references, free-generation hypotheses).

    Each record in the file looks like::

        target string: <reference>
        predicted string with tf: <teacher-forcing output>
        predicted string: <free-generation output>
        ##############################

    We keep the reference and the free-generation hypothesis (teacher forcing is
    ignored here).
    """
    references: list[str] = []
    hypotheses: list[str] = []
    record: dict[str, str] = {}
    for line in path.open(encoding="utf-8", errors="replace"):
        if line.startswith("target string:"):
            record["ref"] = line.split("target string:", 1)[1].strip()
        elif line.startswith("predicted string with tf:"):
            continue  # teacher-forcing output is not used here
        elif line.startswith("predicted string:"):
            record["hyp"] = line.split("predicted string:", 1)[1].strip()
        elif line.startswith("####"):
            if "ref" in record and "hyp" in record:
                references.append(record["ref"])
                hypotheses.append(record["hyp"])
            record = {}
    if "ref" in record and "hyp" in record:  # final record (no trailing rule)
        references.append(record["ref"])
        hypotheses.append(record["hyp"])
    return references, hypotheses


# --------------------------------------------------------------------------- #
# Stage 1: BLEU / ROUGE-1 sufficient statistics (needs the BART tokenizer)
# --------------------------------------------------------------------------- #


def dump_bleu_stats() -> None:
    """Tokenise with BART and dump per-sentence n-gram counts for BLEU/ROUGE-1.

    BLEU in the evaluation was computed by ``nltk.corpus_bleu`` over BART
    sub-word tokens, so the clipped/total n-gram counts are taken over those same
    tokens; recomputing corpus BLEU from these reproduces the paper exactly.
    """
    from transformers import BartTokenizer

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    def tokens(text: str) -> list[str]:
        return tokenizer.convert_ids_to_tokens(tokenizer.encode(text), skip_special_tokens=True)

    def ngram_counts(seq: list[str], n: int) -> Counter:
        return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))

    out: dict[str, list[dict[str, int]]] = {}
    for model, paths in MODELS.items():
        for cond in CONDITIONS:
            refs, hyps = parse_predictions(REPO / paths[cond])
            rows: list[dict[str, int]] = []
            for ref, hyp in zip(refs, hyps):
                cand, gold = tokens(hyp), tokens(ref)
                row = {"clen": len(cand), "rlen": len(gold)}
                for n in range(1, 5):
                    cand_ng, gold_ng = ngram_counts(cand, n), ngram_counts(gold, n)
                    row[f"clip{n}"] = sum(min(c, gold_ng.get(g, 0)) for g, c in cand_ng.items())
                    row[f"tot{n}"] = sum(cand_ng.values())
                cand_uni, gold_uni = Counter(hyp.split()), Counter(ref.split())
                row["ws_ov"] = sum(min(c, gold_uni.get(g, 0)) for g, c in cand_uni.items())
                row["ws_cl"] = sum(cand_uni.values())
                row["ws_rl"] = sum(gold_uni.values())
                rows.append(row)
            out[f"{model}|{cond}"] = rows
            print(f"dump-bleu: {model}/{cond} -> {len(rows)} sentences", flush=True)
    BLEU_STATS.write_text(json.dumps(out))
    print(f"dump-bleu: wrote {BLEU_STATS}", flush=True)


# --------------------------------------------------------------------------- #
# Stage 2: ROUGE-2/L, WER, CER sufficient statistics (jiwer + stdlib, no BART)
# --------------------------------------------------------------------------- #


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (two-row dynamic programming)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def dump_text_stats() -> None:
    """Dump per-sentence stats for ROUGE-2/L and (via jiwer) WER/CER.

    jiwer is the backend HuggingFace ``evaluate`` uses for WER/CER, so summing
    these edit/length counts reproduces the corpus (micro-averaged) rates exactly.
    The sentence order matches stage 1, so the two JSON files align by index.
    """
    import jiwer

    def edit_counts(ref: str, hyp: str, *, characters: bool) -> tuple[int, int]:
        """Return (edits = S+D+I, reference length = S+D+H)."""
        if not hyp.strip():  # empty hypothesis -> every reference token is a deletion
            length = len(ref) if characters else len(ref.split())
            return length, length
        proc = jiwer.process_characters if characters else jiwer.process_words
        result = proc(ref, hyp)
        edits = result.substitutions + result.deletions + result.insertions
        ref_len = result.substitutions + result.deletions + result.hits
        return edits, ref_len

    out: dict[str, list[dict[str, int]]] = {}
    for model, paths in MODELS.items():
        for cond in CONDITIONS:
            refs, hyps = parse_predictions(REPO / paths[cond])
            rows: list[dict[str, int]] = []
            for ref, hyp in zip(refs, hyps):
                cand, gold = hyp.split(), ref.split()
                cand_bi = Counter(zip(cand, cand[1:]))
                gold_bi = Counter(zip(gold, gold[1:]))
                word_edits, word_ref = edit_counts(ref, hyp, characters=False)
                char_edits, char_ref = edit_counts(ref, hyp, characters=True)
                rows.append(
                    {
                        "r2_ov": sum(min(c, gold_bi.get(g, 0)) for g, c in cand_bi.items()),
                        "r2_c": sum(cand_bi.values()),
                        "r2_r": sum(gold_bi.values()),
                        "rl_lcs": _lcs_length(cand, gold),
                        "rl_c": len(cand),
                        "rl_r": len(gold),
                        "w_edits": word_edits,
                        "w_ref": word_ref,
                        "c_edits": char_edits,
                        "c_ref": char_ref,
                    }
                )
            out[f"{model}|{cond}"] = rows
            print(f"dump-text: {model}/{cond} -> {len(rows)} sentences", flush=True)
    TEXT_STATS.write_text(json.dumps(out))
    print(f"dump-text: wrote {TEXT_STATS}", flush=True)


# --------------------------------------------------------------------------- #
# Stage 3: confidence intervals + significance (numpy only)
# --------------------------------------------------------------------------- #


def _load_arrays():
    """Load both stat files into per-condition dicts of numpy arrays."""
    import numpy as np

    if not (BLEU_STATS.exists() and TEXT_STATS.exists()):
        raise SystemExit("Missing stats; run `dump-bleu` and `dump-text` first.")
    bleu = json.loads(BLEU_STATS.read_text())
    text = json.loads(TEXT_STATS.read_text())

    def f_score(overlap, cand_len, ref_len):
        """Per-sentence F1 = 2 * overlap / (cand + ref); 0 when both are empty."""
        denom = cand_len + ref_len
        return np.where(denom > 0, 2 * overlap / np.where(denom > 0, denom, 1), 0.0)

    def col(rows, key):
        return np.array([r[key] for r in rows], dtype=float)

    data = {}
    for key in PAPER:
        a, b = bleu[key], text[key]
        data[key] = {
            "clip": np.array([[r[f"clip{k}"] for k in range(1, 5)] for r in a], dtype=float),
            "tot": np.array([[r[f"tot{k}"] for k in range(1, 5)] for r in a], dtype=float),
            "clen": col(a, "clen"),
            "rlen": col(a, "rlen"),
            "r1": f_score(col(a, "ws_ov"), col(a, "ws_cl"), col(a, "ws_rl")),
            "r2": f_score(col(b, "r2_ov"), col(b, "r2_c"), col(b, "r2_r")),
            "rl": f_score(col(b, "rl_lcs"), col(b, "rl_c"), col(b, "rl_r")),
            "we": col(b, "w_edits"),
            "wr": col(b, "w_ref"),
            "ce": col(b, "c_edits"),
            "cr": col(b, "c_ref"),
            # Raw denominator counts, kept only for the reference Wald sample size.
            "n_r1": col(a, "ws_cl"),
            "n_r2": col(b, "r2_c"),
            "n_rl": col(b, "rl_c"),
        }
    return data


def _score_vector(arr: dict) -> list[float]:
    """Compute the 10 metrics from a set of per-sentence arrays.

    BLEU/sacreBLEU are corpus ratios (aggregate counts, then divide); ROUGE is the
    mean of per-sentence F (matching the ``rouge`` library's ``avg=True``); WER/CER
    are micro-averaged (total edits / total reference length).  sacreBLEU is
    approximated by BLEU-4 (the two agree to <0.1 here).
    """
    import numpy as np

    clip = arr["clip"].sum(0)
    tot = arr["tot"].sum(0)
    cand_len = arr["clen"].sum()
    ref_len = arr["rlen"].sum()
    brevity = 1.0 if cand_len > ref_len else (math.exp(1 - ref_len / cand_len) if cand_len else 0.0)

    bleu = []
    for k in range(1, 5):
        if (tot[:k] == 0).any() or (clip[:k] == 0).any():
            bleu.append(0.0)  # nltk reports 0 when any order has no match
        else:
            bleu.append(brevity * float(np.prod(clip[:k] / tot[:k]) ** (1 / k)) * 100)

    return bleu + [
        bleu[3],  # sacreBLEU ~ BLEU-4
        arr["r1"].mean() * 100,
        arr["r2"].mean() * 100,
        arr["rl"].mean() * 100,
        arr["we"].sum() / arr["wr"].sum() * 100,
        arr["ce"].sum() / arr["cr"].sum() * 100,
    ]


def _wald_halfwidth(value_pct: float, n: int) -> float | None:
    """Wald 95% half-width in score points, or None if the proportion is invalid."""
    p = value_pct / 100
    if not 0 <= p <= 1 or n <= 0:
        return None
    return Z_95 * math.sqrt(p * (1 - p) / n) * 100


def analyse(n_boot: int, n_ar: int, seed: int) -> None:
    import numpy as np

    rng = np.random.default_rng(seed)
    data = _load_arrays()

    def select(arr, idx):
        return {k: v[idx] for k, v in arr.items()}

    def swap(a, b, mask):
        """Build the two systems after swapping per-sentence outputs where mask is True."""
        a_swapped, b_swapped = {}, {}
        for k in a:
            m = mask[:, None] if a[k].ndim == 2 else mask
            a_swapped[k] = np.where(m, b[k], a[k])
            b_swapped[k] = np.where(m, a[k], b[k])
        return a_swapped, b_swapped

    print("Recomputation vs. paper (should match within rounding; ROUGE-L approximate):")
    for key, arr in data.items():
        got = _score_vector(arr)
        print(
            f"  {key:18s} "
            + "  ".join(
                f"{m}={g:.2f}/{p}"
                for m, g, p in zip(
                    ["B1", "B4", "R1", "R2", "RL", "WER", "CER"],
                    [got[0], got[3], got[5], got[6], got[7], got[8], got[9]],
                    [PAPER[key][i] for i in (0, 3, 5, 6, 7, 8, 9)],
                )
            )
        )

    print("\nCell 95% intervals  ::  value  |  Wald +-  |  bootstrap [lo, hi]")
    for key, arr in data.items():
        tot = arr["tot"].sum(0)
        # Wald sample size per metric: candidate n-grams (BLEU/sacreBLEU),
        # token / n-gram counts (ROUGE), or reference length (WER/CER).
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
        samples = np.array(
            [
                _score_vector(select(arr, rng.integers(0, arr["clen"].size, arr["clen"].size)))
                for _ in range(n_boot)
            ]
        )
        lo, hi = np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)
        print(f"\n{key}:")
        for i, name in enumerate(METRICS):
            value = PAPER[key][i]
            half = _wald_halfwidth(value, wald_n[i])
            wald = f"+-{half:.2f}" if half is not None else "n/a(>100%)"
            print(f"   {name:10s} {value:7.2f}  | {wald:>10s} | [{lo[i]:6.2f}, {hi[i]:6.2f}]")

    print("\nSignificance of differences  ::  bootstrap 95% CI + approximate-randomisation p")
    comparisons = [("contrastive|EEG", "vocab|EEG", "contrastive - vocab (real EEG)")]
    comparisons += [(f"{m}|EEG", f"{m}|noise", f"{m}: real EEG - noise") for m in MODELS]
    for a_key, b_key, label in comparisons:
        a, b = data[a_key], data[b_key]
        n = a["clen"].size
        observed = np.array(_score_vector(a)) - np.array(_score_vector(b))
        diffs = np.array(
            [
                np.array(_score_vector(select(a, idx))) - np.array(_score_vector(select(b, idx)))
                for idx in (rng.integers(0, n, n) for _ in range(n_boot))
            ]
        )
        b_lo, b_hi = np.percentile(diffs, 2.5, axis=0), np.percentile(diffs, 97.5, axis=0)
        exceed = np.zeros(len(METRICS))
        for _ in range(n_ar):
            a_sw, b_sw = swap(a, b, rng.integers(0, 2, n).astype(bool))
            d = np.array(_score_vector(a_sw)) - np.array(_score_vector(b_sw))
            exceed += np.abs(d) >= np.abs(observed) - 1e-9
        p_value = (exceed + 1) / (n_ar + 1)
        print(f"\n{label}:")
        for i, name in enumerate(METRICS):
            verdict = "significant" if p_value[i] < 0.05 else "n.s."
            print(
                f"   {name:10s} d={observed[i]:+6.2f}  boot[{b_lo[i]:+6.2f}, {b_hi[i]:+6.2f}]"
                f"  AR p={p_value[i]:.4f}  {verdict}"
            )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("dump-bleu", help="stage 1: BART-token BLEU/ROUGE-1 counts")
    sub.add_parser("dump-text", help="stage 2: ROUGE-2/L, WER, CER counts")
    analyse_parser = sub.add_parser("analyse", help="stage 3: CIs + significance")
    analyse_parser.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples")
    analyse_parser.add_argument(
        "--n-ar", type=int, default=10000, help="approximate-randomisation trials"
    )
    analyse_parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.stage == "dump-bleu":
        dump_bleu_stats()
    elif args.stage == "dump-text":
        dump_text_stats()
    elif args.stage == "analyse":
        analyse(args.n_boot, args.n_ar, args.seed)


if __name__ == "__main__":
    main()
