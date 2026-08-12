"""Reproducible FLAN-T5-XXL control experiment; does not edit original code."""
import argparse
import json
import os
import platform
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from algorithms import (Algorithm, ConllConfig, CrossNERAIConfig,
                        CrossNERLiteratureConfig, CrossNERMusicConfig,
                        CrossNERNaturalSciencesConfig, CrossNERPoliticsConfig,
                        FewNERDINTRATestConfig)
from data import load_conll2003, load_cross_ner, load_few_nerd, sample_all_types
from models_extra import get_flan_t5_model
from normalization_extra import LabelSchema, NormalizedAlgorithm


TARGETS = {
    "conll": (71.69, 2.53), "fewnerd": (55.32, 1.56),
    "politics": (62.98, 3.21), "literature": (66.51, 1.56),
    "music": (73.47, 2.34), "ai": (61.88, 3.67), "science": (70.38, 4.64),
}
CONFIGS = {
    "politics": CrossNERPoliticsConfig, "literature": CrossNERLiteratureConfig,
    "music": CrossNERMusicConfig, "ai": CrossNERAIConfig,
    "science": CrossNERNaturalSciencesConfig,
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_splits(name):
    if name == "conll":
        return ConllConfig(), load_conll2003("train"), load_conll2003("test")
    if name == "fewnerd":
        return FewNERDINTRATestConfig(), load_few_nerd(category="intra", split="test"), load_few_nerd(category="intra", split="test")
    return CONFIGS[name](), load_cross_ner(category=name, split="train"), load_cross_ner(category=name, split="test")


def configure(config, train, model, normalized):
    schema = LabelSchema.from_dataset(train)
    algorithm = (NormalizedAlgorithm(label_schema=schema) if normalized else Algorithm())
    algorithm.set_model_fn(model)
    examples = sample_all_types(train, 3)
    texts = examples["text"].tolist()
    tokens = examples["text"].apply(lambda text: text.split(" ")).tolist()
    config.autogenerate_annotations(algorithm, texts, tokens, examples["exact_types"].tolist())
    config.set_config(algorithm, exemplar=True, coT=True, defn=True, tf=True)
    return algorithm, examples


def evaluate(test, algorithm, limit):
    # Direct calls intentionally avoid run.py's fork-based timeout, which is
    # incompatible with CUDA models initialized in the parent process.
    try:
        from seqeval.metrics import f1_score
    except ImportError as exc:
        raise RuntimeError(
            "The replication runner requires seqeval. Install the packages in "
            "requirements-t5.txt before running a benchmark."
        ) from exc
    subset = test.sample(min(limit, len(test))).reset_index(drop=True)
    rows, predictions, truths = [], [], []
    for index, row in subset.iterrows():
        algorithm.set_para(row["text"])
        if hasattr(algorithm, "normalization_records"):
            algorithm.normalization_records = []
        tokens = row.get("true_tokens", None)
        try:
            pred, raw = algorithm.perform_span(true_tokens=tokens, verbose=False)
            error = None
        except Exception as exc:  # retain the example as all-O; never silently skip it
            pred, raw, error = ["O"] * len(row["exact_types"]), None, repr(exc)
        predictions.append(pred)
        truths.append(row["exact_types"])
        rows.append({
            "index": index, "text": row["text"], "truth": row["exact_types"],
            "pred": pred, "raw_output": raw, "error": error,
            "normalization_records": getattr(algorithm, "normalization_records", None),
            "f1": f1_score([row["exact_types"]], [pred]),
        })
    return f1_score(truths, predictions, average="micro"), pd.DataFrame(rows)


def run_benchmark(name, model, seed, limit, normalized, output_dir):
    seed_everything(seed)
    config, train, test = get_splits(name)
    algorithm, examples = configure(config, train, model, normalized)
    score, rows = evaluate(test, algorithm, limit)
    mode = "normalized" if normalized else "legacy"
    path = os.path.join(output_dir, f"t5xxl_{name}_{mode}_seed{seed}.csv")
    rows.to_csv(path, index=False)
    return score, examples["text"].tolist(), path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="results/t5xxl_replication")
    parser.add_argument("--datasets", nargs="+", choices=tuple(TARGETS), default=list(TARGETS),
                        help="Benchmark subsets to run (default: all seven).")
    parser.add_argument("--modes", nargs="+", choices=("legacy", "normalized"),
                        default=("legacy", "normalized"),
                        help="Parser variants to run (default: both).")
    parser.add_argument("--model-name", default="google/flan-t5-xxl")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--no-8bit", action="store_true",
                        help="Disable bitsandbytes 8-bit loading; needs substantially more memory.")
    parser.add_argument("--smoke", action="store_true", help="one 5-example run; does not test paper scores")
    args = parser.parse_args()
    if args.smoke:
        args.limit, args.runs = 5, 1
    if args.limit <= 0 or args.runs <= 0:
        parser.error("--limit and --runs must both be positive")
    os.makedirs(args.output_dir, exist_ok=True)
    model = get_flan_t5_model(model_name=args.model_name, load_in_8bit=not args.no_8bit,
                               max_new_tokens=args.max_new_tokens)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
              "limit": args.limit, "runs": args.runs, "model": model.model,
              "quantization": "full precision" if args.no_8bit else "8-bit", "torch": torch.__version__,
              "python": platform.python_version(), "targets": TARGETS, "results": {}}
    for name in args.datasets:
        modes = {}
        for normalized in (mode == "normalized" for mode in args.modes):
            scores, paths, exemplars = [], [], None
            for run in range(args.runs):
                score, exemplars, path = run_benchmark(name, model, args.seed + run,
                                                       args.limit, normalized, args.output_dir)
                scores.append(float(score * 100)); paths.append(path)
            mean, std = float(np.mean(scores)), float(np.std(scores))
            target, tolerance = TARGETS[name]
            modes["normalized" if normalized else "legacy"] = {
                "scores": scores, "mean": mean, "std": std, "within_paper_std": abs(mean - target) <= tolerance,
                "files": paths, "exemplar_texts": exemplars,
            }
        report["results"][name] = modes
        print(name, json.dumps({key: value["mean"] for key, value in modes.items()}))
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
