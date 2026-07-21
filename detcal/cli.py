"""detcal CLI.

  detcal run --corpus hc3-mini --n 100 --detector fast-detect-gpt --out reports/
  detcal run --corpus path/to/labeled.jsonl ...
  detcal corpora
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import __version__
from . import calibrate
from .corpus import fetch_hc3_mini, load_jsonl
from .receipts import write_report

TARGET_FPRS = (0.01, 0.05)


def _build_detector(args):
    if args.detector == "fast-detect-gpt":
        from .detectors.fast_detect_gpt import FastDetectGPT

        return FastDetectGPT(model_id=args.model, quant=args.quant, max_tokens=args.max_tokens)
    raise SystemExit(f"unknown detector: {args.detector}")


def cmd_run(args) -> int:
    root = Path(args.root).resolve()
    if args.corpus == "hc3-mini":
        corpus = fetch_hc3_mini(root / "corpora", n_per_class=args.n, seed=args.seed)
    else:
        corpus = load_jsonl(Path(args.corpus))
    print(f"[corpus] {corpus.name} counts={corpus.counts} sha256={corpus.sha256[:16]}…")

    det = _build_detector(args)
    print(f"[detector] loading {det.name} ({args.model}, {args.quant})…")
    det.load()
    print(f"[detector] loaded + quantization verified")

    rows = []
    t_all = time.time()
    for i, item in enumerate(corpus.items, 1):
        t0 = time.time()
        s = det.score(item["text"])
        rows.append(
            {"id": item["id"], "label": item["label"], "score": s, "seconds": round(time.time() - t0, 3)}
        )
        if i % 25 == 0 or i == len(corpus.items):
            print(f"[score] {i}/{len(corpus.items)} ({time.time() - t_all:.0f}s)")

    scores = np.array([r["score"] for r in rows], dtype=np.float64)
    labels = np.array([r["label"] for r in rows])

    auroc = calibrate.auroc(scores, labels)
    auroc_ci = calibrate.bootstrap_ci(scores, labels, calibrate.auroc, n_boot=args.n_boot, seed=args.seed)
    thresholds = []
    for tf in TARGET_FPRS:
        row = calibrate.threshold_at_fpr(scores, labels, tf)
        thr = row["threshold"]
        row["tpr_ci"] = calibrate.bootstrap_ci(
            scores,
            labels,
            lambda s, l, thr=thr: float((s[l == "ai"] >= thr).mean()),
            n_boot=args.n_boot,
            seed=args.seed,
        )
        thresholds.append(row)

    results = {
        "detcal_version": __version__,
        "auroc": auroc,
        "auroc_ci": auroc_ci,
        "thresholds": thresholds,
        "roc": calibrate.roc_points(scores, labels),
        "n_boot": args.n_boot,
    }
    outdir = root / "reports" / f"{datetime.now():%Y%m%d-%H%M%S}-{args.detector}"
    reproduce = (
        f"detcal run --corpus {args.corpus} --n {args.n} --seed {args.seed} "
        f"--detector {args.detector} --model {args.model} --quant {args.quant} "
        f"--max-tokens {args.max_tokens}"
    )
    report = write_report(outdir, corpus, det.metadata(), rows, results, reproduce)
    print(f"\n[receipt] {report}")
    print(f"[receipt] AUROC {auroc:.4f} (CI {auroc_ci[0]:.4f}-{auroc_ci[1]:.4f})")
    for row in thresholds:
        print(
            f"[receipt] @FPR<={row['target_fpr']:.0%}: thr={row['threshold']:.4f} "
            f"achievedFPR={row['achieved_fpr']:.3%} TPR={row['achieved_tpr']:.1%}"
        )
    return 0


def cmd_corpora(_args) -> int:
    print("hc3-mini    seeded subsample of HC3 (Hello-SimpleAI), human vs ChatGPT answers")
    print("<path>      any JSONL with rows {id?, text, label: human|ai}")
    return 0


def cmd_score(args) -> int:
    import json as _json

    from .single import format_result, score_document

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()
    if not text.strip():
        raise SystemExit("no text to score")
    r = score_document(Path(args.report), text)
    print(_json.dumps(r, indent=1) if args.json else format_result(r))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="detcal", description=__doc__)
    p.add_argument("--version", action="version", version=f"detcal {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="score a labeled corpus and emit a receipts report")
    r.add_argument("--corpus", default="hc3-mini")
    r.add_argument("--n", type=int, default=100, help="per-class sample count for built-in corpora")
    r.add_argument("--seed", type=int, default=17)
    r.add_argument("--detector", default="fast-detect-gpt")
    r.add_argument("--model", default="tiiuae/falcon-7b")
    r.add_argument("--quant", default="4bit", choices=["4bit", "fp16"])
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--root", default=".", help="project root holding corpora/ and reports/")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("corpora", help="list built-in corpora")
    c.set_defaults(fn=cmd_corpora)

    s = sub.add_parser(
        "score",
        help="score ONE document against a calibrated report (stdin, --file, or --text); logs nothing",
    )
    s.add_argument("--report", required=True, help="path to a reports/<ts>-<detector>/ directory")
    s.add_argument("--file")
    s.add_argument("--text")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_score)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
