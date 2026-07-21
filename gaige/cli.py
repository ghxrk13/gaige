# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""gaige CLI.

gaige run     --corpus hc3-mini --n 100 --detector fast-detect-gpt
gaige run     --corpus path/to/labeled.jsonl ...
gaige analyze --report reports/<ts>-<detector>/     # re-derive results, no GPU needed
gaige analyze --scores path/to/scores.csv
gaige score   --report reports/<ts>-<detector>/ --file draft.md
gaige corpora
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from . import __version__, runstate
from . import analyze as analyze_mod
from .corpus import fetch_hc3_mini, load_jsonl
from .receipts import write_report

TARGET_FPRS = analyze_mod.TARGET_FPRS


def _resolve_model(args):
    """Pick a scoring model when the caller didn't. Returns (model_id, quant, auto_selected).

    Auto-selection exists so `gaige run` works on the machine you actually have. It is recorded
    on the receipt, because a number whose instrument was chosen for you should say so.
    """
    from .detectors.fast_detect_gpt import DEFAULT_MODEL, FastDetectGPT

    if args.model:
        return args.model, args.quant, False
    device, _fell_back = FastDetectGPT(device=args.device).resolve_device()
    model_id, quant = DEFAULT_MODEL[device]
    print(
        f"[detector] no --model given; auto-selected {model_id} ({quant}) for device={device}. "
        "Pass --model to choose deliberately."
    )
    return model_id, quant, True


def _build_detector(args):
    if args.detector == "fast-detect-gpt":
        from .detectors.fast_detect_gpt import FastDetectGPT

        model_id, quant, auto = _resolve_model(args)
        args.model, args.quant = model_id, quant  # so the reproduce line is explicit
        return FastDetectGPT(
            model_id=model_id,
            quant=quant,
            max_tokens=args.max_tokens,
            device=args.device,
            model_auto_selected=auto,
        )
    raise SystemExit(f"unknown detector: {args.detector}")


def cmd_run(args) -> int:
    root = Path(args.root).resolve()
    if args.corpus == "hc3-mini":
        corpus = fetch_hc3_mini(root / "corpora", n_per_class=args.n, seed=args.seed)
    else:
        corpus = load_jsonl(Path(args.corpus))
    print(f"[corpus] {corpus.name} counts={corpus.counts} sha256={corpus.sha256[:16]}...")

    det = _build_detector(args)

    resuming = bool(args.resume)
    state = None
    done: dict[str, dict] = {}
    if resuming:
        outdir = Path(args.resume).resolve()
        state = runstate.read_runstate(outdir)
        # Cheap check first: refuse before spending minutes on a model load.
        runstate.check_args_match(state, corpus, args.model, args.quant, args.max_tokens)
        done = {r["id"]: r for r in runstate.load_partial(outdir)}
        print(f"[resume] {outdir}")
        print(f"[resume] {len(done)}/{len(corpus.items)} already scored; continuing")
    else:
        outdir = root / "reports" / f"{datetime.now():%Y%m%d-%H%M%S}-{args.detector}"

    print(f"[detector] loading {det.name} ({args.model}, {args.quant}, device={args.device})...")
    det.load()
    print("[detector] loaded + quantization verified")

    reproduce = (
        f"gaige run --corpus {args.corpus} --n {args.n} --seed {args.seed} "
        f"--detector {args.detector} --model {args.model} --quant {args.quant} "
        f"--device {args.device} --max-tokens {args.max_tokens}"
    )
    if resuming:
        # Full fingerprint check: only now are library versions and the resolved device known.
        runstate.check_instrument_match(state, det.metadata())
    else:
        runstate.write_runstate(outdir, corpus, det.metadata(), reproduce)

    fh, writer = runstate.open_partial(outdir)
    rows = []
    t_all = time.time()
    try:
        for i, item in enumerate(corpus.items, 1):
            if item["id"] in done:
                rows.append(done[item["id"]])
                continue
            t0 = time.time()
            s = det.score(item["text"])
            row = {
                "id": item["id"],
                "label": item["label"],
                "score": s,
                "seconds": round(time.time() - t0, 3),
            }
            runstate.append_row(fh, writer, row)  # on disk before we move on
            rows.append(row)
            if i % 25 == 0 or i == len(corpus.items):
                print(f"[score] {i}/{len(corpus.items)} ({time.time() - t_all:.0f}s)")
    except (KeyboardInterrupt, Exception):
        fh.close()
        print(
            f"\n[interrupted] {len(rows)}/{len(corpus.items)} scores are safe on disk.\n"
            f"[interrupted] resume with:  gaige run --corpus {args.corpus} --n {args.n} "
            f"--seed {args.seed} --resume {outdir}"
        )
        raise
    fh.close()

    results = analyze_mod.compute_results(
        rows, target_fprs=TARGET_FPRS, n_boot=args.n_boot, seed=args.seed
    )
    report = write_report(outdir, corpus, det.metadata(), rows, results, reproduce)
    runstate.mark_complete(outdir)
    _print_receipt(report, results)
    return 0


def _print_receipt(report_path: Path, results: dict) -> None:
    lo, hi = results["auroc_ci"]
    print(f"\n[receipt] {report_path}")
    print(f"[receipt] AUROC {results['auroc']:.4f} (CI {lo:.4f}-{hi:.4f})")
    for row in results["thresholds"]:
        print(
            f"[receipt] @FPR<={row['target_fpr']:.0%}: thr={row['threshold']:.4f} "
            f"achievedFPR={row['achieved_fpr']:.3%} TPR={row['achieved_tpr']:.1%}"
        )


def cmd_analyze(args) -> int:
    """Re-derive results from scores that already exist. No model, no GPU, no re-scoring."""
    if args.report:
        src = Path(args.report).resolve()
        rows, corpus, detector_meta = analyze_mod.load_report(src)
        origin = f"report {src}"
    else:
        src = Path(args.scores).resolve()
        rows = analyze_mod.read_scores_csv(src)
        corpus, detector_meta = analyze_mod.UNKNOWN_CORPUS, dict(analyze_mod.UNKNOWN_DETECTOR)
        origin = f"scores {src}"

    n_h = sum(1 for r in rows if r["label"] == "human")
    print(f"[analyze] {origin}")
    print(
        f"[analyze] {len(rows)} scores (human {n_h}, ai {len(rows) - n_h}) | corpus {corpus.name}"
    )
    if detector_meta.get("instrument_unknown"):
        print(
            "[analyze] WARNING: no instrument fingerprint accompanies these scores. Thresholds "
            "below describe THIS score set only and attest to no measurable instrument."
        )

    results = analyze_mod.compute_results(
        rows, target_fprs=TARGET_FPRS, n_boot=args.n_boot, seed=args.seed
    )

    outdir = (
        Path(args.out).resolve()
        if args.out
        else (
            src.parent / f"{datetime.now():%Y%m%d-%H%M%S}-analyze"
            if src.is_file()
            else src.parent / f"{datetime.now():%Y%m%d-%H%M%S}-analyze"
        )
    )
    reproduce = (
        f"gaige analyze {'--report' if args.report else '--scores'} {src} "
        f"--n-boot {args.n_boot} --seed {args.seed}"
    )
    report = write_report(outdir, corpus, detector_meta, rows, results, reproduce)
    _print_receipt(report, results)
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
    p = argparse.ArgumentParser(prog="gaige", description=__doc__)
    p.add_argument("--version", action="version", version=f"gaige {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="score a labeled corpus and emit a receipts report")
    r.add_argument("--corpus", default="hc3-mini")
    r.add_argument("--n", type=int, default=100, help="per-class sample count for built-in corpora")
    r.add_argument("--seed", type=int, default=17)
    r.add_argument("--detector", default="fast-detect-gpt")
    r.add_argument(
        "--model",
        default=None,
        help="scoring model; default is chosen per device (see --help output of run)",
    )
    r.add_argument(
        "--quant",
        default="4bit",
        choices=["4bit", "fp16", "fp32"],
        help="4bit is CUDA-only; use fp32 on CPU",
    )
    r.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="auto prefers CUDA and falls back to CPU, recording the fallback",
    )
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--root", default=".", help="project root holding corpora/ and reports/")
    r.add_argument(
        "--resume",
        help="continue an interrupted run directory; refuses if the instrument or corpus changed",
    )
    r.set_defaults(fn=cmd_run)

    a = sub.add_parser(
        "analyze",
        help="re-derive results from existing scores (no model, no GPU) — for replaying a "
        "report, re-running the statistics, or calibrating where the GPU isn't",
    )
    a_src = a.add_mutually_exclusive_group(required=True)
    a_src.add_argument("--report", help="an existing reports/<ts>-<detector>/ directory")
    a_src.add_argument("--scores", help="a bare scores.csv (columns: label, score[, id, seconds])")
    a.add_argument("--n-boot", type=int, default=1000)
    a.add_argument("--seed", type=int, default=17)
    a.add_argument(
        "--out", help="output directory (default: a new timestamped dir beside the source)"
    )
    a.set_defaults(fn=cmd_analyze)

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
