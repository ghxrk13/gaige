# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""gaige CLI.

gaige run     --corpus hc3-mini --n 100 --detector fast-detect-gpt
gaige run     --corpus path/to/labeled.jsonl ...
gaige analyze --report reports/<ts>-<detector>/     # re-derive results, no GPU needed
gaige analyze --scores path/to/scores.csv
gaige score   --report reports/<ts>-<detector>/ --file draft.md
gaige probe run --probes probes.jsonl --provider local-hf --model gpt2-large --cutoff 2024-06-01
gaige providers
gaige test-connection --endpoint http://127.0.0.1:8080 [--gguf model.gguf]
gaige corpora
gaige corpus prepare-raid --generators gpt4,mistral-chat --domains abstracts,reddit
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
    if args.detector == "binoculars":
        from .detectors.binoculars import Binoculars

        return Binoculars(
            observer_id=args.observer,
            performer_id=args.performer,
            quant=args.quant,
            max_tokens=args.max_tokens,
            device=args.device,
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

    model_desc = (
        f"{args.observer} + {args.performer}" if args.detector == "binoculars" else args.model
    )
    print(f"[detector] loading {det.name} ({model_desc}, {args.quant}, device={args.device})...")
    det.load()
    print("[detector] loaded + quantization verified")

    # The reproduce command pins the RESOLVED device, never "auto": auto re-run on another
    # machine would silently swap instruments, which is the exact failure receipts exist to
    # prevent. Model is already resolved above for the same reason.
    resolved_device = det.metadata()["device"]
    model_bits = (
        f"--observer {args.observer} --performer {args.performer}"
        if args.detector == "binoculars"
        else f"--model {args.model}"
    )
    reproduce = (
        f"gaige run --corpus {args.corpus} --n {args.n} --seed {args.seed} "
        f"--detector {args.detector} {model_bits} --quant {args.quant} "
        f"--device {resolved_device} --max-tokens {args.max_tokens}"
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
            # Derived word count (and optional metadata) ride along for subgroup receipts.
            # The corpus is the source of truth for both, so resumed rows get them too.
            extras = {"n_words": len(item["text"].split())}
            if item.get("meta"):
                extras["meta"] = item["meta"]
            if item["id"] in done:
                rows.append({**done[item["id"]], **extras})
                continue
            t0 = time.time()
            s = det.score(item["text"])
            row = {
                "id": item["id"],
                "label": item["label"],
                "score": s,
                "seconds": round(time.time() - t0, 3),
                **extras,
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
        rows,
        target_fprs=TARGET_FPRS,
        n_boot=args.n_boot,
        seed=args.seed,
        harm_volume=args.harm_volume,
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
            f"FPRcal={row['achieved_fpr']:.3%} (in-sample) TPR={row['achieved_tpr']:.1%}"
        )
    for row in results.get("conformal", []):
        if "unavailable" in row:
            print(f"[receipt] conformal a={row['alpha']:g}: refused ({row['unavailable']})")
        else:
            print(
                f"[receipt] conformal a={row['alpha']:g}: thr={row['threshold']:.4f} "
                f"TPR={row['tpr']:.1%} (marginal FPR guarantee <= {row['alpha']:g})"
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
        rows,
        target_fprs=TARGET_FPRS,
        n_boot=args.n_boot,
        seed=args.seed,
        harm_volume=args.harm_volume,
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
    print(
        "raid        seeded slices of RAID (Dugan et al., ACL 2024) via `gaige corpus prepare-raid`"
    )
    print("<path>      any JSONL with rows {id?, text, label: human|ai}")
    return 0


def cmd_corpus_prepare_raid(args) -> int:
    from .corpus_raid import prepare_raid_slice

    gens = [s.strip() for s in args.generators.split(",") if s.strip()]
    doms = [s.strip() for s in args.domains.split(",") if s.strip()]
    atks = [s.strip() for s in args.attacks.split(",") if s.strip()]
    c = prepare_raid_slice(
        Path(args.root) / "corpora",
        generators=gens,
        domains=doms,
        attacks=atks,
        per_cell=args.per_cell,
        seed=args.seed,
        min_words=args.min_words,
        max_words=args.max_words,
        source=args.source,
        raid_csv=Path(args.csv) if args.csv else None,
    )
    print(f"[corpus] {c.name} counts={c.counts} sha256={c.sha256[:16]}")
    print(f"[corpus] revision={c.meta.get('dataset_revision')} → {c.path}")
    print(f"next: gaige run --corpus {c.path} --detector fast-detect-gpt")
    return 0


def _build_provider(args):
    import os

    if args.provider == "local-hf":
        from .providers.local_hf import LocalHF

        if not args.model:
            raise SystemExit("--model is required with --provider local-hf")
        return LocalHF(model_id=args.model, dtype=args.dtype, device=args.device)
    if args.provider == "llamacpp":
        from .providers.llamacpp import LlamaCpp

        endpoint = args.endpoint or os.environ.get("GAIGE_AI_ENDPOINT")
        if not endpoint:
            raise SystemExit(
                "--endpoint (or GAIGE_AI_ENDPOINT) is required with --provider llamacpp"
            )
        return LlamaCpp(
            endpoint=endpoint,
            model=args.model or os.environ.get("GAIGE_AI_MODEL", ""),
            gguf_path=args.gguf,
        )
    if args.provider == "ollama":
        from .providers.ollama import Ollama

        if not args.model:
            raise SystemExit(
                "--model is required with --provider ollama (e.g. qwen2.5:7b-instruct)"
            )
        return Ollama(
            endpoint=args.endpoint or os.environ.get("GAIGE_AI_ENDPOINT", "http://127.0.0.1:11434"),
            model=args.model,
        )
    raise SystemExit(f"unknown provider: {args.provider}")


def cmd_probe_run(args) -> int:
    from datetime import datetime as _dt

    from . import probeauthor
    from .proberun import run_probes
    from .probes import load_probes
    from .providers.base import Decoding, require_local_or_optin

    probeset = load_probes(Path(args.probes))
    print(f"[probes] {probeset.name} vintages={probeset.vintages} sha256={probeset.sha256[:16]}...")
    provider = _build_provider(args)
    require_local_or_optin(provider, args.allow_remote_text)
    decoding = Decoding(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.gen_seed,
        max_new_tokens=args.max_new_tokens,
    )
    try:
        print(f"[manifest] {probeauthor.check_run_against_manifest(Path(args.probes), decoding)}")
    except probeauthor.ManifestViolation as e:
        raise SystemExit(str(e)) from None
    print(f"[provider] loading {args.provider}...")
    meta = provider.metadata()  # triggers load/connect; resolves device + attestation
    print(f"[provider] ready · attestation: {meta.get('attestation', '?')}")

    resolved_bits = ""
    if args.provider == "local-hf":
        resolved_bits = f" --model {args.model} --dtype {args.dtype} --device {meta['device']}"
    elif args.provider == "llamacpp":
        resolved_bits = f" --endpoint {provider.endpoint}" + (
            f" --gguf {args.gguf}" if args.gguf else ""
        )
    elif args.provider == "ollama":
        resolved_bits = f" --model {args.model} --endpoint {provider.endpoint}"
    reproduce = (
        f"gaige probe run --probes {args.probes} --provider {args.provider}{resolved_bits} "
        f"--cutoff {args.cutoff} --temperature {args.temperature:g} "
        f"--max-new-tokens {args.max_new_tokens} --n-boot {args.n_boot} --seed {args.seed}"
        + (" --ptrue" if args.ptrue else "")
    )

    if args.resume and args.replicates > 1:
        raise SystemExit("--resume continues ONE run directory; do not combine with --replicates")
    n_reps = max(1, args.replicates)
    base = Path(args.root).resolve() / "reports" / f"{_dt.now():%Y%m%d-%H%M%S}-probes"
    reg_dir = Path(args.registry) if args.registry else Path(args.root).resolve() / "registry"

    for i in range(1, n_reps + 1):
        if args.resume:
            outdir = Path(args.resume).resolve()
        elif n_reps == 1:
            outdir = base
        else:
            outdir = Path(f"{base}-r{i}")
        results = run_probes(
            probeset,
            provider,
            decoding,
            cutoff=args.cutoff,
            outdir=outdir,
            n_boot=args.n_boot,
            seed=args.seed,
            resume=bool(args.resume),
            reproduce_cmd=reproduce,
            with_ptrue=args.ptrue,
        )
        print(f"\n[receipt] {outdir / 'report.md'}")
        for v, d in sorted(results["by_vintage"].items()):
            ci = d.get("accuracy_ci")
            ci_s = f" (CI {ci[0]:.1%}-{ci[1]:.1%})" if ci else ""
            print(f"[receipt] vintage {v}: accuracy {d['accuracy']:.1%} n={d['n']}{ci_s}")
            if "m3" in d:
                m = d["m3"]
                print(
                    f"[receipt] vintage {v}: P(True) {m['mean_confidence']:.1%} "
                    f"gap {m['gap']:+.1%} ECE {m['ece']:.3f}"
                )
        for v, d in sorted(results["post_cutoff_share"].items()):
            print(
                f"[receipt] vintage {v}: post-cutoff {d['post_cutoff']}/{d['n']} ({d['share']:.0%})"
            )
        if args.register:
            from . import registry as registry_mod

            series = registry_mod.record_run(reg_dir, outdir, replicate=n_reps > 1)
            print(
                f"[series] {series['series_id']}: {len(series['runs'])} run(s) -> "
                f"{Path(reg_dir) / series['series_id'] / 'series-report.md'}"
            )
    return 0


def cmd_probe_new(args) -> int:
    from .probeauthor import new_probe_set

    try:
        out, mpath = new_probe_set(Path(args.out), vintage=args.vintage, cutoff=args.cutoff)
    except (FileExistsError, ValueError) as e:
        raise SystemExit(str(e)) from None
    print(f"[new] probe template: {out}")
    print(f"[new] manifest:       {mpath}")
    print(
        f"[next] author probes (short checkable answers; every source_date after "
        f"{args.cutoff}), link the frozen control subset in the manifest, then: "
        f"gaige probe lint --probes {out}"
    )
    return 0


def cmd_probe_lint(args) -> int:
    from .probeauthor import lint

    rep = lint(Path(args.probes))
    for w in rep.warnings:
        print(f"[warn] {w}")
    for e in rep.errors:
        print(f"[FAIL] {e}")
    for v, d in sorted(rep.vintages.items()):
        pc = f" · post-cutoff {d['post_cutoff']}/{d['n']}" if "post_cutoff" in d else ""
        print(f"[lint] vintage {v}: n={d['n']}{pc}")
    if rep.ok:
        print(
            f"[lint] PASS — meets the signed authoring decisions ({len(rep.warnings)} warning(s))"
        )
        return 0
    print(f"[lint] FAIL — {len(rep.errors)} error(s); a study set must lint clean")
    return 1


def cmd_series(args) -> int:
    from . import registry as registry_mod

    reg = Path(args.registry)
    if args.series_cmd == "list":
        rows = registry_mod.list_series(reg)
        if not rows:
            print(f"(no series under {reg})")
            return 0
        for s in rows:
            print(
                f"{s['series_id']}  {s['provider']}:{s['model']}  runs={s['runs']}  "
                f"vintages={','.join(s['vintages'])}"
            )
        return 0
    if args.series_cmd == "watch":
        return _cmd_series_watch(args, reg, registry_mod)
    p = reg / args.id / "series-report.md"
    if not p.exists():
        raise SystemExit(f"no series {args.id!r} under {reg} (try: gaige series list)")
    print(p.read_text(encoding="utf-8"))
    return 0


def _cmd_series_watch(args, reg: Path, registry_mod) -> int:
    import json as _json

    from . import monitors

    sdir = reg / args.id
    if not (sdir / "series.json").exists():
        raise SystemExit(f"no series {args.id!r} under {reg} (try: gaige series list)")
    series = _json.loads((sdir / "series.json").read_text(encoding="utf-8"))
    direction = args.direction or ("up" if args.quantity == "gap" else "down")
    vintages = [args.vintage] if args.vintage else sorted(series["vintage_hashes"])

    lines = [
        f"# gaige monitors — series {series['series_id']} · quantity {args.quantity} "
        f"(alarm direction: {direction})",
        "",
        "Per-interval conformal alarms carry a marginal finite-sample false-alarm bound "
        "calibrated on the Day-0 replicates; Page-Hinkley and CUSUM are cumulative "
        "detectors reported with tuned constants and NO claimed guarantee (drift-literature "
        "practice; conformal test martingales are the principled extension, future work).",
    ]
    for v in vintages:
        seq = registry_mod.vintage_sequences(series, v, quantity=args.quantity)
        lines += [
            "",
            f"## vintage {v} — reference n={len(seq['reference'])} (replicates), "
            f"observed n={len(seq['observed'])} (intervals)",
        ]
        if not seq["observed"]:
            lines.append("- no observed intervals with this quantity; nothing to watch")
            continue
        if not seq["reference"]:
            lines.append("- no replicate reference; run the Day-0 protocol first")
            continue
        for res in monitors.watch(
            seq["reference"], seq["observed"], alpha=args.alpha, direction=direction
        ):
            if res.get("refused"):
                lines.append(f"- **{res['monitor']}**: REFUSED — {res['refused']}")
                continue
            when = (
                "; ".join(f"interval {i} ({seq['labels'][i]})" for i in res["alarms"])
                if res["alarms"]
                else "no alarms"
            )
            extra = (
                f" · threshold {res['threshold']:+.4f} · {res['guarantee']}"
                if "threshold" in res
                else f" · params {res['params']} · {res['guarantee']}"
            )
            lines.append(f"- **{res['monitor']}**: {when}{extra}")
    text = "\n".join(lines) + "\n"
    (sdir / "monitors-report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"[monitors] written to {sdir / 'monitors-report.md'}")
    return 0


def cmd_plan(_args) -> int:
    from . import plan as plan_mod

    env = plan_mod.inspect_environment()
    print(plan_mod.render(env, plan_mod.build_plan(env)))
    return 0


def cmd_providers(_args) -> int:
    import os

    print("local-hf    in-process transformers load; attestation: verified")
    print("llamacpp    llama.cpp server via /v1; attestation graded: verified (with --gguf")
    print("            hash match) / self-reported (/props answers) / opaque")
    print("ollama      local ollama server; attestation graded: verified (gaige re-hashes")
    print("            the content-addressed blob) / self-reported (digest) / opaque")
    for var in ("GAIGE_AI_ENDPOINT", "GAIGE_AI_MODEL"):
        val = os.environ.get(var)
        print(f"{var}={val}" if val else f"{var} (unset)")
    return 0


def cmd_test_connection(args) -> int:
    import os
    import time as _time

    from .providers.base import Decoding
    from .providers.llamacpp import LlamaCpp

    endpoint = args.endpoint or os.environ.get("GAIGE_AI_ENDPOINT")
    if not endpoint:
        raise SystemExit("--endpoint (or GAIGE_AI_ENDPOINT) is required")
    p = LlamaCpp(endpoint=endpoint, gguf_path=args.gguf)
    ident = p.connect()
    print(f"[connect] {endpoint}")
    print(f"[connect] attestation: {ident['attestation']} — {ident['attestation_basis']}")
    for k in ("server_model_path", "build_info", "artifact_sha256"):
        if ident.get(k):
            print(f"[connect] {k}: {str(ident[k])[:72]}")
    t0 = _time.time()
    out = p.complete("2+2=", Decoding(max_new_tokens=8))
    print(f"[connect] completion probe ok ({_time.time() - t0:.1f}s): {out.strip()[:40]!r}")
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
    r.add_argument(
        "--detector", default="fast-detect-gpt", choices=["fast-detect-gpt", "binoculars"]
    )
    r.add_argument(
        "--model",
        default=None,
        help="scoring model (fast-detect-gpt); default is chosen per device",
    )
    r.add_argument(
        "--observer",
        default="tiiuae/falcon-7b",
        help="binoculars observer model (must share a tokenizer with the performer)",
    )
    r.add_argument(
        "--performer",
        default="tiiuae/falcon-7b-instruct",
        help="binoculars performer model",
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
    r.add_argument(
        "--harm-volume",
        type=int,
        default=analyze_mod.HARM_VOLUME_DEFAULT,
        help="documents/year for the base-rate harm line "
        "(default: Vanderbilt's published 75,000 submissions/year)",
    )
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
        "--harm-volume",
        type=int,
        default=analyze_mod.HARM_VOLUME_DEFAULT,
        help="documents/year for the base-rate harm line "
        "(default: Vanderbilt's published 75,000 submissions/year)",
    )
    a.add_argument(
        "--out", help="output directory (default: a new timestamped dir beside the source)"
    )
    a.set_defaults(fn=cmd_analyze)

    c = sub.add_parser("corpora", help="list built-in corpora")
    c.set_defaults(fn=cmd_corpora)

    co = sub.add_parser("corpus", help="corpus tools: prepare slices from public benchmarks")
    co_sub = co.add_subparsers(dest="corpus_cmd", required=True)
    cor = co_sub.add_parser(
        "prepare-raid",
        help="prepare a seeded RAID slice (Dugan et al., ACL 2024) into corpora/ — "
        "fetched at run time, never redistributed",
    )
    cor.add_argument(
        "--generators",
        default="gpt4,mistral-chat",
        help="comma list of RAID generator names (the model column); "
        "'human' is always sampled per domain regardless",
    )
    cor.add_argument("--domains", default="abstracts,reddit", help="comma list of RAID domains")
    cor.add_argument(
        "--attacks",
        default="none",
        help="comma list of RAID attack names applied to the AI cells (humans are attack=none)",
    )
    cor.add_argument("--per-cell", type=int, default=60, help="rows per sampling cell")
    cor.add_argument("--seed", type=int, default=17)
    cor.add_argument("--min-words", type=int, default=50)
    cor.add_argument("--max-words", type=int, default=500)
    cor.add_argument(
        "--source",
        choices=["hub", "csv"],
        default="hub",
        help="hub = page filtered rows from datasets-server (no big download); "
        "csv = stream a locally downloaded RAID csv",
    )
    cor.add_argument("--csv", help="path to the downloaded RAID csv (with --source csv)")
    cor.add_argument("--root", default=".", help="gaige root (slice lands in <root>/corpora/)")
    cor.set_defaults(fn=cmd_corpus_prepare_raid)

    pr = sub.add_parser("probe", help="probe runner: dated probes -> answers -> graded receipt")
    pr_sub = pr.add_subparsers(dest="probe_cmd", required=True)
    prr = pr_sub.add_parser(
        "run",
        help="run a probe set against a model provider and emit an accuracy-by-vintage receipt",
    )
    prr.add_argument("--probes", required=True, help="probe-set JSONL (see gaige/probes.py schema)")
    prr.add_argument(
        "--provider",
        default="local-hf",
        choices=["local-hf", "llamacpp", "ollama"],
        help="local-hf = in-process (attestation verified); llamacpp = server via /v1",
    )
    prr.add_argument("--model", default=None, help="model id (local-hf) or served name (llamacpp)")
    prr.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    prr.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    prr.add_argument("--endpoint", default=None, help="llamacpp server URL (or GAIGE_AI_ENDPOINT)")
    prr.add_argument("--gguf", default=None, help="local GGUF path; enables verified attestation")
    prr.add_argument(
        "--cutoff",
        required=True,
        help="model training cutoff (YYYY-MM-DD); receipts print per-vintage post-cutoff share",
    )
    prr.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 = greedy, the pre-registered study default",
    )
    prr.add_argument("--top-p", type=float, default=1.0)
    prr.add_argument("--top-k", type=int, default=0)
    prr.add_argument(
        "--gen-seed", type=int, default=None, help="sampling seed (ignored at temperature 0)"
    )
    prr.add_argument("--max-new-tokens", type=int, default=64)
    prr.add_argument("--n-boot", type=int, default=1000)
    prr.add_argument("--seed", type=int, default=17, help="bootstrap seed")
    prr.add_argument("--root", default=".", help="project root holding reports/ and registry/")
    prr.add_argument("--resume", help="continue an interrupted probe run directory")
    prr.add_argument(
        "--allow-remote-text",
        action="store_true",
        help="explicit opt-in to send prompts to a NON-local endpoint; never the default",
    )
    prr.add_argument(
        "--ptrue",
        action="store_true",
        help="also measure M3: Kadavath-style P(True) per answer (needs a provider with "
        "option_logprobs; the P(True) template joins the instrument fingerprint)",
    )
    prr.add_argument(
        "--register",
        action="store_true",
        help="record the completed run(s) in the series registry (instrument-keyed)",
    )
    prr.add_argument(
        "--registry", default=None, help="registry directory (default: <root>/registry)"
    )
    prr.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="run the probe set k times (the Day-0 variance-bound protocol); implies "
        "replicate-tagged runs when k>1",
    )
    prr.set_defaults(fn=cmd_probe_run)

    prn = pr_sub.add_parser(
        "new",
        help="scaffold a probe-set template + manifest (the signed authoring decisions "
        "pre-filled; placeholders fail lint until authored)",
    )
    prn.add_argument("--out", required=True, help="path for the new probe-set JSONL")
    prn.add_argument("--vintage", default="t0", help="vintage label for the template rows")
    prn.add_argument(
        "--cutoff",
        required=True,
        help="model training cutoff (YYYY-MM-DD) the set is authored against; every "
        "source_date must post-date it",
    )
    prn.set_defaults(fn=cmd_probe_new)

    prl = pr_sub.add_parser(
        "lint",
        help="lint a probe set + manifest against the signed probe-authoring decisions "
        "(provenance fields, nem grading declaration, greedy decoding, control linkage)",
    )
    prl.add_argument(
        "--probes", required=True, help="probe-set JSONL (manifest read from beside it)"
    )
    prl.set_defaults(fn=cmd_probe_lint)

    se = sub.add_parser("series", help="the run registry: list series, show a series report")
    se_sub = se.add_subparsers(dest="series_cmd", required=True)
    sl = se_sub.add_parser("list", help="list registered series")
    sl.add_argument("--registry", default="registry")
    sl.set_defaults(fn=cmd_series)
    ss = se_sub.add_parser("show", help="print a series report")
    ss.add_argument("id")
    ss.add_argument("--registry", default="registry")
    ss.set_defaults(fn=cmd_series)
    sw = se_sub.add_parser(
        "watch", help="M5: run the drift-monitor panel over a series (post-hoc replay)"
    )
    sw.add_argument("id")
    sw.add_argument("--registry", default="registry")
    sw.add_argument("--vintage", default=None, help="default: every vintage in the series")
    sw.add_argument("--quantity", default="accuracy", choices=["accuracy", "gap"])
    sw.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="conformal per-interval false-alarm bound; needs ceil(1/alpha)-1 zero-drift "
        "reference intervals (0.2 -> 4). Refuses honestly below that.",
    )
    sw.add_argument(
        "--direction",
        default=None,
        choices=["down", "up"],
        help="default: down for accuracy, up for gap",
    )
    sw.set_defaults(fn=cmd_series)

    pl = sub.add_parser(
        "plan",
        help="what can THIS machine run, at what measured cost, with what attestation "
        "(feasibility only — quality lives in receipts)",
    )
    pl.set_defaults(fn=cmd_plan)

    pv = sub.add_parser("providers", help="list model providers and environment configuration")
    pv.set_defaults(fn=cmd_providers)

    tc = sub.add_parser(
        "test-connection", help="prove a llamacpp endpoint answers BEFORE a long run"
    )
    tc.add_argument("--endpoint", default=None, help="server URL (or GAIGE_AI_ENDPOINT)")
    tc.add_argument("--gguf", default=None, help="local GGUF path to attest against")
    tc.set_defaults(fn=cmd_test_connection)

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
    try:
        return args.fn(args)
    except ModuleNotFoundError as e:
        if e.name in ("torch", "transformers", "bitsandbytes", "accelerate"):
            print(
                f"error: scoring needs the GPU extra ({e.name} is not installed).\n"
                'Install it with:  pip install "gaige[gpu]"\n'
                "Analysis commands (analyze, plan, corpora, series) run without it.",
                file=sys.stderr,
            )
            return 2
        raise
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        # Refusals are expected outputs, not crashes: print the message, keep the
        # traceback for genuinely unexpected error types only.
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
