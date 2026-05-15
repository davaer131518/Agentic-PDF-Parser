"""Single focused smoke-test run.

Usage:
    python smoke_tests/run_one.py
    python smoke_tests/run_one.py --device cpu
    python smoke_tests/run_one.py --pdf examples/pdfs/table_heavy_world_bank.pdf
    python smoke_tests/run_one.py --pdf examples/pdfs/table_heavy_world_bank.pdf --debug
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_PDF = REPO_ROOT / "examples" / "pdfs" / "simple_adobe_demo.pdf"
RESULTS_DIR = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="paddle_vl")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--pdf", default=None, help="Path to PDF file (default: simple_adobe_demo.pdf)")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--dpi", default=None, type=int)
    parser.add_argument("--out", default=None, help="Override output directory")
    parser.add_argument("--debug", action="store_true", help="Save bbox-annotated debug images")
    args = parser.parse_args()

    pdf_path = Path(args.pdf) if args.pdf else _DEFAULT_PDF
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        raise SystemExit(1)

    pdf_name = pdf_path.stem          # e.g. "complex_attention_is_all_you_need"
    run_name = f"{args.backend}_{args.device}"
    out_dir = Path(args.out) if args.out else RESULTS_DIR / pdf_name / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "agentic_pdf_parser",
        str(pdf_path),
        "--backend", args.backend,
        "--device", args.device,
        "--out", str(out_dir),
        "--keep-raw",
    ]
    if args.config:
        cmd += ["--config", args.config]
    if args.dpi:
        cmd += ["--dpi", str(args.dpi)]
    if args.debug:
        cmd += ["--debug"]

    print(f"\nRUN : {run_name}")
    print(f"CMD : {' '.join(cmd)}\n")

    log_path = out_dir / "run.log"
    start = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            cwd=str(REPO_ROOT),
        )
        log_fh.write(proc.stdout or "")

    elapsed = time.perf_counter() - start
    success = proc.returncode == 0

    print(proc.stdout or "")
    status = "SUCCESS" if success else f"FAILED (exit {proc.returncode})"
    print(f"\n{'='*50}")
    print(f"Status  : {status}")
    print(f"Elapsed : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"Output  : {out_dir}")

    result = {
        "run_name": run_name,
        "backend": args.backend,
        "device": args.device,
        "pdf": str(pdf_path),
        "config_file": args.config,
        "dpi_override": args.dpi,
        "success": success,
        "elapsed_seconds": round(elapsed, 2),
        "output_dir": str(out_dir),
        "error": None if success else f"exit code {proc.returncode}",
    }

    timing_path = out_dir / "timing.json"
    timing_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Timing  : {timing_path}")


if __name__ == "__main__":
    main()
