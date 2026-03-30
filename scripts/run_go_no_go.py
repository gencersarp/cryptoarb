#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", default="../results_robust_eval.json")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()

    p = Path(args.input_json)
    if not p.exists():
        if args.allow_missing:
            print(f"NO-GO: missing input file {p}")
            raise SystemExit(1)
        raise FileNotFoundError(f"No such file: {p}")
    payload = json.loads(p.read_text())
    best = payload.get("best_eval", {})
    wf_results = best.get("wf_results", [])

    if not wf_results:
        print("NO-GO: no walk-forward results found.")
        raise SystemExit(1)

    failed = []
    for r in wf_results:
        asset = r.get("asset", "UNKNOWN")
        checks = r.get("pass_fail", {}).get("checks", {})
        bad = [k for k, v in checks.items() if not v]
        if bad:
            failed.append((asset, bad))

    if failed:
        print("NO-GO: robustness checks failed")
        for asset, bad in failed:
            print(f"  {asset}: {', '.join(bad)}")
        raise SystemExit(1)

    print("GO: all mandatory robustness checks passed across assets.")


if __name__ == "__main__":
    main()
