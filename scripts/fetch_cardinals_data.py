from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib import request
from urllib.error import URLError, HTTPError

import pandas as pd

from clean_data import clean_cap_data


BASE_URL = "https://www.spotrac.com/nfl/arizona-cardinals/overview/_/year/{year}"
REQUIRED_COLUMNS = ["Age", "Cap Hit", "Dead Cap", "Cash Total"]


def fetch_html(url: str, timeout: int = 30) -> str:
    req = request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def validate_and_process(raw_path: Path, processed_path: Path, year: int) -> tuple[bool, str]:
    try:
        tables = pd.read_html(raw_path)
    except Exception as exc:
        return False, f"{year}: parse failed ({exc})"

    if not tables:
        return False, f"{year}: no tables found"

    contracts = tables[0].copy()
    contracts.columns = contracts.columns.str.strip()

    missing = [c for c in REQUIRED_COLUMNS if c not in contracts.columns]
    if missing:
        return False, f"{year}: missing required columns {missing}"

    contracts = clean_cap_data(contracts)
    contracts["Age"] = pd.to_numeric(contracts["Age"], errors="coerce")

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    contracts.to_csv(processed_path, index=False)

    return True, f"{year}: processed {len(contracts)} rows"


def run(
    teams: list[str],
    years: list[int],
    raw_dir: Path,
    processed_dir: Path,
    timeout: int,
    delay_seconds: float,
    skip_existing: bool,
    process: bool,
    dry_run: bool,
) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    failures = 0

    jobs = [(team, year) for team in teams for year in years]

    for idx, (team, year) in enumerate(jobs):
        url = f"https://www.spotrac.com/nfl/{team}/overview/_/year/{year}"
        raw_path = raw_dir / f"{team}_overview_{year}.html"
        processed_path = processed_dir / f"{team}_overview_{year}.csv"

        if dry_run:
            print(f"DRY-RUN {team} {year}: {url} -> {raw_path}")
            continue

        if skip_existing and raw_path.exists():
            print(f"SKIP {team} {year}: raw file exists ({raw_path})")
        else:
            try:
                html = fetch_html(url, timeout=timeout)
                raw_path.write_text(html, encoding="utf-8")
                print(f"OK {team} {year}: saved raw html to {raw_path}")
            except (HTTPError, URLError, TimeoutError) as exc:
                failures += 1
                print(f"FAIL {team} {year}: fetch error ({exc})")
                if idx < len(jobs) - 1:
                    time.sleep(delay_seconds)
                continue
            except Exception as exc:
                failures += 1
                print(f"FAIL {team} {year}: unexpected fetch error ({exc})")
                if idx < len(jobs) - 1:
                    time.sleep(delay_seconds)
                continue

        if process:
            ok, message = validate_and_process(raw_path, processed_path, year)
            print(("OK " if ok else "FAIL ") + message)
            if not ok:
                failures += 1

        if idx < len(jobs) - 1:
            time.sleep(delay_seconds)

    if failures:
        print(f"\nCompleted with {failures} failure(s).")
        return 1

    print("\nCompleted successfully.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Spotrac overview pages by team/year and save raw+processed data."
    )
    parser.add_argument("--teams", nargs="+", type=str, default=["arizona-cardinals"])
    parser.add_argument("--years", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--delay-seconds", type=float, default=1.5)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-process", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(
        teams=args.teams,
        years=args.years,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        timeout=args.timeout,
        delay_seconds=args.delay_seconds,
        skip_existing=args.skip_existing,
        process=not args.no_process,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
