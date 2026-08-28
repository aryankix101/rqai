"""Finalize the development-only prompt-interaction research package.

This command waits for the declared suite to finish, runs its offline analysis,
and refreshes the consolidated package without starting new model requests.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE = PROJECT_ROOT / "backtest_results/research_thesis_package_20260808"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def wait_and_finalize(package_root: Path, timeout_hours: float) -> dict[str, Any]:
    package_root = Path(package_root).resolve()
    interaction = package_root / "prompt_interaction"
    execution_path = interaction / "execution_state.json"
    analysis_path = interaction / "analysis/summary.json"
    watcher_path = interaction / "finalizer_state.json"
    deadline = time.monotonic() + timeout_hours * 3600.0
    write_json(
        watcher_path,
        {"status": "waiting", "started_at": utc_now(), "blind_period_touched": False},
    )
    while time.monotonic() < deadline:
        analysis = read_json(analysis_path)
        if analysis.get("status") == "completed" and not analysis.get(
            "blind_period_touched", True
        ):
            break
        execution = read_json(execution_path)
        if execution.get("status") == "failed":
            result = {
                "status": "failed",
                "reason": "prompt interaction execution failed",
                "updated_at": utc_now(),
                "blind_period_touched": False,
            }
            write_json(watcher_path, result)
            return result
        time.sleep(30.0)
    else:
        result = {
            "status": "timed_out",
            "updated_at": utc_now(),
            "blind_period_touched": False,
        }
        write_json(watcher_path, result)
        return result

    log_path = interaction / "finalizer.log"
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(package_root.parent / ".mplconfig")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "mean_reversion_research_diagnostics.py"),
        "finalize",
        "--output-dir",
        str(package_root),
        "--n-boot",
        "5000",
    ]
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = {
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "updated_at": utc_now(),
        "blind_period_touched": False,
        "log": str(log_path),
    }
    write_json(watcher_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    args = parser.parse_args()
    print(json.dumps(wait_and_finalize(args.package_root, args.timeout_hours), indent=2))


if __name__ == "__main__":
    main()
