"""Run selected sleeve-splice experiments on locked 2015-2019 OOS dates.

The supervisor validates the frozen in-sample selection, prepares scorer
commands, and analyzes only the predeclared out-of-sample experiment pair.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import mean_reversion_llm_v2 as runner
import mean_reversion_llm_v2_sleeve_splice_experiments as experiments


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "backtest_results/v2_sleeve_splice_oos_2015_2020"
SUITE = OUTPUT / "suite_manifest.json"
SOURCE_CACHE = next(
    (ROOT / "backtest_results/llm_reversal_signal_v2_phase2_cache").glob("*.json")
)
ARMS = ("original_v2", "b_deprescribed", "b_original_range")
START = "2015-01-01"
END = "2020-01-01"
DATES = 251
CALL_CAP = 275
TOKEN_CAP = 1_250_000
CONFIRMATION = "RUN_2_OOS_TESTS_753_CALLS"


class OOSSuiteError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OOSSuiteError(f"Expected JSON object: {path}")
    return value


def frozen_dates() -> list[str]:
    payload = read_json(SOURCE_CACHE)
    signals = payload.get("signals")
    if not isinstance(signals, dict):
        raise OOSSuiteError("Archival OOS cache has no signals mapping")
    dates = sorted({str(key).split("|", 1)[0] for key in signals})
    if len(dates) != DATES or dates[0] != "2015-01-06" or dates[-1] != "2019-12-23":
        raise OOSSuiteError("Archival OOS date lock is not the expected 251-date window")
    return dates


def command(output: Path, arm: str, prompt_spec: Path, manifest: Path) -> list[str]:
    arm_root = output / "scorers" / arm
    validation_freeze = output / "validation_freeze.json"
    return [
        sys.executable,
        str(Path(runner.__file__).resolve()),
        "--phase", "llm-cache",
        "--start", START,
        "--end", END,
        "--output-dir", str((arm_root / "run").resolve()),
        "--cache-dir", str((arm_root / "cache").resolve()),
        "--experimental-prompt-spec", str(prompt_spec.resolve()),
        "--signal-dates-file", str(manifest.resolve()),
        "--validation-freeze", str(validation_freeze.resolve()),
        "--progress-file", str((arm_root / "runner_events.jsonl").resolve()),
        "--schedule", "anchors",
        "--anchor-offsets", "3",
        "--pool-sizes", "60",
        "--book-sizes", "30",
        "--llm-pool-size", "60",
        "--model", "deepseek-v4-flash",
        "--temperature", "1",
        "--thinking-mode", "disabled",
        "--anonymize-tickers",
        "--cost-bps-levels", "0",
        "--rate-limit", "2",
        "--run-output-token-budget", str(TOKEN_CAP),
        "--run-call-budget", str(CALL_CAP),
        "--llm-max-output-tokens", "4096",
        "--llm-max-attempts", "2",
        "--experimental-override",
    ]


def parsed_args(frozen_command: Sequence[str]) -> argparse.Namespace:
    return runner.parse_args([str(item) for item in frozen_command][2:])


def cache_path(frozen_command: Sequence[str]) -> Path:
    return runner.cache_path_for_args(parsed_args(frozen_command)).resolve()


def prepare(output: Path = OUTPUT) -> dict[str, Any]:
    output = output.resolve()
    dates = frozen_dates()
    date_manifest = output / "oos_signal_dates.json"
    write_json(
        date_manifest,
        {
            "campaign_id": "sleeve_splice_locked_oos_2015_2020",
            "date_count": len(dates),
            "dates": dates,
            "period": {"start": START, "end_exclusive": END},
            "source": str(SOURCE_CACHE.resolve()),
        },
    )
    specs = experiments.write_prompt_specs(output / "prompt_specs")
    labels = {
        "original_v2": "Experiment 1 source — Original V2",
        "b_deprescribed": "Experiment 1 source — B",
        "b_original_range": "Experiment 2 — B Context + V2 Range",
    }
    commands: dict[str, list[str]] = {}
    arms: dict[str, Any] = {}
    for arm in ARMS:
        frozen_command = command(output, arm, specs[arm], date_manifest)
        commands[arm] = frozen_command
        spec = runner.load_experimental_prompt_spec(specs[arm])
        arms[arm] = {
            "label": labels[arm],
            "prompt_spec": str(specs[arm]),
            "prompt_sha256": spec.prompt_sha256,
            "candidate_id": spec.candidate_id,
            "cache": str(cache_path(frozen_command)),
            "nominal_calls": DATES,
            "hard_call_attempt_cap": CALL_CAP,
        }
    loaded_manifest = runner.load_signal_date_manifest(date_manifest)
    development_results = (
        ROOT
        / "backtest_results/v2_sleeve_splice_in_sample_2009_2014/analysis/experiment_scorecard.csv"
    ).resolve()
    if not development_results.is_file():
        raise OOSSuiteError(f"Missing frozen development scorecard: {development_results}")
    lineage = [
        {
            "candidate_id": "SLEEVE_SPLICE_SELECTED_OOS",
            "generation": "manual_phase2_selection",
            "parent_id": None,
            "prompt_hash": "multiple_allowlisted_specs",
        }
    ]
    canonical_lineage = json.dumps(
        lineage, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    allowed_specs = []
    for arm in ARMS:
        spec = runner.load_experimental_prompt_spec(Path(arms[arm]["prompt_spec"]))
        allowed_specs.append(
            {
                "role": "challenger",
                "candidate_id": spec.candidate_id,
                "side_scope": spec.side_scope,
                "replicate_id": spec.replicate_id,
                "prompt_hash": spec.prompt_sha256,
                "prompt_spec_hash": spec.spec_sha256,
                "harness_fingerprint": runner.autoresearch_harness_fingerprint(
                    parsed_args(commands[arm])
                ),
            }
        )
    freeze = {
        "schema_version": 1,
        "approved": True,
        "approval_status": "approved",
        "approved_at": now(),
        "approved_by": "user_oos_instruction_2026-08-26",
        "campaign_id": "sleeve_splice_selected_oos_2015_2020",
        "allowed_prompt_specs": allowed_specs,
        "anonymize_tickers": True,
        "side_scope": "both",
        "date_manifest_hash": loaded_manifest.date_manifest_hash,
        "date_manifest_file_hash": loaded_manifest.source_sha256,
        "lineage": lineage,
        "lineage_hash": hashlib.sha256(canonical_lineage).hexdigest(),
        "supervisor_code_fingerprint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "code_fingerprint": runner.autoresearch_code_fingerprint(),
        "development_results_path": str(development_results),
        "development_results_hash": hashlib.sha256(development_results.read_bytes()).hexdigest(),
        "selection": {
            "advance": [
                "Experiment 1 — Original V2 Long + B Short — Two-Call Splice",
                "Experiment 2 — B Context + Original V2 Range",
            ],
            "reject": "Experiment 3 — One-Call Asymmetric Hybrid",
            "selected_on_period": "development_2009_2013",
        },
    }
    write_json(output / "validation_freeze.json", freeze)
    suite = {
        "schema_version": 1,
        "artifact_type": "v2_sleeve_splice_oos_suite",
        "period": "locked_oos_2015_2019",
        "start": START,
        "end_exclusive": END,
        "signal_dates": dates,
        "signal_manifest": str(date_manifest),
        "signal_date_count": DATES,
        "arms": arms,
        "commands": commands,
        "scorer_legs": len(ARMS),
        "portfolio_experiments": 2,
        "nominal_paid_calls": len(ARMS) * DATES,
        "hard_aggregate_call_attempt_cap": len(ARMS) * CALL_CAP,
        "paid_confirmation_required": CONFIRMATION,
        "harness": {
            "model": "deepseek-v4-flash",
            "temperature": 1.0,
            "thinking_mode": "disabled",
            "anonymous_tickers": True,
            "pool_per_side": 60,
            "positions_per_side": 30,
            "anchor_offset": 3,
            "capital_split": "50/50",
            "cost_bps": [0.0],
        },
        "dashboard_target": str(
            (ROOT / "backtest_results/llm_reversal_signal_v2_2009_2014/phase2_validation_dashboard.html").resolve()
        ),
    }
    write_json(output / "suite_manifest.json", suite)
    write_json(output / "readiness_report.json", status(output / "suite_manifest.json"))
    return suite


def load_suite(path: Path) -> dict[str, Any]:
    suite = read_json(path.resolve())
    if suite.get("artifact_type") != "v2_sleeve_splice_oos_suite":
        raise OOSSuiteError("Not the sleeve-splice OOS suite")
    if set(suite.get("commands", {})) != set(ARMS) or set(suite.get("arms", {})) != set(ARMS):
        raise OOSSuiteError("OOS suite must contain exactly the three selected scorer legs")
    if suite.get("signal_date_count") != DATES or suite.get("signal_dates") != frozen_dates():
        raise OOSSuiteError("OOS date lock changed")
    return suite


def arm_status(suite: dict[str, Any], arm: str) -> dict[str, Any]:
    path = Path(suite["arms"][arm]["cache"])
    if not path.exists():
        return {"arm": arm, "status": "not_started", "completed_dates": 0, "remaining_dates": DATES}
    payload = read_json(path)
    signals = payload.get("signals", {})
    completed = len(signals) if isinstance(signals, dict) else 0
    return {
        "arm": arm,
        "status": "complete" if completed == DATES else "partial_resumable",
        "completed_dates": completed,
        "remaining_dates": DATES - completed,
        "cache": str(path),
    }


def status(path: Path = SUITE) -> dict[str, Any]:
    suite = load_suite(path) if path.exists() else prepare(path.parent)
    arms = [arm_status(suite, arm) for arm in ARMS]
    completed = sum(item["completed_dates"] for item in arms)
    return {
        "status": "scoring_complete" if completed == len(ARMS) * DATES else "ready_or_running",
        "completed_nominal_calls": completed,
        "remaining_nominal_calls": len(ARMS) * DATES - completed,
        "nominal_paid_calls": len(ARMS) * DATES,
        "arms": arms,
    }


def analyze(path: Path) -> dict[str, Any]:
    suite = load_suite(path)
    payloads = {}
    for arm in ARMS:
        item = arm_status(suite, arm)
        if item["status"] != "complete":
            raise OOSSuiteError(f"Cannot analyze incomplete arm: {arm}")
        payloads[arm] = read_json(Path(suite["arms"][arm]["cache"]))
    manifest = runner.load_signal_date_manifest(Path(suite["signal_manifest"]))
    config = runner.build_config(parsed_args(suite["commands"][ARMS[0]]))
    return experiments.evaluate_completed_suite(
        output_dir=path.parent,
        cache_payloads=payloads,
        signal_manifest=manifest,
        config=config,
    )


def run_paid(path: Path, confirmation: str) -> dict[str, Any]:
    if confirmation != CONFIRMATION:
        raise OOSSuiteError(f"Paid authorization requires {CONFIRMATION!r}")
    suite = load_suite(path)
    state_path = path.parent / "execution_state.json"
    state = read_json(state_path) if state_path.exists() else {"events": [], "arms": {}}
    state.update({"status": "running", "pid": os.getpid(), "updated_at": now()})
    write_json(state_path, state)
    for arm in ARMS:
        before = arm_status(suite, arm)
        if before["status"] == "complete":
            state["arms"][arm] = before
            continue
        log = path.parent / "scorers" / arm / "supervisor_runner.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        state.update({"current_arm": arm, "updated_at": now()})
        write_json(state_path, state)
        with log.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(suite["commands"][arm], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        after = arm_status(suite, arm)
        state["arms"][arm] = {**after, "returncode": completed.returncode}
        write_json(state_path, state)
        if completed.returncode or after["status"] != "complete":
            state.update({"status": "paused_resumable", "current_arm": None, "updated_at": now()})
            write_json(state_path, state)
            raise OOSSuiteError(f"{arm} paused; cache is safe to resume. See {log}")
    state.update({"status": "analyzing", "current_arm": None, "updated_at": now()})
    write_json(state_path, state)
    result = analyze(path)
    state.update({"status": "completed", "analysis": result, "completed_at": now(), "updated_at": now()})
    write_json(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output-dir", type=Path, default=OUTPUT)
    stat = sub.add_parser("status")
    stat.add_argument("--suite", type=Path, default=SUITE)
    paid = sub.add_parser("run-paid")
    paid.add_argument("--suite", type=Path, default=SUITE)
    paid.add_argument("--confirmation", required=True)
    offline = sub.add_parser("analyze")
    offline.add_argument("--suite", type=Path, default=SUITE)
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.output_dir), indent=2, sort_keys=True))
    elif args.action == "status":
        print(json.dumps(status(args.suite), indent=2, sort_keys=True))
    elif args.action == "run-paid":
        print(json.dumps(run_paid(args.suite, args.confirmation), indent=2, sort_keys=True))
    else:
        print(json.dumps(analyze(args.suite), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
