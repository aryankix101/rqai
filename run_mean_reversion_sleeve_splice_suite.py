"""Prepare, guard, run, and finalize the three in-sample sleeve tests.

Safe commands (``prepare``, ``validate``, and ``status``) never start a model
request.  The only command that may invoke DeepSeek is ``run-paid``, and it
requires an exact acknowledgement string.  All four scorer legs are resumable
from per-date immutable caches; portfolio construction and gross analysis are
performed offline after scoring completes. Transaction-cost analysis is deferred
until the later combined in-sample/OOS review.
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
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import mean_reversion_llm_v1 as v1
import mean_reversion_llm_v2 as runner
import mean_reversion_llm_v2_sleeve_splice_experiments as experiments
import technical_llm


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backtest_results/v2_sleeve_splice_in_sample_2009_2014"
DEFAULT_SUITE = DEFAULT_OUTPUT_DIR / "suite_manifest.json"
DEFAULT_SIGNAL_MANIFEST = (
    PROJECT_ROOT
    / "backtest_results/autoresearch_campaigns/"
    "prompt_search_ticker_optional_resumed_20260806/manifests/full.json"
)
SCORER_ARMS = (
    "original_v2",
    "b_deprescribed",
    "b_original_range",
    "asymmetric_hybrid",
)
PAID_CONFIRMATION = "RUN_3_IN_SAMPLE_TESTS_1004_CALLS"
NOMINAL_CALLS_PER_ARM = 251
CALL_BUDGET_PER_ARM = 275
OUTPUT_TOKEN_BUDGET_PER_ARM = 1_250_000
MAX_OUTPUT_TOKENS_PER_CALL = 4_096
EXPECTED_DATE_MANIFEST_HASH = "7a8b1e7fc7999c8cab709c1d5c2a539134be0c04b495906ac9c70454f1c2c2b7"
EXPECTED_DATE_FILE_HASH = "dae2d678b179a809c2af60f7865f858e7488e6ff352c5fd3ed675caa8346af88"


class SuiteError(RuntimeError):
    """A suite contract, cache, or paid-run safety check failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SuiteError(f"Expected a JSON object at {path}")
    return payload


def _arg_value(command: Sequence[str], flag: str) -> str:
    if flag not in command:
        raise SuiteError(f"Frozen command is missing {flag}")
    index = list(command).index(flag)
    if index + 1 >= len(command):
        raise SuiteError(f"Frozen command has no value for {flag}")
    return str(command[index + 1])


def _runner_args(command: Sequence[str]) -> argparse.Namespace:
    command = [str(value) for value in command]
    if len(command) < 3 or Path(command[1]).resolve() != Path(runner.__file__).resolve():
        raise SuiteError("Frozen scorer command does not invoke mean_reversion_llm_v2.py")
    return runner.parse_args(command[2:])


def _cache_path_from_command(command: Sequence[str]) -> Path:
    return runner.cache_path_for_args(_runner_args(command)).resolve()


def _scorer_command(
    *,
    output_dir: Path,
    arm: str,
    prompt_spec: Path,
    signal_manifest: Path,
) -> list[str]:
    arm_root = Path(output_dir) / "scorers" / arm
    return [
        sys.executable,
        str(Path(runner.__file__).resolve()),
        "--phase",
        "llm-cache",
        "--start",
        "2009-01-01",
        "--end",
        "2014-01-01",
        "--output-dir",
        str((arm_root / "run").resolve()),
        "--cache-dir",
        str((arm_root / "cache").resolve()),
        "--experimental-prompt-spec",
        str(Path(prompt_spec).resolve()),
        "--signal-dates-file",
        str(Path(signal_manifest).resolve()),
        "--progress-file",
        str((arm_root / "runner_events.jsonl").resolve()),
        "--schedule",
        "anchors",
        "--anchor-offsets",
        "3",
        "--pool-sizes",
        "60",
        "--book-sizes",
        "30",
        "--llm-pool-size",
        "60",
        "--model",
        "deepseek-v4-flash",
        "--temperature",
        "1",
        "--thinking-mode",
        "disabled",
        "--anonymize-tickers",
        "--cost-bps-levels",
        "0",
        "--rate-limit",
        "2",
        "--run-output-token-budget",
        str(OUTPUT_TOKEN_BUDGET_PER_ARM),
        "--run-call-budget",
        str(CALL_BUDGET_PER_ARM),
        "--llm-max-output-tokens",
        str(MAX_OUTPUT_TOKENS_PER_CALL),
        "--llm-max-attempts",
        "2",
        "--experimental-override",
    ]


def _runbook_text(suite_path: Path) -> str:
    script = Path(__file__).resolve()
    python = Path(sys.executable).resolve()
    return f"""# Three in-sample sleeve tests

This suite contains three portfolio experiments and four fresh scorer legs:

1. Original V2 long scores + B short scores (two full two-sided calls per date).
2. B context with only its range rule changed to Original V2's rule.
3. One-call asymmetric Original-V2-long / B-short prompt.

Frozen development window: 251 signal dates, 2009-01-06 through 2013-12-23.
Harness: 60 long + 60 short candidates, top 30 per side, equal side weights,
50/50 capital normalization, anchor 3, five-trading-day C2C, anonymous tickers,
DeepSeek V4 Flash, temperature 1, thinking disabled.

Safe readiness check (no model calls):

```sh
{python} {script} validate --suite {suite_path} --verify-inputs
```

Paid execution (1,004 nominal calls; 1,100 hard aggregate attempt cap):

```sh
{python} {script} run-paid --suite {suite_path} --confirmation {PAID_CONFIRMATION}
```

The paid command is resumable. After all scorer caches are complete, it replays
the three portfolios offline and writes gross results only. Transaction-cost
analysis is deferred until the later combined in-sample/OOS review. No LLM
credits are used for portfolio construction or analysis.
"""


def prepare_suite(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    signal_manifest: Path = DEFAULT_SIGNAL_MANIFEST,
    freeze_inputs: bool = False,
) -> dict[str, Any]:
    """Write immutable specs/commands. No subprocess or HTTP client is started."""
    output_dir = Path(output_dir).resolve()
    signal_manifest = Path(signal_manifest).resolve()
    experiments.assert_prompt_contracts()
    manifest = runner.load_signal_date_manifest(signal_manifest)
    if manifest.date_manifest_hash != EXPECTED_DATE_MANIFEST_HASH:
        raise SuiteError("The frozen 251-date development manifest hash has changed")
    if manifest.source_sha256 != EXPECTED_DATE_FILE_HASH:
        raise SuiteError("The frozen development manifest file hash has changed")
    if len(manifest.dates) != NOMINAL_CALLS_PER_ARM:
        raise SuiteError(f"Expected 251 development dates, got {len(manifest.dates)}")

    spec_paths = experiments.write_prompt_specs(output_dir / "prompt_specs")
    spec_payloads = experiments.prompt_spec_payloads()
    labels = {
        "original_v2": "Experiment 1 source — full Original V2 scorer",
        "b_deprescribed": "Experiment 1 source — full B scorer",
        "b_original_range": "Experiment 2 — B Context + Original V2 Range",
        "asymmetric_hybrid": "Experiment 3 — One-Call Asymmetric V2-Long / B-Short Hybrid",
    }
    arms: dict[str, Any] = {}
    commands: dict[str, list[str]] = {}
    for arm in SCORER_ARMS:
        command = _scorer_command(
            output_dir=output_dir,
            arm=arm,
            prompt_spec=spec_paths[arm],
            signal_manifest=signal_manifest,
        )
        commands[arm] = command
        parsed_spec = runner.load_experimental_prompt_spec(spec_paths[arm])
        arms[arm] = {
            "label": labels[arm],
            "prompt_spec": str(spec_paths[arm]),
            "prompt_spec_sha256": parsed_spec.spec_sha256,
            "prompt_sha256": parsed_spec.prompt_sha256,
            "candidate_id": parsed_spec.candidate_id,
            "cache": str(_cache_path_from_command(command)),
            "nominal_calls": NOMINAL_CALLS_PER_ARM,
            "hard_call_attempt_cap": CALL_BUDGET_PER_ARM,
            "fresh_replication": True,
        }

    suite_path = output_dir / "suite_manifest.json"
    suite = {
        "schema_version": 1,
        "artifact_type": "v2_sleeve_splice_in_sample_suite",
        "purpose": "three_predeclared_sleeve_specialization_tests",
        "development_only": True,
        "blind_period_touched": False,
        "paid_execution_started": False,
        "portfolio_experiments": 3,
        "scorer_legs": 4,
        "nominal_paid_calls": len(SCORER_ARMS) * NOMINAL_CALLS_PER_ARM,
        "hard_aggregate_call_attempt_cap": len(SCORER_ARMS) * CALL_BUDGET_PER_ARM,
        "paid_confirmation_required": PAID_CONFIRMATION,
        "signal_manifest": str(signal_manifest),
        "signal_manifest_sha256": manifest.source_sha256,
        "date_manifest_hash": manifest.date_manifest_hash,
        "signal_date_count": len(manifest.dates),
        "signal_date_first": str(manifest.dates[0].date()),
        "signal_date_last": str(manifest.dates[-1].date()),
        "harness": {
            "period": {"start": "2009-01-01", "end_exclusive": "2014-01-01"},
            "model": "deepseek-v4-flash",
            "temperature": 1.0,
            "thinking_mode": "disabled",
            "ticker_visibility": "anonymous_SEC_identifiers",
            "candidate_pool_per_side": 60,
            "selected_per_side": 30,
            "anchor_offset": 3,
            "holding_days": 5,
            "within_side_weighting": "equal",
            "capital_split": {"long": 0.5, "short": 0.5},
            "paid_runner_cost_bps": [0.0],
            "offline_evaluation_cost_bps": [0.0],
            "transaction_cost_analysis": "deferred_until_combined_in_sample_oos_review",
        },
        "experiments": {
            "1": {
                "label": "Original V2 Long + B Short — Two-Call Splice",
                "scorer_legs": ["original_v2", "b_deprescribed"],
                "construction": "Original V2 long_scores + B short_scores; discard the opposite output from each full call.",
                "primary_comparison": "Trail5, with positive paired lift required on both sleeves.",
            },
            "2": {
                "label": "B Context + Original V2 Range",
                "scorer_legs": ["b_original_range"],
                "construction": "One full two-sided call; B differs by exactly one range-rule sentence.",
                "primary_comparison": "Fresh B to isolate the range-rule interaction.",
            },
            "3": {
                "label": "One-Call Asymmetric V2-Long / B-Short Hybrid",
                "scorer_legs": ["asymmetric_hybrid"],
                "construction": "One full call with explicit side-relative, intentionally different philosophies.",
                "primary_comparison": "Experiment 1 to measure one-call preservation.",
            },
        },
        "arms": arms,
        "commands": commands,
        "input_freeze": str((output_dir / "input_freeze.json").resolve()),
        "execution_policy": (
            "All four scorer legs are fresh current-vintage replications. Run sequentially and resume only "
            "from exact prompt/date/input hashes. Never reuse historical caches for the primary comparison. "
            "Portfolio replay is gross-only and offline; transaction-cost analysis is deferred."
        ),
        "historical_cache_policy": {
            "matching_archival_caches_exist": True,
            "reuse_for_primary": False,
            "reason": "Fresh Experiment 1 calls keep the Experiment 1 versus Experiment 3 comparison on one model vintage.",
        },
    }
    write_json(suite_path, suite)
    (output_dir / "RUNBOOK.md").write_text(_runbook_text(suite_path), encoding="utf-8")
    validate_suite(suite_path)
    if freeze_inputs:
        freeze_suite_inputs(suite_path)
    report = suite_status(suite_path)
    report.update({"network_calls_made_during_setup": 0, "paid_calls_made_during_setup": 0})
    write_json(output_dir / "readiness_report.json", report)
    return suite


def read_suite(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    suite = read_json(path)
    if suite.get("artifact_type") != "v2_sleeve_splice_in_sample_suite":
        raise SuiteError("Not a sleeve-splice suite manifest")
    if suite.get("development_only") is not True or suite.get("blind_period_touched") is not False:
        raise SuiteError("Suite is not frozen as development-only")
    if set(suite.get("commands", {})) != set(SCORER_ARMS):
        raise SuiteError("Suite must contain exactly the four frozen scorer legs")
    if set(suite.get("arms", {})) != set(SCORER_ARMS):
        raise SuiteError("Suite arm metadata does not match the four scorer legs")
    return suite


def validate_suite(path: Path) -> dict[str, Any]:
    """Validate specs, date firewall, harness identity, budgets, and path isolation."""
    suite_path = Path(path).resolve()
    suite = read_suite(suite_path)
    signal_manifest = runner.load_signal_date_manifest(Path(suite["signal_manifest"]))
    if signal_manifest.date_manifest_hash != EXPECTED_DATE_MANIFEST_HASH:
        raise SuiteError("Suite date-manifest hash is not the frozen development hash")
    if signal_manifest.source_sha256 != EXPECTED_DATE_FILE_HASH:
        raise SuiteError("Suite date-manifest file hash is not frozen")
    if len(signal_manifest.dates) != NOMINAL_CALLS_PER_ARM:
        raise SuiteError("Suite must use exactly 251 development signal dates")
    if any(date >= runner.AUTORESEARCH_VALIDATION_START for date in signal_manifest.dates):
        raise SuiteError("Suite attempts to access validation dates")

    expected_specs = experiments.prompt_spec_payloads()
    harnesses: list[dict[str, Any]] = []
    cache_paths: list[Path] = []
    call_cap = 0
    for arm in SCORER_ARMS:
        arm_meta = suite["arms"][arm]
        spec_path = Path(arm_meta["prompt_spec"])
        spec = runner.load_experimental_prompt_spec(spec_path)
        if spec.raw != expected_specs[arm]:
            raise SuiteError(f"{arm} prompt spec differs from the frozen requested treatment")
        if spec.spec_sha256 != arm_meta["prompt_spec_sha256"]:
            raise SuiteError(f"{arm} prompt-spec hash drift")
        if spec.prompt_sha256 != arm_meta["prompt_sha256"]:
            raise SuiteError(f"{arm} prompt hash drift")

        command = [str(value) for value in suite["commands"][arm]]
        if "--validation-freeze" in command or _arg_value(command, "--phase") != "llm-cache":
            raise SuiteError(f"{arm} is not a development llm-cache command")
        if _arg_value(command, "--start") != "2009-01-01" or _arg_value(command, "--end") != "2014-01-01":
            raise SuiteError(f"{arm} date arguments drifted")
        args = _runner_args(command)
        args.thinking_mode = technical_llm.effective_thinking_mode(args.model, args.thinking_mode)
        args.side_scope = spec.side_scope
        args.replicate_id = spec.replicate_id
        runner.validate_autoresearch_args(args)
        config = runner.build_config(args)
        if runner.validate_autoresearch_manifest_scope(signal_manifest, config) != "development":
            raise SuiteError(f"{arm} failed the development firewall")
        if Path(args.experimental_prompt_spec).resolve() != spec_path.resolve():
            raise SuiteError(f"{arm} command points at a different prompt spec")
        if Path(args.signal_dates_file).resolve() != Path(suite["signal_manifest"]).resolve():
            raise SuiteError(f"{arm} command points at a different date manifest")
        harnesses.append(runner.autoresearch_harness_manifest(args))
        cache_path = runner.cache_path_for_args(args).resolve()
        cache_paths.append(cache_path)
        if str(cache_path) != str(Path(arm_meta["cache"]).resolve()):
            raise SuiteError(f"{arm} expected cache path drift")
        call_cap += int(args.run_call_budget)

    if len({canonical_hash(item) for item in harnesses}) != 1:
        raise SuiteError("The four scorer legs do not share one identical harness")
    if len(set(cache_paths)) != len(SCORER_ARMS):
        raise SuiteError("Every scorer leg must have a separate immutable cache")
    if suite.get("nominal_paid_calls") != len(SCORER_ARMS) * NOMINAL_CALLS_PER_ARM:
        raise SuiteError("Nominal paid-call count is not 1,004")
    if call_cap != len(SCORER_ARMS) * CALL_BUDGET_PER_ARM:
        raise SuiteError("Aggregate paid-call attempt cap is not 1,100")
    if suite.get("hard_aggregate_call_attempt_cap") != call_cap:
        raise SuiteError("Manifest aggregate call cap does not match scorer commands")
    freeze_path = Path(suite["input_freeze"])
    recorded_freeze_hash = suite.get("input_freeze_sha256")
    if recorded_freeze_hash is not None:
        if not freeze_path.is_file() or sha256_file(freeze_path) != recorded_freeze_hash:
            raise SuiteError("Frozen input artifact hash drift")
    return {
        "status": "valid",
        "portfolio_experiments": 3,
        "scorer_legs": 4,
        "signal_dates": len(signal_manifest.dates),
        "nominal_paid_calls": suite["nominal_paid_calls"],
        "hard_aggregate_call_attempt_cap": call_cap,
        "development_only": True,
        "blind_period_touched": False,
        "network_calls_made": 0,
    }


def _input_freeze_payload(suite_path: Path) -> dict[str, Any]:
    """Hash every exact anonymous user request using local point-in-time inputs."""
    suite = read_suite(suite_path)
    signal_manifest = runner.load_signal_date_manifest(Path(suite["signal_manifest"]))
    first_command = suite["commands"][SCORER_ARMS[0]]
    args = _runner_args(first_command)
    config = runner.build_config(args)
    _, membership_lookup, market_frames, feature_frames, spy_df = runner.load_experiment_inputs(config)
    pools = runner.collect_v2_llm_source_pools(
        market_frames=market_frames,
        feature_frames=feature_frames,
        membership_lookup=membership_lookup,
        spy_df=spy_df,
        config=config,
        schedule="anchors",
        anchor_offsets=(3,),
        llm_pool_size=60,
        allowed_signal_dates=signal_manifest.date_set,
    )
    runner.validate_pool_date_coverage(pools, signal_manifest.date_set)
    runner.validate_autoresearch_pool_dates(pools, "development")
    rows: list[dict[str, Any]] = []
    for storage_key, pool in sorted(pools.items()):
        if len(pool.long_tickers) != 60 or len(pool.short_tickers) != 60:
            raise SuiteError(f"{storage_key} is not an exact 60L/60S pool")
        anonymous_pool, _ = runner.anonymize_pool_for_llm(pool)
        user_prompt = v1.format_user_prompt(anonymous_pool)
        rows.append(
            {
                "storage_key": storage_key,
                "signal_date": str(pool.signal_date.date()),
                "exit_date": str(pool.exit_date.date()),
                "long_count": len(pool.long_tickers),
                "short_count": len(pool.short_tickers),
                "real_universe_hash": pool.universe_hash,
                "anonymous_universe_hash": anonymous_pool.universe_hash,
                "anonymous_user_prompt_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
            }
        )
    if len(rows) != NOMINAL_CALLS_PER_ARM:
        raise SuiteError(f"Input freeze expected 251 pools, got {len(rows)}")
    return {
        "schema_version": 1,
        "artifact_type": "v2_sleeve_splice_input_freeze",
        "development_only": True,
        "blind_period_touched": False,
        "signal_manifest_sha256": signal_manifest.source_sha256,
        "date_manifest_hash": signal_manifest.date_manifest_hash,
        "code_fingerprint": runner.autoresearch_code_fingerprint(),
        "harness_fingerprint": runner.autoresearch_harness_fingerprint(args),
        "pool_count": len(rows),
        "pool_input_fingerprint": canonical_hash(rows),
        "pools": rows,
        "network_calls_made": 0,
    }


def freeze_suite_inputs(suite_path: Path) -> dict[str, Any]:
    """Materialize the exact local candidate-input hashes without model access."""
    suite_path = Path(suite_path).resolve()
    validate_suite(suite_path)
    suite = read_suite(suite_path)
    payload = _input_freeze_payload(suite_path)
    freeze_path = Path(suite["input_freeze"])
    write_json(freeze_path, payload)
    suite["input_freeze_sha256"] = sha256_file(freeze_path)
    suite["pool_input_fingerprint"] = payload["pool_input_fingerprint"]
    write_json(suite_path, suite)
    validate_suite(suite_path)
    return payload


def verify_suite_inputs(suite_path: Path) -> dict[str, Any]:
    suite_path = Path(suite_path).resolve()
    suite = read_suite(suite_path)
    freeze_path = Path(suite["input_freeze"])
    if not freeze_path.is_file():
        raise SuiteError("Missing input_freeze.json; run prepare with --freeze-inputs")
    frozen = read_json(freeze_path)
    if suite.get("input_freeze_sha256") != sha256_file(freeze_path):
        raise SuiteError("input_freeze.json does not match the suite manifest hash")
    if suite.get("pool_input_fingerprint") != frozen.get("pool_input_fingerprint"):
        raise SuiteError("Input pool fingerprint does not match the suite manifest")
    current = _input_freeze_payload(suite_path)
    if current != frozen:
        changed = [
            key
            for key in (
                "signal_manifest_sha256",
                "date_manifest_hash",
                "code_fingerprint",
                "harness_fingerprint",
                "pool_count",
                "pool_input_fingerprint",
            )
            if current.get(key) != frozen.get(key)
        ]
        raise SuiteError(f"Frozen local scorer inputs changed: {changed}")
    return {
        "status": "verified",
        "pool_count": frozen["pool_count"],
        "pool_input_fingerprint": frozen["pool_input_fingerprint"],
        "network_calls_made": 0,
    }


def _validate_score_rows(rows: Any, *, storage_key: str, side: str) -> None:
    if not isinstance(rows, list) or len(rows) != 60:
        raise SuiteError(f"{storage_key} must contain exactly 60 {side} scores")
    tickers = [str(row.get("ticker", "")).strip().upper() for row in rows if isinstance(row, dict)]
    scores = pd.to_numeric(
        [row.get("score") for row in rows if isinstance(row, dict)], errors="coerce"
    )
    if len(tickers) != 60 or len(set(tickers)) != 60 or any(not ticker for ticker in tickers):
        raise SuiteError(f"{storage_key} has invalid or duplicated {side} tickers")
    if len(scores) != 60 or not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 100.0):
        raise SuiteError(f"{storage_key} has invalid {side} scores")


def inspect_arm_cache(suite_path: Path, arm: str, *, require_complete: bool = False) -> dict[str, Any]:
    suite = read_suite(suite_path)
    if arm not in SCORER_ARMS:
        raise SuiteError(f"Unknown scorer arm {arm!r}")
    cache_path = Path(suite["arms"][arm]["cache"])
    if not cache_path.is_file():
        if require_complete:
            raise SuiteError(f"{arm} cache does not exist")
        return {
            "arm": arm,
            "status": "not_started",
            "cache": str(cache_path),
            "completed_dates": 0,
            "remaining_nominal_dates": NOMINAL_CALLS_PER_ARM,
            "calls_used": 0,
        }
    freeze_path = Path(suite["input_freeze"])
    if not freeze_path.is_file():
        raise SuiteError("Cannot validate scorer cache without input_freeze.json")
    freeze = read_json(freeze_path)
    expected_pools = {row["storage_key"]: row for row in freeze["pools"]}
    payload = read_json(cache_path)
    metadata = payload.get("metadata")
    signals = payload.get("signals")
    if not isinstance(metadata, dict) or not isinstance(signals, dict):
        raise SuiteError(f"{arm} cache is missing metadata/signals")
    arm_meta = suite["arms"][arm]
    expected_metadata = {
        "experimental_candidate_id": arm_meta["candidate_id"],
        "experimental_prompt_sha256": arm_meta["prompt_sha256"],
        "experimental_prompt_spec_sha256": arm_meta["prompt_spec_sha256"],
        "date_manifest_hash": suite["date_manifest_hash"],
        "model": "deepseek-v4-flash",
        "thinking_mode": "disabled",
        "anonymize_tickers": True,
        "side_scope": "both",
        "replicate_id": "r1",
    }
    bad = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if bad:
        raise SuiteError(f"{arm} cache metadata drift: {bad}")
    if not set(signals).issubset(expected_pools):
        raise SuiteError(f"{arm} cache contains dates outside the frozen input set")
    for storage_key, entry in signals.items():
        if not isinstance(entry, dict):
            raise SuiteError(f"{arm} cache entry {storage_key} is not an object")
        expected = expected_pools[storage_key]
        if entry.get("universe_hash") != expected["real_universe_hash"]:
            raise SuiteError(f"{arm} cache universe drift at {storage_key}")
        if entry.get("rebalance_date") != expected["signal_date"]:
            raise SuiteError(f"{arm} cache date drift at {storage_key}")
        if entry.get("prompt_hash") != str(arm_meta["prompt_sha256"])[:12]:
            raise SuiteError(f"{arm} cache prompt drift at {storage_key}")
        _validate_score_rows(entry.get("long_scores"), storage_key=storage_key, side="long")
        _validate_score_rows(entry.get("short_scores"), storage_key=storage_key, side="short")
        long_tickers = {str(row["ticker"]).strip().upper() for row in entry["long_scores"]}
        short_tickers = {str(row["ticker"]).strip().upper() for row in entry["short_scores"]}
        if long_tickers & short_tickers:
            raise SuiteError(f"{arm} cache crosses ticker sides at {storage_key}")
    complete = len(signals) == len(expected_pools)
    if require_complete and not complete:
        raise SuiteError(
            f"{arm} cache is resumably incomplete: {len(signals)}/{len(expected_pools)} dates"
        )
    return {
        "arm": arm,
        "status": "complete" if complete else "partial_resumable",
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "completed_dates": len(signals),
        "remaining_nominal_dates": len(expected_pools) - len(signals),
        "calls_used": int(metadata.get("run_calls_used", 0) or 0),
    }


def suite_status(suite_path: Path) -> dict[str, Any]:
    suite_path = Path(suite_path).resolve()
    validation = validate_suite(suite_path)
    suite = read_suite(suite_path)
    input_freeze_exists = Path(suite["input_freeze"]).is_file()
    arms = [inspect_arm_cache(suite_path, arm) for arm in SCORER_ARMS]
    completed = sum(int(item["completed_dates"]) for item in arms)
    all_complete = all(item["status"] == "complete" for item in arms)
    return {
        **validation,
        "status": "scoring_complete" if all_complete else "ready_for_paid_scoring",
        "input_freeze_exists": input_freeze_exists,
        "completed_nominal_calls": completed,
        "remaining_nominal_calls": validation["nominal_paid_calls"] - completed,
        "arms": arms,
    }


def _write_execution_state(path: Path, state: dict[str, Any], event: str) -> None:
    state["updated_at"] = utc_now()
    state.setdefault("events", []).append({"event": event, "recorded_at": state["updated_at"]})
    write_json(path, state)


def analyze_suite(suite_path: Path) -> dict[str, Any]:
    """Finalize completed caches locally; this function has no HTTP path."""
    suite_path = Path(suite_path).resolve()
    validate_suite(suite_path)
    suite = read_suite(suite_path)
    signal_manifest = runner.load_signal_date_manifest(Path(suite["signal_manifest"]))
    payloads: dict[str, dict[str, Any]] = {}
    for arm in SCORER_ARMS:
        inspect_arm_cache(suite_path, arm, require_complete=True)
        payloads[arm] = read_json(Path(suite["arms"][arm]["cache"]))
    args = _runner_args(suite["commands"][SCORER_ARMS[0]])
    result = experiments.evaluate_completed_suite(
        output_dir=suite_path.parent,
        cache_payloads=payloads,
        signal_manifest=signal_manifest,
        config=runner.build_config(args),
    )
    return result


def run_paid_suite(suite_path: Path, *, confirmation: str) -> dict[str, Any]:
    """Run the four fresh scorer legs only after an exact paid-call acknowledgement."""
    if confirmation != PAID_CONFIRMATION:
        raise SuiteError(
            "Paid execution was not authorized. Supply the exact confirmation string "
            f"{PAID_CONFIRMATION!r}."
        )
    suite_path = Path(suite_path).resolve()
    validate_suite(suite_path)
    verify_suite_inputs(suite_path)
    suite = read_suite(suite_path)
    state_path = suite_path.parent / "execution_state.json"
    state = (
        read_json(state_path)
        if state_path.is_file()
        else {
            "schema_version": 1,
            "artifact_type": "v2_sleeve_splice_execution_state",
            "status": "initialized",
            "started_at": utc_now(),
            "pid": os.getpid(),
            "blind_period_touched": False,
            "arms": {},
            "events": [],
        }
    )
    state.update({"status": "running", "pid": os.getpid(), "paid_confirmation": confirmation})
    _write_execution_state(state_path, state, "paid_run_started_or_resumed")
    initial_code_fingerprint = runner.autoresearch_code_fingerprint()
    for arm in SCORER_ARMS:
        status = inspect_arm_cache(suite_path, arm)
        if status["status"] == "complete":
            state["arms"][arm] = {**status, "resumed": True, "updated_at": utc_now()}
            _write_execution_state(state_path, state, f"{arm}_cache_hit_complete")
            continue
        if runner.autoresearch_code_fingerprint() != initial_code_fingerprint:
            raise SuiteError("Core scorer code changed during paid execution")
        command = [str(value) for value in suite["commands"][arm]]
        log_path = suite_path.parent / "scorers" / arm / "supervisor_runner.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        state["current_arm"] = arm
        state["arms"][arm] = {"status": "running", "started_at": utc_now(), **status}
        _write_execution_state(state_path, state, f"{arm}_started")
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{utc_now()}] starting/resuming {arm}\n")
            log_handle.flush()
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        try:
            finished = inspect_arm_cache(suite_path, arm, require_complete=True)
        except SuiteError as exc:
            partial = inspect_arm_cache(suite_path, arm)
            state.update({"status": "paused_resumable", "current_arm": None})
            state["arms"][arm] = {
                **partial,
                "returncode": completed.returncode,
                "log": str(log_path.resolve()),
                "reason": str(exc),
            }
            _write_execution_state(state_path, state, f"{arm}_paused_resumable")
            raise SuiteError(
                f"{arm} stopped before all 251 dates; the cache is safe to resume. "
                f"Inspect {log_path}."
            ) from exc
        if completed.returncode != 0:
            state.update({"status": "failed", "current_arm": None})
            state["arms"][arm] = {
                **finished,
                "returncode": completed.returncode,
                "log": str(log_path.resolve()),
            }
            _write_execution_state(state_path, state, f"{arm}_failed_nonzero")
            raise SuiteError(f"{arm} returned {completed.returncode}; inspect {log_path}")
        state["arms"][arm] = {**finished, "returncode": 0, "completed_at": utc_now()}
        state["current_arm"] = None
        _write_execution_state(state_path, state, f"{arm}_completed")

    state.update({"status": "analyzing_offline", "current_arm": None})
    _write_execution_state(state_path, state, "offline_analysis_started")
    analysis = analyze_suite(suite_path)
    state.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "current_arm": None,
            "blind_period_touched": False,
            "analysis": analysis,
        }
    )
    _write_execution_state(state_path, state, "suite_completed")
    return state


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three guarded in-sample sleeve-splice tests")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Write frozen artifacts; never calls a model")
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare.add_argument("--signal-manifest", type=Path, default=DEFAULT_SIGNAL_MANIFEST)
    prepare.add_argument(
        "--freeze-inputs",
        action="store_true",
        help="Also hash all 251 exact anonymous scorer inputs from local data.",
    )

    validate = subparsers.add_parser("validate", help="Validate readiness; never calls a model")
    validate.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    validate.add_argument(
        "--verify-inputs",
        action="store_true",
        help="Rebuild and compare all 251 local anonymous input hashes.",
    )
    validate.add_argument("--show-commands", action="store_true")

    status = subparsers.add_parser("status", help="Inspect resumable cache status; no model calls")
    status.add_argument("--suite", type=Path, default=DEFAULT_SUITE)

    paid = subparsers.add_parser("run-paid", help="Explicitly execute the fresh DeepSeek scorer legs")
    paid.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    paid.add_argument("--confirmation", required=True)

    analyze = subparsers.add_parser("analyze", help="Offline replay after scorer caches complete")
    analyze.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        suite = prepare_suite(
            args.output_dir,
            signal_manifest=args.signal_manifest,
            freeze_inputs=bool(args.freeze_inputs),
        )
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "suite": str((Path(args.output_dir) / "suite_manifest.json").resolve()),
                    "portfolio_experiments": suite["portfolio_experiments"],
                    "scorer_legs": suite["scorer_legs"],
                    "nominal_paid_calls": suite["nominal_paid_calls"],
                    "paid_calls_made": 0,
                    "input_freeze_written": bool(args.freeze_inputs),
                },
                indent=2,
            )
        )
    elif args.command == "validate":
        result = validate_suite(args.suite)
        if args.verify_inputs:
            result["input_verification"] = verify_suite_inputs(args.suite)
        if args.show_commands:
            result["commands"] = read_suite(args.suite)["commands"]
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "status":
        print(json.dumps(suite_status(args.suite), indent=2, sort_keys=True))
    elif args.command == "run-paid":
        print(
            json.dumps(
                run_paid_suite(args.suite, confirmation=args.confirmation),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(analyze_suite(args.suite), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
