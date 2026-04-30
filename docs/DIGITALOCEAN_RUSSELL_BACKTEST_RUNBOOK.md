# DigitalOcean Russell Backtest Runbook

This runbook explains how the Russell backtest flow will work on a DigitalOcean Droplet with the current codebase.

## What The Pipeline Expects

The repo is now set up for a full Russell OHLCV workflow.

Expected Russell source file:

- default path: `RusselData/compressed/Russell_1000_OHLCV.parquet`

Configured in:

- `data_source_config.py`

The Russell dataset should be long-form and include at least:

- `date`
- `ticker`
- `open`
- `high`
- `low`
- `close`
- `volume`

Optional but supported:

- `adj_open`
- `adj_high`
- `adj_low`
- `adj_close`
- `asset_id`
- `source_column`

If the file is CSV or Excel instead of Parquet, update `RUSSELL_OHLCV_PATH` in `data_source_config.py` to point to that file.

## How The Run Will Work

Once the OHLCV file is in place, the Droplet run is:

1. load Russell OHLCV histories
2. build per-ticker Russell signal dumps with the same indicator contract used by the legacy path
3. build per-ticker composite score files
4. run factor backtests for one or more horizons
5. sync results back to your Mac

In code, that means:

- `russell_signal_dump.py` reads the OHLCV dataset and emits `backtest_results/russell_signal_dump/*.csv`
- `composite_score_calculator.py` turns those into `backtest_results/composite_scores_russell/*.csv`
- `factor_backtest_engine.py` reads those composites and writes `backtest_results/factor_backtest/...`

## Recommended Droplet

For research runs:

- OS: Ubuntu 24.04 LTS
- CPU: Premium AMD or CPU-Optimized
- Starter size: `8 vCPU / 16 GB RAM`
- Better size for heavy parallel runs: `16 vCPU / 32 GB RAM`
- Disk: `100 GB` minimum

Use a Block Storage volume if you expect to keep multiple large result sets or snapshots.

## One-Time Droplet Setup

Upload the repo from your Mac:

```bash
rsync -azP \
  --exclude '.git' \
  --exclude '.venv' \
  /Users/aryankumawat/Desktop/RQA/rqai/ \
  root@YOUR_DROPLET_IP:~/rqai/
```

SSH in:

```bash
ssh root@YOUR_DROPLET_IP
```

Install system packages:

```bash
apt-get update
apt-get install -y python3 python3-venv build-essential gfortran libopenblas-dev liblapack-dev tmux htop
```

Create the environment:

```bash
cd ~/rqai
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install numpy pandas scipy matplotlib seaborn pyarrow openpyxl requests ta-lib
mkdir -p backtest_results/.mplconfig
```

Use `tmux` so the run survives disconnects:

```bash
tmux new -s russell
```

## Step 1: Put The Russell OHLCV File On The Droplet

Copy the Russell OHLCV file to the configured location:

```bash
rsync -azP \
  /path/to/your/Russell_1000_OHLCV.parquet \
  root@YOUR_DROPLET_IP:~/rqai/RusselData/compressed/Russell_1000_OHLCV.parquet
```

If your file is named differently, either rename it on upload or update `RUSSELL_OHLCV_PATH` in `data_source_config.py`.

Quick verification on the Droplet:

```bash
cd ~/rqai
ls -lh RusselData/compressed/Russell_1000_OHLCV.parquet
```

## Step 2: Build Russell Signal Dumps

This stage computes the indicator set required by `composite_score_calculator.py`.

Run the full history:

```bash
cd ~/rqai
MPLCONFIGDIR=backtest_results/.mplconfig \
./.venv/bin/python russell_signal_dump.py --start 1997-01-01 --end 2025-12-31
```

Run a smaller validation subset first:

```bash
cd ~/rqai
MPLCONFIGDIR=backtest_results/.mplconfig \
./.venv/bin/python russell_signal_dump.py --start 2018-01-01 --end 2020-12-31 --tickers AAPL,MSFT,NVDA
```

Output:

- `backtest_results/russell_signal_dump/russell_signal_dump_<TICKER>_<START>_<END>.csv`

## Step 3: Build Composite Scores

This stage converts each signal dump into a composite score file.

Run in parallel:

```bash
cd ~/rqai
find backtest_results/russell_signal_dump -name 'russell_signal_dump_*.csv' -print0 | \
  xargs -0 -n 1 -P 8 ./.venv/bin/python composite_score_calculator.py
```

If you have a bigger Droplet, raise `-P 8` to `-P 12` or `-P 16`.

Output:

- `backtest_results/composite_scores_russell/composite_scores_<TICKER>_<START>_<END>.csv`

## Step 4: Run The Backtest

The backtest engine already defaults to `backtest_results/composite_scores_russell`.

One horizon:

```bash
cd ~/rqai
MPLCONFIGDIR=backtest_results/.mplconfig \
./.venv/bin/python factor_backtest_engine.py 10 2000-01-01 2025-12-31 --skip-plots
```

Multiple horizons:

```bash
cd ~/rqai
for h in 5 10 20 30 60; do
  MPLCONFIGDIR=backtest_results/.mplconfig \
  ./.venv/bin/python factor_backtest_engine.py "$h" 2000-01-01 2025-12-31 --skip-plots
done
```

Smaller validation slice:

```bash
cd ~/rqai
MPLCONFIGDIR=backtest_results/.mplconfig \
./.venv/bin/python factor_backtest_engine.py 10 2018-01-01 2020-12-31 --skip-plots
```

Output:

- `backtest_results/factor_backtest/<variant>/<horizon>/...`
- or baseline paths like `backtest_results/factor_backtest/10d_horizon/...`

## Fastest Practical Run Pattern

For normal research, do this:

1. use a small ticker/date slice first
2. confirm `russell_signal_dump.py` runs cleanly
3. confirm `composite_score_calculator.py` writes files to `composite_scores_russell`
4. run `factor_backtest_engine.py` with `--skip-plots`
5. only then launch the full-history full-universe run

That avoids burning hours on a schema mismatch or bad file format.

## Pull Results Back To Your Mac

Backtest outputs:

```bash
rsync -azP \
  root@YOUR_DROPLET_IP:~/rqai/backtest_results/factor_backtest/ \
  /Users/aryankumawat/Desktop/RQA/rqai/backtest_results/factor_backtest/
```

Composite scores too, if you want them cached locally:

```bash
rsync -azP \
  root@YOUR_DROPLET_IP:~/rqai/backtest_results/composite_scores_russell/ \
  /Users/aryankumawat/Desktop/RQA/rqai/backtest_results/composite_scores_russell/
```

## Speed Tips

The biggest practical wins are:

- always use `--skip-plots` during research loops
- keep `MPLCONFIGDIR=backtest_results/.mplconfig`
- parallelize `composite_score_calculator.py` across many files
- parallelize horizons across cores
- keep the generated signal dumps and composites so you do not rebuild them unnecessarily

Example: run horizons in parallel on a larger Droplet

```bash
cd ~/rqai
printf '%s\n' 5 10 20 30 60 | \
  xargs -n 1 -P 5 -I {} sh -c 'MPLCONFIGDIR=backtest_results/.mplconfig ./.venv/bin/python factor_backtest_engine.py {} 2000-01-01 2025-12-31 --skip-plots'
```

## Failure Points To Check First

If something fails, check these in order:

1. Russell OHLCV file exists at `RUSSELL_OHLCV_PATH`
2. the file columns match the expected schema
3. `pyarrow` is installed if the file is Parquet
4. `ta-lib` is installed in the venv
5. `backtest_results/russell_signal_dump` is populated before running `composite_score_calculator.py`
6. `backtest_results/composite_scores_russell` is populated before running `factor_backtest_engine.py`

## End-To-End Command Block

If your OHLCV file is already in the expected location, this is the full run:

```bash
cd ~/rqai
source .venv/bin/activate
export MPLCONFIGDIR=backtest_results/.mplconfig

./.venv/bin/python russell_signal_dump.py --start 1997-01-01 --end 2025-12-31

find backtest_results/russell_signal_dump -name 'russell_signal_dump_*.csv' -print0 | \
  xargs -0 -n 1 -P 8 ./.venv/bin/python composite_score_calculator.py

for h in 5 10 20 30 60; do
  ./.venv/bin/python factor_backtest_engine.py "$h" 2000-01-01 2025-12-31 --skip-plots
done
```

## Best Next Step

Once you have the OHLCV file, the best sequence is:

1. upload it to `RusselData/compressed/Russell_1000_OHLCV.parquet`
2. run a tiny validation subset on the Droplet
3. run the full signal-dump build
4. run composite generation in parallel
5. run the full backtest set
