# Minimal-change LLM contribution: replica and attribution study

**Study date:** 2026-10-27  
**Evaluation period:** 2009-01 through 2013-12  
**Question:** Does LLM scoring add a repeatable return improvement over matched Trail5, and which substitutions account for it?

## Summary

Three scoring replicas used the same minimal-change prompt, model alias, request settings, candidate pools, and portfolio construction. The runs were not ranked or selected after seeing results. Two replicas had small positive lifts on the common model-evaluation period, and the third was negative. The no-selection mean across all three was slightly below Trail5, with a wide interval that includes both gains and losses.

On the full 251-date LLM period, the mean replica lift was positive, but the best 5% of dates explained more than the net lift in every run. Removing those dates made all three replicas negative. The evidence does not establish a reliable improvement.

The LLM's added long names had better average one-week returns than the long names they replaced. Short substitutions were also slightly better on average after reversing return direction. Both side-level portfolio contributions have wide intervals spanning zero, and substitution outcomes vary by replica, year, and regime.

Simple indicator-rank combinations were close to Trail5 by point estimate. Prior-only Ridge and HistGradientBoosting baselines had lower point estimates. Paired intervals for these baselines versus the three-run LLM mean are wide and include zero, so this study does not show that the LLM's extra indicator processing adds value.

## Frozen experiment design

| Setting | Value |
|---|---|
| Prompt | Minimal-change control, hash `5300b8966c29` |
| Model name recorded by the runner | `deepseek-v4-flash` |
| Temperature / thinking | 1.0 / disabled |
| Candidate pool | 60 stocks per side per date |
| Selection | 30 longs and 30 shorts per date |
| Construction | Equal weight within side; 50% capital long and 50% short |
| Schedule | Anchors, offset 3 |
| Ticker treatment | Anonymized for LLM scoring |
| Transaction cost | 0 bps |
| Scoring replicas | 3, all included in the reported mean |

The three prompt strings and the principal request settings match across the run manifests. The prompt used in the run is included at [`inputs/minimal_change_prompt.txt`](inputs/minimal_change_prompt.txt); the individual manifests are under [`inputs/replicas/`](inputs/replicas/).

## Matched comparison: 199 common weekly dates

The Ridge and HistGradientBoosting predictions have a 52-date warmup, so the primary comparison uses the 199 dates where every strategy has predictions. Annualized returns are arithmetic means of weekly capital returns multiplied by 52. Paired lift is strategy return minus the same-date Trail5 return. Intervals use a 5,000-draw, four-week moving-block bootstrap. For the three-replica mean, the bootstrap also resamples the scoring runs.

| Strategy | Annualized return | Paired lift vs Trail5 (95% CI), pp/year | Bootstrap share of draws above zero |
|---|---:|---:|---:|
| Matched Trail5 | 8.52% | 0.00 | — |
| LLM replica 1 | 9.28% | +0.76 (−2.57, +4.10) | 68% |
| LLM replica 2 | 8.85% | +0.33 (−2.17, +2.81) | 60% |
| LLM replica 3 | 6.88% | −1.63 (−4.34, +1.04) | 12% |
| **LLM mean across all 3 replicas** | **8.34%** | **−0.18 (−2.88, +2.46)** | **45%** |
| Trail5 + RSI5, equal within-side ranks | 8.43% | −0.09 (−2.66, +2.48) | 46% |
| Trail5 + RSI5 + Rng55, equal within-side ranks | 8.07% | −0.45 (−3.22, +2.23) | 38% |
| Prior-only Ridge | 5.49% | −3.03 (−7.52, +1.40) | 9% |
| Prior-only HistGradientBoosting | 6.48% | −2.03 (−6.04, +1.97) | 16% |

All paired 95% intervals include zero. The individual replica results vary materially, and the three-run mean is not positive on this common period.

## Full 251-date LLM comparison

The complete period is useful for comparing replicas with Trail5, but the prior-only models do not have predictions for the first 52 dates. Keep this result separate from the 199-date comparison above.

| Strategy | Annualized return | Paired lift vs Trail5 (95% CI), pp/year |
|---|---:|---:|
| Matched Trail5 | 13.02% | 0.00 |
| LLM replica 1 | 16.18% | +3.16 (−0.31, +6.69) |
| LLM replica 2 | 13.39% | +0.37 (−2.41, +3.05) |
| LLM replica 3 | 14.32% | +1.30 (−1.79, +4.46) |
| **LLM mean across all 3 replicas** | **14.63%** | **+1.61 (−1.28, +4.77)** |

Each run's best 5% of dates contributed more than 100% of its net lift: 120% for replica 1, 780% for replica 2, and 213% for replica 3. After removing the best 5% of dates, annualized lift was −0.66, −2.63, and −1.55 pp/year, respectively. This is strong sensitivity to exceptional dates, not a typical weekly improvement.

## Added-versus-dropped attribution

For each date and side, an added stock is selected by the LLM but not Trail5; a dropped stock is selected by Trail5 but not the LLM. Directional return is raw forward C2C return for longs and its negative for shorts, so positive values represent better performance for that side. The annualized capital contribution accounts for the 50% allocation to each side, equal weighting, and the fraction of positions changed.

| Side | Added mean directional return | Dropped mean directional return | Annualized portfolio contribution (95% CI), pp/year | Positive substitution dates |
|---|---:|---:|---:|---:|
| Long | +0.776% | +0.514% | +1.36 (−0.74, +3.78) | 51.8% |
| Short, sign-adjusted | −0.254% | −0.374% | +0.25 (−1.74, +2.44) | 48.2% |

The long-side average favors LLM additions, but the confidence interval spans zero and only about half of the date-level substitutions helped. For shorts, additions were less adverse than the dropped names on average, but the short-side contribution is small and uncertain.

The year splits are not stable across replicas. Long attribution was positive in 2009 and 2010 in all three runs, but mixed from 2011 onward. Short attribution was negative in every run in 2012 and 2013, with mixed signs in earlier years. By regime, total portfolio attribution was positive for all runs in `grind_up`, mixed in `normal` and `crisis`, and negative for all runs in `squeeze`; the squeeze regime has only four dates. These are descriptive splits, not reliable regime-specific estimates.

## Comparison with simpler methods

The simple combinations rank each candidate by the equal average of within-date, within-side indicator percentiles. The statistical baselines reuse the earlier expanding walk-forward Ridge and HistGradientBoosting predictions. Their 13 candidate indicators were MRL, MRS, Trail5, MRSig, RSI5, Rng55, Vel/V, %MA100, 52WH, Mom3m, AutoC, ADX, and VolPctl; the models also had one-hot market-regime context. Candidate keys, all features, forward returns, and regime labels were checked against the exact LLM candidate panel on all 199 evaluation dates. The leakage audit reports all training rows as strictly prior; the model used a one-period embargo, 52-date warmup, and refit every four dates. Reused model inputs are preserved, compressed, in [`inputs/tabular_baselines/`](inputs/tabular_baselines/).

Direct paired differences against the three-replica LLM mean, in pp/year, were:

| Baseline minus LLM mean | Difference (95% CI) |
|---|---:|
| Trail5 + RSI5 | +0.09 (−3.13, +3.08) |
| Trail5 + RSI5 + Rng55 | −0.27 (−3.15, +2.61) |
| Ridge | −2.85 (−6.90, +1.22) |
| HistGradientBoosting | −1.85 (−5.65, +1.84) |

All intervals include zero. The simple combinations are near Trail5; the statistical methods have lower point estimates, but uncertainty is too broad to conclude that the LLM beats them or that the indicator information itself explains a lift.

## What “backend revision not recorded” means

The run manifests and cached score records name `deepseek-v4-flash` and record temperature, thinking mode, prompt hash, and portfolio settings. The fields `provider`, `model_revision`, and `backend_revision` are blank in all three manifests; the cached score records also contain no revision or request ID. Therefore we know which model alias the client requested, but not the exact provider-side checkpoint/build that answered each request.

This **does not mean the provider changed the model**. It means the logs cannot rule out that possibility. A hosted alias can sometimes keep the same name while its underlying model serving version changes. Since these replicas were run at different times, their differences could reflect sampling variation, a provider-side version change, or both. We cannot separate those causes from the saved metadata.

For stronger reproducibility, log a pinned snapshot/build identifier and provider response metadata or request IDs whenever the API exposes them. This study does not claim that any backend change occurred.

## Limitations

- Only three scoring replicas were run; between-run uncertainty is therefore estimated from a small number of runs.
- The test period is historical development data (2009–2013), not a new untouched forward period.
- Transaction costs are zero; results are gross of trading costs.
- The provider-side model revision was not recorded.
- The date and regime subgroup results have less data than the aggregate comparison, especially `squeeze` (four dates).

## Artifact guide

- [`tables/comparison_table.csv`](tables/comparison_table.csv): common 199-date strategy comparison.
- [`tables/replica_full_window.csv`](tables/replica_full_window.csv): all 251 LLM dates and outlier sensitivity.
- [`tables/added_dropped_attribution.csv`](tables/added_dropped_attribution.csv): per-date, per-side added/dropped names and return attribution.
- [`tables/attribution_by_side.csv`](tables/attribution_by_side.csv), [`tables/attribution_by_year_and_side.csv`](tables/attribution_by_year_and_side.csv), and [`tables/attribution_by_regime_and_side.csv`](tables/attribution_by_regime_and_side.csv): attribution summaries.
- [`tables/paired_lifts_by_date.csv`](tables/paired_lifts_by_date.csv): date-matched return differences.
- [`tables/summary.json`](tables/summary.json): configuration and validation summary.
- [`inputs/`](inputs/): prompt, run manifests, portfolio outputs, compressed candidate checks, and prior-only model predictions.
- [`code/analyze_minimal_change_replicas.py`](code/analyze_minimal_change_replicas.py): snapshot of the analysis code; the same working copy is also present at `scripts/analyze_minimal_change_replicas.py` in the repository.

The package contains derived research artifacts and the candidate/model panels used for this comparison. It does not include raw historical market data or API credentials.
