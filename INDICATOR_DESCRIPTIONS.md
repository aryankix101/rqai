# Technical Indicators Reference Guide

## One-Liner Descriptions (Based on Source Code)

| Indicator | Explanation |
|-----------|-------------|
| **adx_14d** | 14-day Average Directional Index measuring trend strength (0-100 scale) |
| **adx_14d_zscore** | ADX normalized to standard deviations from 50-day rolling mean |
| **atr_14d** | 14-day Average True Range measuring price volatility in absolute terms |
| **atr_14d_pct** | ATR as percentage of current price |
| **autocorr_10d** | 10-day autocorrelation of returns showing if price changes repeat patterns |
| **autocorr_21d** | 21-day autocorrelation of returns showing if price changes repeat patterns |
| **autocorr_5d** | 5-day autocorrelation of returns showing if price changes repeat patterns |
| **avg_dollar_vol_21d** | Average daily trading volume in dollars over 21 days (dollar turnover) |
| **avg_vol_3m** | Average daily volume over 3 months (63 trading days) |
| **close_loc_10d** | Close price position within 10-day high-low range (0=low, 1=high) |
| **days_since_52w_high** | Number of trading days since price touched 52-week high |
| **days_since_52w_low** | Number of trading days since price touched 52-week low |
| **directional_efficiency_10d** | 10-day net move divided by total movement (trend straightness, 0-1 scale) |
| **directional_efficiency_20d** | 20-day net move divided by total movement (trend straightness, 0-1 scale) |
| **gap_size_pct** | Opening gap as percentage of previous close (positive=gap up) |
| **gap_size_zscore** | Gap size normalized to standard deviations from historical average |
| **intraday_ret** | Daily return from open to close (intraday directional strength) |
| **intraday_ret_zscore** | Intraday return normalized to standard deviations from 50-day history |
| **is_contraction** | Binary flag (1/0) if money supply growth (M2 ROC) is in bottom third |
| **is_expansion** | Binary flag (1/0) if money supply growth (M2 ROC) is in top third |
| **is_neutral** | Binary flag (1/0) if money supply growth (M2 ROC) is in middle third |
| **is_opex_day** | Binary flag (1/0) if day is options expiration Friday |
| **is_opex_normal** | Binary flag (1/0) if day is NOT near options expiration |
| **is_post_opex** | Binary flag (1/0) if day is 1-3 days after options expiration |
| **is_vix_high** | Binary flag (1/0) if VIX is in top third of 252-day range |
| **is_vix_low** | Binary flag (1/0) if VIX is in bottom third of 252-day range |
| **is_vix_medium** | Binary flag (1/0) if VIX is in middle third of 252-day range |
| **liquidity_score_0_100** | Composite liquidity score (0-100) based on dollar volume, bid-ask spread, volume spikes |
| **m1_roc_4w** | M1 money supply rate of change over 4 weeks (transaction velocity) |
| **m2_m1_ratio** | M2 divided by M1 (monetary supply structure, shows excess liquidity) |
| **m2_m1_ratio_percentile_52w** | M2/M1 ratio ranked within 52-week history (0-100 percentile) |
| **m2_roc_4w** | M2 money supply rate of change over 4 weeks (monetary expansion/contraction) |
| **ma10_distance_vol_normalized** | Distance from 10-day MA scaled by volatility and price |
| **ma10_distance_zscore** | Distance from 10-day MA normalized to standard deviations |
| **ma5_distance_vol_normalized** | Distance from 5-day MA scaled by volatility and price |
| **ma5_distance_zscore** | Distance from 5-day MA normalized to standard deviations |
| **mom_1m** | 1-month price momentum (return over 21 days) |
| **mom_1m_zscore** | 1-month momentum normalized to standard deviations from history |
| **mom_3m** | 3-month price momentum (return over 63 days) |
| **mom_3m_zscore** | 3-month momentum normalized to standard deviations from history |
| **pct_from_52w_high** | Current price as percentage below 52-week high (0%=at high, 100%=far from high) |
| **pct_from_ma100** | Current price as percentage from 100-day moving average |
| **pct_from_ma20** | Current price as percentage from 20-day moving average |
| **pct_from_ma75** | Current price as percentage from 75-day moving average |
| **pct_up_days_10d** | Percentage of up days in 10-day period (0-100%) |
| **pct_up_days_20d** | Percentage of up days in 20-day period (0-100%) |
| **pct_up_days_5d** | Percentage of up days in 5-day period (0-100%) |
| **range_pct** | Daily price range (high-low) as percentage of close |
| **range_pct_zscore** | Daily range normalized to standard deviations from 50-day history |
| **range_pos_20d** | Close position within 20-day high-low range (0=20d-low, 1=20d-high) |
| **range_pos_55d** | Close position within 55-day high-low range (0=55d-low, 1=55d-high) |
| **range_vol_impulse** | Daily price range multiplied by volume z-score (range with urgency) |
| **realized_vol_10d** | 10-day historical volatility annualized to % scale |
| **realized_vol_14d** | 14-day historical volatility annualized to % scale |
| **realized_vol_21d** | 21-day historical volatility annualized to % scale |
| **ret_10d** | 10-day return |
| **ret_10d_zscore** | 10-day return normalized to standard deviations |
| **ret_1d** | 1-day return |
| **ret_1d_zscore** | 1-day return normalized to standard deviations |
| **ret_2d** | 2-day return |
| **ret_2d_zscore** | 2-day return normalized to standard deviations |
| **ret_3d** | 3-day return |
| **ret_3d_zscore** | 3-day return normalized to standard deviations |
| **ret_5d** | 5-day return |
| **ret_5d_zscore** | 5-day return normalized to standard deviations |
| **roll_spread_21** | Rolling-based bid-ask spread estimate from 21-day covariance (0-1 scale) |
| **rsi_2d** | 2-day RSI measuring short-term momentum exhaustion (0-100 scale) |
| **rsi_3d** | 3-day RSI measuring short-term momentum exhaustion (0-100 scale) |
| **rsi_4d** | 4-day RSI measuring short-term momentum exhaustion (0-100 scale) |
| **rsi_5d** | 5-day RSI measuring short-term momentum exhaustion (0-100 scale) |
| **stretch_pct_from_ma20** | Price percentage distance above 20-day MA (positive=above, negative=below) |
| **stretch_z_20** | Volatility-adjusted stretch above 20-day MA in standard deviations |
| **vel_10d** | 10-day price velocity (10-day return as decimal) |
| **vel_10d_over_vol** | 10-day velocity divided by 21-day volatility (momentum relative to recent vol) |
| **vel_5d** | 5-day price velocity (5-day return as decimal) |
| **vel_5d_over_vol** | 5-day velocity divided by 21-day volatility (momentum relative to recent vol) |
| **vix_level** | Current VIX index value (market implied volatility) |
| **vix_percentile_252d** | VIX ranked within 252-day history (0-100 percentile) |
| **vix_roc_1d** | 1-day VIX rate of change (volatility momentum) |
| **vix_roc_20d** | 20-day VIX rate of change (volatility trend) |
| **vix_roc_5d** | 5-day VIX rate of change (short-term volatility trend) |
| **vol_adjusted_trend_ma100** | 100-day trend slope adjusted for current volatility level |
| **vol_adjusted_trend_ma100_slope** | Slope of 100-day MA normalized by volatility (trend strength) |
| **vol_adjusted_trend_ma20** | 20-day trend slope adjusted for current volatility level |
| **vol_adjusted_trend_ma20_slope** | Slope of 20-day MA normalized by volatility (trend strength) |
| **vol_imbalance_10d** | 10-day directional volume imbalance: (vol_up - vol_down) / (vol_up + vol_down) [-1=bearish, +1=bullish] |
| **vol_imbalance_20d** | 20-day directional volume imbalance: (vol_up - vol_down) / (vol_up + vol_down) [-1=bearish, +1=bullish] |
| **vol_last** | Most recent day's trading volume (absolute shares/contracts) |
| **vol_percentile** | Current realized volatility ranked within recent rolling history |
| **vol_percentile_252d** | Volatility ranked within 252-day history (0-100 percentile) |
| **vol_percentile_504d** | Volatility ranked within 504-day history (0-100 percentile, 2 years) |
| **vol_roc_pct_20v10** | Percentage change from 10-day to 20-day volatility |
| **vol_roc_ratio_20v10** | Ratio of 20-day volatility to 10-day volatility (vol regime change) |
| **volume_spike_z** | Today's volume in standard deviations from 21-day average (volume extremeness) |

## Key Definitions

### Volume Imbalance
- **UP days**: days where close > previous close
- **DOWN days**: days where close < previous close
- **Calculation**: (volume_on_up_days - volume_on_down_days) / (total_volume)
- **Interpretation**: +1 = all volume on bullish days, -1 = all volume on bearish days, 0 = neutral

### Z-Score Indicators
All "_zscore" indicators are normalized to standard deviations from a rolling historical window (typically 50-100 days), showing how extreme the current reading is relative to recent history.

### Volatility Percentile
Ranks current realized volatility against historical rolling window (252d = 1 year, 504d = 2 years), where 0 = lowest volatility ever, 100 = highest volatility ever in that window.

### Money Supply Indicators
- **M1**: Narrow money (cash, checking accounts) - transaction velocity
- **M2**: Broad money (M1 + savings, money market) - total liquidity
- **Regime**: Expansion (top 33%), Neutral (middle), Contraction (bottom 33%) based on M2 ROC percentile

### VIX Regime
- **Low**: VIX in bottom 33% of 252-day range
- **Medium**: VIX in middle 33%
- **High**: VIX in top 33%

### OPEX (Options Expiration)
- **Normal**: Not near options expiration
- **Pre-OPEX**: 1-3 days before third Friday
- **OPEX day**: Third Friday of month
- **Post-OPEX**: 1-3 days after third Friday
