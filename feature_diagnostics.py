"""
Feature Saturation & Confirmation Quality Diagnostics

Analyzes all composite signals and their components to detect:
1. Saturation: % of days features hit ±1.0 bounds (clipping)
2. Distribution: Mean, std, quantiles of each feature
3. Correlation: How redundant are your features?
4. Weight Impact: Does a feature's weight actually matter given its distribution?
5. Tier1-Alpha Confirmation: How often do they agree/disagree and P&L impact

Usage:
    python feature_diagnostics.py SYMBOL START_DATE END_DATE
    python feature_diagnostics.py AAPL 2020-01-01 2024-12-31
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict
import requests
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

# Import your modules
from technical_indicators_calculator import TechnicalIndicatorCalculator
from composite_scores import (
    calculate_momentum_composite,
    calculate_trend_composite,
    calculate_relative_strength_composite,
    calculate_volume_confirmation_composite,
    calculate_risk_composite,
    calculate_next_day_alpha_composite
)
from deterministic_scoring import calculate_tier1_master_score
from regime_classifier import classify_market_regime
from hit_rate_validator import calculate_forward_return_label

# Load Tiingo API credentials
with open('secrets.json', 'r') as f:
    secrets = json.load(f)
    TIINGO_API_KEY = secrets['TIINGO_API_KEY']


class FeatureDiagnostics:
    """Analyze feature saturation and quality across time."""
    
    def __init__(self, symbol: str, start_date: str, end_date: str, execution_model: str = "next_open_to_close"):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.execution_model = execution_model
        self.features_history = []
    
    def fetch_tiingo_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV data from Tiingo API"""
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {TIINGO_API_KEY}'
        }
        params = {
            'startDate': start_date,
            'endDate': end_date
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch {ticker}: {response.status_code} - {response.text}")
            return pd.DataFrame()
        
        data = response.json()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Rename columns to match TechnicalIndicatorCalculator expected format (Capitalized)
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adjOpen': 'Adj_Open',
            'adjHigh': 'Adj_High',
            'adjLow': 'Adj_Low',
            'adjClose': 'Adj_Close',
            'adjVolume': 'Adj_Volume'
        })
        
        return df
        
    def collect_features_over_time(self):
        """Run through historical data and collect all feature values."""
        print(f"\n{'='*80}")
        print(f"Collecting feature data for {self.symbol}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"{'='*80}\n")
        
        # Fetch data from Tiingo
        print(f"Fetching data from Tiingo API...")
        hist = self.fetch_tiingo_data(self.symbol, self.start_date, self.end_date)
        
        if hist.empty:
            print(f"No data for {self.symbol}")
            return
        
        # Calculate indicators on full history
        calculator = TechnicalIndicatorCalculator(hist)
        
        # Iterate through dates (skip warmup period)
        dates = hist.index[252:]  # Skip first year for indicator warmup
        
        print(f"Processing {len(dates)} dates...")
        for idx, current_date in enumerate(dates):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(dates)} ({idx/len(dates)*100:.1f}%)")
            
            # Get data up to current date
            hist_slice = hist[hist.index <= current_date].copy()
            
            if len(hist_slice) < 252:
                continue
            
            # Recalculate indicators
            calc = TechnicalIndicatorCalculator(hist_slice)
            indicators = calc.calculate_all_indicators(
                symbol=self.symbol,
                sector="Unknown",
                industry="Unknown"
            )
            
            
            if 'events' in indicators and 'price_events' not in indicators:
                indicators['price_events'] = indicators['events']
            
            # Calculate composites
            mom_comp = calculate_momentum_composite(indicators)
            trend_comp = calculate_trend_composite(indicators)
            rs_comp = calculate_relative_strength_composite(indicators)
            vol_comp = calculate_volume_confirmation_composite(indicators)
            risk_comp = calculate_risk_composite(indicators)
            
            # Package composites and calculate regime
            composites = {
                'momentum_composite': mom_comp,
                'trend_composite': trend_comp,
                'volume_composite': vol_comp,
                'risk_composite': risk_comp,
                'rs_composite': rs_comp
            }
            
            from regime_classifier import classify_market_regime, classify_volatility_regime
            regime = classify_market_regime(indicators)
            vol_regime = classify_volatility_regime(indicators)
            
            # Calculate tier1 score
            tier1_result = calculate_tier1_master_score(composites, regime)
            tier1_score_val = tier1_result['tier1_score']
            tier1_quality_val = tier1_result.get('tier1_quality', 0.5)
            
            alpha_comp = calculate_next_day_alpha_composite(
                indicators,
                tier1_context=tier1_result
            )
            
            # Get next day's return for validation (using centralized function)
            next_day_ret = calculate_forward_return_label(hist, current_date, horizon=1, execution_model=self.execution_model)
            
            # Calculate overnight gap separately for diagnostics
            next_date_idx = hist.index.get_loc(current_date) + 1
            if next_date_idx < len(hist):
                next_open = hist.iloc[next_date_idx]['Open']
                current_close = hist.iloc[hist.index.get_loc(current_date)]['Close']
                overnight_gap = (next_open / current_close) - 1.0 if current_close > 0 else 0.0
            else:
                overnight_gap = None
            
            # Extract ALL feature values
            feature_dict = {
                'date': current_date,
                
                # Forward returns for validation
                'next_day_ret': next_day_ret,
                'overnight_gap': overnight_gap,
                
                # Momentum composite components
                'mom_score': mom_comp['score'],
                'mom_1m_norm': mom_comp['components'].get('momentum_returns', 0.0),
                'mom_macd_signal': mom_comp['components'].get('macd_signal', 0.0),
                'mom_rsi_mod': mom_comp['components'].get('rsi_moderation', 0.0),
                
                # Trend composite components
                'trend_score': trend_comp['score'],
                'trend_ma_structure': trend_comp['components'].get('ma_structure', 0.0),
                'trend_price_structure': trend_comp['components'].get('price_structure', 0.0),
                'trend_adx_level': trend_comp['components'].get('adx_level_score', 0.0),
                'trend_cross_boost': trend_comp['components'].get('cross_boost', 0.0),
                
                # RS composite components
                'rs_score': rs_comp['score'],
                'rs_relative_perf': rs_comp['components'].get('relative_performance', 0.0),
                'rs_trend_bonus': rs_comp['components'].get('trend_bonus', 0.0),
                'rs_capture': rs_comp['components'].get('capture_ratio_score', 0.0),
                
                # Volume composite components
                'vol_score': vol_comp['score'],
                'vol_obv': vol_comp['components'].get('obv', 0.0),
                'vol_ad': vol_comp['components'].get('accumulation_distribution', 0.0),
                'vol_spike': vol_comp['components'].get('volume_spike', 0.0),
                'vol_confirmation': vol_comp['components'].get('confirmation_bonus', 0.0),
                
                # Risk composite components
                'risk_score': risk_comp['score'],
                'risk_sharpe': risk_comp['components'].get('sharpe', 0.0),
                'risk_sortino': risk_comp['components'].get('sortino', 0.0),
                'risk_drawdown': risk_comp['components'].get('max_drawdown', 0.0),
                'risk_volatility': risk_comp['components'].get('volatility', 0.0),
                'risk_beta': risk_comp['components'].get('beta', 0.0),
                
                # Next-day alpha components
                'alpha_score': alpha_comp['score'],
                'alpha_reversion_signal': alpha_comp['components'].get('reversion_signal', 0.0),
                'alpha_continuation_signal': alpha_comp['components'].get('continuation_signal', 0.0),
                'alpha_reversion_weight': alpha_comp['components'].get('reversion_weight', 0.0),
                'alpha_continuation_weight': alpha_comp['components'].get('continuation_weight', 0.0),
                'alpha_rsi_reversion': alpha_comp['components'].get('rsi_reversion', 0.0),
                'alpha_close_loc_reversion': alpha_comp['components'].get('close_loc_reversion', 0.0),
                'alpha_short_ret_z': alpha_comp['components'].get('short_ret_z_signal', 0.0),
                'alpha_gap_signal_z': alpha_comp['components'].get('gap_signal_z', 0.0),
                'alpha_ret_3d_signal_z': alpha_comp['components'].get('ret_3d_signal_z', 0.0),
                'alpha_regime': alpha_comp['regime'],
                'alpha_decision': alpha_comp['decision'],
                
                # Tier-1 score
                'tier1_score': tier1_score_val,
                'tier1_quality': tier1_quality_val,
                'tier1_regime': regime.get('regime', 'unknown'),
                'volatility_regime': vol_regime.get('regime', 'unknown'),
                
                # Alpha decision for direction conditioning
                'alpha_decision': alpha_comp.get('decision', 'HOLD'),
                
                # Forward returns for validation
                'forward_return_1d': next_day_ret,
                'overnight_gap': overnight_gap,
                
                # Tier1-Alpha Confirmation
                'tier1_confirmation_confidence': None,
                'tier1_confirmation_size_mult': None,
                'tier1_confirmation_agreement': None,
                'tier1_confirmation_alpha_strong': None,
                'tier1_confirmation_tier1_meaningful': None,
            }
            
            # Extract tier1 confirmation if available
            if alpha_comp.get('tier1_confirmation'):
                conf = alpha_comp['tier1_confirmation']
                feature_dict['tier1_confirmation_confidence'] = conf['confidence']
                feature_dict['tier1_confirmation_size_mult'] = conf['size_multiplier']
                feature_dict['tier1_confirmation_agreement'] = conf['agreement']
                feature_dict['tier1_confirmation_alpha_strong'] = conf['alpha_strong']
                feature_dict['tier1_confirmation_tier1_meaningful'] = conf['tier1_strong']
            
            self.features_history.append(feature_dict)
        
        print(f"\n✓ Collected {len(self.features_history)} data points\n")
    
    def analyze_saturation(self) -> pd.DataFrame:
        """Analyze what % of days each feature hits ±1.0 bounds."""
        df = pd.DataFrame(self.features_history)
        
        # Numeric features only, exclude forward returns and target leakage
        exclude_cols = {
            'forward_return_1d', 'forward_return_3d', 'forward_return_5d',
            'next_day_ret', 'overnight_gap', 'date'
        }
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        
        results = []
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) == 0:
                continue
            
            # Saturation metrics
            pct_at_pos_1 = (values >= 0.999).sum() / len(values) * 100
            pct_at_neg_1 = (values <= -0.999).sum() / len(values) * 100
            pct_saturated = pct_at_pos_1 + pct_at_neg_1
            
            # Distribution metrics
            mean_val = values.mean()
            std_val = values.std()
            q25, q50, q75 = values.quantile([0.25, 0.50, 0.75])
            min_val = values.min()
            max_val = values.max()
            
            # Effective range (10th to 90th percentile)
            q10, q90 = values.quantile([0.10, 0.90])
            effective_range = q90 - q10
            
            results.append({
                'feature': col,
                'pct_at_+1': pct_at_pos_1,
                'pct_at_-1': pct_at_neg_1,
                'pct_saturated': pct_saturated,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'q25': q25,
                'q50': q50,
                'q75': q75,
                'max': max_val,
                'effective_range_10_90': effective_range,
                'n_samples': len(values)
            })
        
        return pd.DataFrame(results).sort_values('pct_saturated', ascending=False)
    
    def analyze_weight_impact(self) -> Dict:
        """
        Analyze if feature weights actually matter given distributions.
        
        If a feature has std=0.1 and weight=0.3, its actual contribution
        is much smaller than a feature with std=0.8 and weight=0.2.
        """
        df = pd.DataFrame(self.features_history)
        
        # Define weights from composite_scores.py
        weights = {
            # Momentum composite
            'mom_1m_norm': 0.30,
            'mom_macd_signal': 0.20,
            'mom_rsi_mod': 0.10,
            
            # Alpha mean reversion
            'alpha_rsi_reversion': 0.30,
            'alpha_close_loc_reversion': 0.30,
            'alpha_short_ret_z': 0.25,
            'alpha_gap_signal_z': 0.15,
            
            # Trend composite
            'trend_ma_structure': 0.36,  # 0.6 * 0.6 from base_trend
            'trend_price_structure': 0.24,  # 0.4 * 0.6 from base_trend
            
            # RS composite
            'rs_relative_perf': 1.0,  # Primary signal
            
            # Volume composite
            'vol_obv': 0.40,
            'vol_ad': 0.40,
            'vol_spike': 0.20,
        }
        
        impact_analysis = {}
        for feature, weight in weights.items():
            if feature not in df.columns:
                continue
            
            values = df[feature].dropna()
            if len(values) == 0:
                continue
            
            std = values.std()
            mean_abs = values.abs().mean()
            
            # Effective contribution = weight × typical_magnitude
            effective_contribution = weight * std
            
            impact_analysis[feature] = {
                'weight': weight,
                'std': std,
                'mean_abs': mean_abs,
                'effective_contribution': effective_contribution,
                'contribution_rank': 0  # Will fill after sorting
            }
        
        # Rank by effective contribution
        sorted_features = sorted(
            impact_analysis.items(),
            key=lambda x: x[1]['effective_contribution'],
            reverse=True
        )
        
        for rank, (feature, metrics) in enumerate(sorted_features, 1):
            impact_analysis[feature]['contribution_rank'] = rank
        
        return impact_analysis
    
    def analyze_tier1_alpha_confirmation(self) -> Dict:
        """
        Analyze effectiveness of tier1-alpha confirmation logic.
        
        Questions answered:
        1. How often do alpha_strong and tier1_meaningful both trigger?
        2. When they agree (high confidence), what's the P&L?
        3. When they disagree (low confidence), what's the P&L?
        4. Is the size multiplier (1.2x for agree, 0.7x for disagree) justified?
        """
        df = pd.DataFrame(self.features_history)
        df = df.dropna(subset=['next_day_ret', 'tier1_confirmation_confidence'])
        
        if len(df) == 0:
            return {"error": "No valid confirmation data"}
        
        # Categorize by confirmation confidence
        high_conf = df[df['tier1_confirmation_confidence'] == 'high']
        low_conf = df[df['tier1_confirmation_confidence'] == 'low']
        neutral_conf = df[df['tier1_confirmation_confidence'] == 'neutral']
        
        # Calculate P&L metrics for each category
        def calc_pnl_metrics(subset: pd.DataFrame, label: str) -> Dict:
            if len(subset) == 0:
                return {
                    'label': label,
                    'count': 0,
                    'pct_of_total': 0.0,
                    'avg_ret': 0.0,
                    'median_ret': 0.0,
                    'std_ret': 0.0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'sharpe': 0.0
                }
            
            returns = subset['next_day_ret'].values
            wins = returns > 0
            losses = returns < 0
            
            avg_ret = returns.mean()
            median_ret = np.median(returns)
            std_ret = returns.std()
            sharpe = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
            
            return {
                'label': label,
                'count': len(subset),
                'pct_of_total': len(subset) / len(df) * 100,
                'avg_ret': avg_ret,
                'median_ret': median_ret,
                'std_ret': std_ret,
                'win_rate': wins.sum() / len(wins) * 100 if len(wins) > 0 else 0.0,
                'avg_win': returns[wins].mean() if wins.sum() > 0 else 0.0,
                'avg_loss': returns[losses].mean() if losses.sum() > 0 else 0.0,
                'sharpe': sharpe
            }
        
        high_metrics = calc_pnl_metrics(high_conf, 'HIGH Confidence (Agreement)')
        low_metrics = calc_pnl_metrics(low_conf, 'LOW Confidence (Disagreement)')
        neutral_metrics = calc_pnl_metrics(neutral_conf, 'NEUTRAL (One or both weak)')
        
        # Compare with size multipliers
        # High conf uses 1.2x, low uses 0.7x, neutral uses 1.0x
        high_scaled_ret = high_metrics['avg_ret'] * 1.2 if high_metrics['count'] > 0 else 0
        low_scaled_ret = low_metrics['avg_ret'] * 0.7 if low_metrics['count'] > 0 else 0
        neutral_scaled_ret = neutral_metrics['avg_ret'] * 1.0 if neutral_metrics['count'] > 0 else 0
        
        # Analyze by agreement type
        both_strong = df[
            (df['tier1_confirmation_alpha_strong'] == True) & 
            (df['tier1_confirmation_tier1_meaningful'] == True)
        ]
        
        agree_metrics = calc_pnl_metrics(
            both_strong[both_strong['tier1_confirmation_agreement'] == True],
            'Both Strong + Agree'
        )
        
        disagree_metrics = calc_pnl_metrics(
            both_strong[both_strong['tier1_confirmation_agreement'] == False],
            'Both Strong + Disagree'
        )
        
        return {
            'summary': {
                'total_days': len(df),
                'high_conf_days': high_metrics['count'],
                'low_conf_days': low_metrics['count'],
                'neutral_conf_days': neutral_metrics['count'],
            },
            'by_confidence': {
                'high': high_metrics,
                'low': low_metrics,
                'neutral': neutral_metrics,
            },
            'by_agreement': {
                'agree': agree_metrics,
                'disagree': disagree_metrics,
            },
            'size_multiplier_impact': {
                'high_conf_scaled': high_scaled_ret,
                'low_conf_scaled': low_scaled_ret,
                'neutral_scaled': neutral_scaled_ret,
                'baseline_avg': df['next_day_ret'].mean(),
            }
        }
    
    def plot_feature_distributions(self, top_n: int = 20):
        """Plot distributions of most important features."""
        df = pd.DataFrame(self.features_history)
        
        # Get saturation analysis to pick top features
        saturation = self.analyze_saturation()
        top_features = saturation.head(top_n)['feature'].tolist()
        
        # Plot
        n_cols = 4
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top_features):
            ax = axes[idx]
            values = df[feature].dropna()
            
            # Histogram
            ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(values.mean(), color='red', linestyle='--', label=f'Mean: {values.mean():.3f}')
            ax.axvline(1.0, color='orange', linestyle=':', linewidth=2, label='Saturation')
            ax.axvline(-1.0, color='orange', linestyle=':', linewidth=2)
            
            sat_pct = saturation[saturation.feature==feature]['pct_saturated'].values[0]
            ax.set_title(f'{feature}\nSaturation: {sat_pct:.1f}%', fontsize=9)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        
        # Hide extra subplots
        for idx in range(len(top_features), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'backtest_results/feature_distributions_{self.symbol}.png', dpi=150, bbox_inches='tight')
        print(f"📊 Saved: backtest_results/feature_distributions_{self.symbol}.png")
        plt.close()
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix of all features."""
        df = pd.DataFrame(self.features_history)
        
        # Exclude forward returns and target leakage
        exclude_cols = {
            'forward_return_1d', 'forward_return_3d', 'forward_return_5d',
            'next_day_ret', 'overnight_gap', 'date'
        }
        numeric_df = df.select_dtypes(include=[np.number]).drop(columns=exclude_cols, errors='ignore')
        
        # Compute correlation
        corr = numeric_df.corr()
        
        # Plot
        fig, ax = plt.subplots(figsize=(20, 18))
        sns.heatmap(
            corr,
            cmap='RdBu_r',
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        ax.set_title(f'Feature Correlation Matrix - {self.symbol}', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'backtest_results/feature_correlation_{self.symbol}.png', dpi=150, bbox_inches='tight')
        print(f"📊 Saved: backtest_results/feature_correlation_{self.symbol}.png")
        plt.close()
    
    def analyze_nonlinear_relationships(self, condition_column: str = None, condition_value: str = None) -> pd.DataFrame:
        """Analyze non-linear relationships between features and forward returns.
        
        Args:
            condition_column: Column to filter on (e.g., 'alpha_decision', 'tier1_regime', 'volatility_regime')
            condition_value: Value to filter for (e.g., 'LONG', 'mean_reversion', 'high')
        """
        
        df = pd.DataFrame(self.features_history)
        
        # Apply conditioning filter
        if condition_column and condition_value:
            if condition_column not in df.columns:
                print(f"⚠️  Condition column '{condition_column}' not found in data")
                return pd.DataFrame()
            df = df[df[condition_column] == condition_value].copy()
            if len(df) < 30:
                print(f"⚠️  Insufficient data for condition {condition_column}={condition_value} (n={len(df)})")
                return pd.DataFrame()
        
        results = []
        
        # Get forward returns
        if 'forward_return_1d' not in df.columns:
            print("⚠️  No forward returns available")
            return pd.DataFrame()
        
        # Exclude forward return columns and other target leakage
        exclude_cols = {
            'forward_return_1d', 'forward_return_3d', 'forward_return_5d',
            'next_day_ret', 'overnight_gap',  # These are forward-looking or target
            'date'  # Not a feature
        }
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in exclude_cols:
                continue
                
            feature_values = df[col].dropna()
            return_values = df.loc[feature_values.index, 'forward_return_1d'].dropna()
            
            if len(feature_values) != len(return_values) or len(feature_values) < 50:
                continue
                
            # Align indices
            common_idx = feature_values.index.intersection(return_values.index)
            feature_values = feature_values.loc[common_idx]
            return_values = return_values.loc[common_idx]
            
            # Pearson correlation (existing)
            pearson_corr = feature_values.corr(return_values)
            
            # Spearman correlation (monotonicity) - skip if constant values
            if feature_values.nunique() <= 1:
                spearman_corr = 0.0
                spearman_p = 1.0
            else:
                spearman_corr, spearman_p = spearmanr(feature_values, return_values)
            
            # Mutual information
            mi = mutual_info_regression(feature_values.values.reshape(-1, 1), return_values.values)[0]
            
            # Binned average returns by feature decile
            try:
                feature_deciles = pd.qcut(feature_values, q=10, duplicates='drop')
                binned_returns = return_values.groupby(feature_deciles, observed=False).mean()
                binned_returns.index = [f'D{i+1}' for i in range(len(binned_returns))]
                
                # Calculate edge (D10 - D1)
                if len(binned_returns) >= 2:
                    decile_edge = binned_returns.iloc[-1] - binned_returns.iloc[0]
                else:
                    decile_edge = 0.0
                    
            except Exception as e:
                binned_returns = pd.Series(dtype=float)
                decile_edge = 0.0
            
            results.append({
                'feature': col,
                'condition_column': condition_column,
                'condition_value': condition_value,
                'pearson_corr': pearson_corr,
                'spearman_corr': spearman_corr,
                'spearman_p_value': spearman_p,
                'mutual_info': mi,
                'decile_edge': decile_edge,
                'binned_returns': binned_returns.to_dict() if len(binned_returns) > 0 else {}
            })
        
        return pd.DataFrame(results)
    
    def plot_nonlinear_diagnostics(self):
        """Plot non-linear relationship diagnostics for different conditions."""
        
        # Define conditions to analyze
        conditions = [
            (None, None, "Overall"),
            ("alpha_decision", "LONG", "BUY Signals"),
            ("alpha_decision", "SHORT", "SELL Signals"),
            ("tier1_regime", "mean_reversion", "Mean Reversion Regime"),
            ("tier1_regime", "continuation", "Continuation Regime"),
            ("volatility_regime", "high", "High Volatility"),
            ("volatility_regime", "low", "Low Volatility"),
        ]
        
        # Collect results for all conditions
        all_results = []
        for cond_col, cond_val, cond_name in conditions:
            try:
                results = self.analyze_nonlinear_relationships(cond_col, cond_val)
                if not results.empty:
                    results = results.copy()
                    results['condition_name'] = cond_name
                    all_results.append(results)
            except Exception as e:
                print(f"⚠️  Failed to analyze {cond_name}: {e}")
                continue
        
        if not all_results:
            print("⚠️  No nonlinear diagnostics data available")
            return
            
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Create summary plots
        self._plot_condition_comparison(combined_results)
        self._plot_top_features_by_condition(combined_results)
    
    def _plot_condition_comparison(self, combined_results: pd.DataFrame):
        """Plot comparison of top features across conditions."""
        
        # Get top features by absolute Spearman correlation for each condition
        top_features_per_condition = {}
        for condition in combined_results['condition_name'].unique():
            cond_data = combined_results[combined_results['condition_name'] == condition]
            if not cond_data.empty:
                # Create temporary column for absolute correlation
                cond_data = cond_data.copy()
                cond_data['abs_spearman'] = cond_data['spearman_corr'].abs()
                top_features = cond_data.nlargest(5, 'abs_spearman')
                top_features_per_condition[condition] = top_features.drop('abs_spearman', axis=1)
                top_features_per_condition[condition] = top_features
        
        if not top_features_per_condition:
            return
            
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Feature Performance Across Conditions - {self.symbol}', fontsize=16, fontweight='bold')
        
        # Plot 1: Spearman correlation comparison for first condition
        ax1 = axes[0, 0]
        conditions = list(top_features_per_condition.keys())
        if conditions:
            first_cond = conditions[0]
            first_data = top_features_per_condition[first_cond]
            
            bars = ax1.barh(range(len(first_data)), first_data['spearman_corr'])
            ax1.set_yticks(range(len(first_data)))
            ax1.set_yticklabels(first_data['feature'], fontsize=8)
            ax1.set_xlabel('Spearman Correlation')
            ax1.set_title(f'Top Features: {first_cond}')
            ax1.grid(alpha=0.3)
            ax1.axvline(0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 2: Mutual information comparison
        ax2 = axes[0, 1]
        if conditions:
            first_data = top_features_per_condition[first_cond]
            bars = ax2.barh(range(len(first_data)), first_data['mutual_info'])
            ax2.set_yticks(range(len(first_data)))
            ax2.set_yticklabels(first_data['feature'], fontsize=8)
            ax2.set_xlabel('Mutual Information')
            ax2.set_title(f'Non-Linear Dependencies: {first_cond}')
            ax2.grid(alpha=0.3)
        
        # Plot 3: Decile edges
        ax3 = axes[1, 0]
        if conditions:
            first_data = top_features_per_condition[first_cond]
            colors = ['red' if x < 0 else 'green' for x in first_data['decile_edge']]
            bars = ax3.barh(range(len(first_data)), first_data['decile_edge'], color=colors)
            ax3.set_yticks(range(len(first_data)))
            ax3.set_yticklabels(first_data['feature'], fontsize=8)
            ax3.set_xlabel('Decile Edge (D10 - D1)')
            ax3.set_title(f'Binned Return Edges: {first_cond}')
            ax3.grid(alpha=0.3)
            ax3.axvline(0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 4: Sample size by condition
        ax4 = axes[1, 1]
        condition_counts = combined_results.groupby('condition_name').size()
        bars = ax4.barh(range(len(condition_counts)), condition_counts.values)
        ax4.set_yticks(range(len(condition_counts)))
        ax4.set_yticklabels(condition_counts.index, fontsize=8)
        ax4.set_xlabel('Sample Size')
        ax4.set_title('Data Availability by Condition')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'backtest_results/conditional_diagnostics_{self.symbol}.png', dpi=150, bbox_inches='tight')
        print(f"📊 Saved: backtest_results/conditional_diagnostics_{self.symbol}.png")
        plt.close()
    
    def _plot_top_features_by_condition(self, combined_results: pd.DataFrame):
        """Plot top features for each condition separately."""
        
        conditions = combined_results['condition_name'].unique()
        n_conditions = len(conditions)
        
        if n_conditions == 0:
            return
            
        # Create subplots for each condition
        n_cols = 2
        n_rows = (n_conditions + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if axes.ndim == 2:
            axes = axes.flatten()
        
        for idx, condition in enumerate(conditions):
            ax = axes[idx]
            cond_data = combined_results[combined_results['condition_name'] == condition]
            
            if cond_data.empty:
                ax.text(0.5, 0.5, f'No data for {condition}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{condition} (No Data)')
                continue
                
            # Get top 5 features by absolute Spearman correlation
            cond_data = cond_data.copy()
            cond_data['abs_spearman'] = cond_data['spearman_corr'].abs()
            top_features = cond_data.nlargest(5, 'abs_spearman')
            top_features = top_features.drop('abs_spearman', axis=1)
            
            if top_features.empty:
                ax.text(0.5, 0.5, f'No significant features for {condition}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{condition} (No Significant Features)')
                continue
            
            # Plot Spearman correlations
            colors = ['red' if x < 0 else 'blue' for x in top_features['spearman_corr']]
            bars = ax.barh(range(len(top_features)), top_features['spearman_corr'], color=colors)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'], fontsize=7)
            ax.set_xlabel('Spearman Correlation')
            ax.set_title(f'{condition} (n={len(cond_data)})')
            ax.grid(alpha=0.3)
            ax.axvline(0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(top_features['spearman_corr']):
                ax.text(v + (0.01 if v >= 0 else -0.01), i, f'{v:.2f}', 
                       ha='left' if v >= 0 else 'right', va='center', fontsize=7)
        
        # Hide unused subplots
        for idx in range(n_conditions, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'backtest_results/top_features_by_condition_{self.symbol}.png', dpi=150, bbox_inches='tight')
        print(f"📊 Saved: backtest_results/top_features_by_condition_{self.symbol}.png")
        plt.close()
    
    def plot_confirmation_pnl(self):
        """Plot P&L by confirmation confidence level."""
        conf_results = self.analyze_tier1_alpha_confirmation()
        
        if 'error' in conf_results:
            print("⚠️  Cannot plot confirmation P&L: No data")
            return
        
        # Extract metrics
        by_conf = conf_results['by_confidence']
        labels = []
        avg_rets = []
        win_rates = []
        sharpes = []
        counts = []
        
        for conf_type in ['high', 'low', 'neutral']:
            metrics = by_conf[conf_type]
            labels.append(f"{conf_type.upper()}\n(n={metrics['count']})")
            avg_rets.append(metrics['avg_ret'] * 100)  # Convert to %
            win_rates.append(metrics['win_rate'])
            sharpes.append(metrics['sharpe'])
            counts.append(metrics['count'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Average Return by Confidence
        ax1 = axes[0, 0]
        bars1 = ax1.bar(labels, avg_rets, color=['green', 'red', 'gray'], alpha=0.7)
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        ax1.set_ylabel('Average Return (%)', fontweight='bold')
        ax1.set_title('Average Next-Day Return by Confirmation Level', fontweight='bold')
        ax1.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # Plot 2: Win Rate by Confidence
        ax2 = axes[0, 1]
        bars2 = ax2.bar(labels, win_rates, color=['green', 'red', 'gray'], alpha=0.7)
        ax2.axhline(50, color='black', linestyle='--', linewidth=1, label='50% baseline')
        ax2.set_ylabel('Win Rate (%)', fontweight='bold')
        ax2.set_title('Win Rate by Confirmation Level', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Plot 3: Sharpe Ratio by Confidence
        ax3 = axes[1, 0]
        bars3 = ax3.bar(labels, sharpes, color=['green', 'red', 'gray'], alpha=0.7)
        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        ax3.set_ylabel('Sharpe Ratio (Annualized)', fontweight='bold')
        ax3.set_title('Risk-Adjusted Return by Confirmation Level', fontweight='bold')
        ax3.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # Plot 4: Frequency Distribution
        ax4 = axes[1, 1]
        bars4 = ax4.bar(labels, counts, color=['green', 'red', 'gray'], alpha=0.7)
        ax4.set_ylabel('Number of Days', fontweight='bold')
        ax4.set_title('Frequency of Each Confirmation Level', fontweight='bold')
        ax4.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars4:
            height = bar.get_height()
            pct = height / sum(counts) * 100
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({pct:.1f}%)', ha='center', va='bottom')
        
        plt.suptitle(f'{self.symbol} - Tier1-Alpha Confirmation Analysis', 
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f'backtest_results/confirmation_pnl_{self.symbol}.png', dpi=150, bbox_inches='tight')
        print(f"📊 Saved: backtest_results/confirmation_pnl_{self.symbol}.png")
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive diagnostics report."""
        print(f"\n{'='*80}")
        print(f"FEATURE DIAGNOSTICS REPORT - {self.symbol}")
        print(f"{'='*80}\n")
        
        # 1. Saturation Analysis
        print("=" * 80)
        print("1. FEATURE SATURATION ANALYSIS")
        print("=" * 80)
        print("(Features hitting ±1.0 bounds frequently lose information)\n")
        
        saturation = self.analyze_saturation()
        
        # Show top saturated features
        print("Top 15 Most Saturated Features:")
        print(saturation.head(15)[['feature', 'pct_saturated', 'pct_at_+1', 'pct_at_-1', 'mean', 'std']].to_string(index=False))
        
        # Flag problematic features
        problematic = saturation[saturation['pct_saturated'] > 10.0]
        if not problematic.empty:
            print(f"\n⚠️  WARNING: {len(problematic)} features saturated >10% of the time:")
            for _, row in problematic.iterrows():
                print(f"   - {row['feature']}: {row['pct_saturated']:.1f}% saturated")
        
        # 2. Distribution Analysis
        print(f"\n{'='*80}")
        print("2. FEATURE DISTRIBUTION ANALYSIS")
        print("=" * 80)
        print("(Narrow distributions mean feature isn't using full range)\n")
        
        narrow_range = saturation[saturation['effective_range_10_90'] < 0.5]
        if not narrow_range.empty:
            print(f"⚠️  {len(narrow_range)} features with narrow effective range (<0.5):")
            print(narrow_range[['feature', 'effective_range_10_90', 'min', 'max']].head(10).to_string(index=False))
        
        # 3. Weight Impact Analysis
        print(f"\n{'='*80}")
        print("3. WEIGHT IMPACT ANALYSIS")
        print("=" * 80)
        print("(Does the weight actually matter given feature variability?)\n")
        
        impact = self.analyze_weight_impact()
        if impact:
            impact_df = pd.DataFrame(impact).T.sort_values('effective_contribution', ascending=False)
            print("Features by Effective Contribution (weight × std):")
            print(impact_df[['weight', 'std', 'effective_contribution', 'contribution_rank']].to_string())
            
            # Check for mismatches
            print("\n🔍 Weight vs Impact Mismatches:")
            mismatch_found = False
            for feature, metrics in impact.items():
                weight_rank = sorted(impact.items(), key=lambda x: x[1]['weight'], reverse=True).index((feature, metrics)) + 1
                impact_rank = metrics['contribution_rank']
                
                if abs(weight_rank - impact_rank) > 2:
                    print(f"   - {feature}: Weight rank #{weight_rank} but Impact rank #{impact_rank}")
                    mismatch_found = True
            
            if not mismatch_found:
                print("   ✓ No major mismatches detected")
        
        # 4. Tier1-Alpha Confirmation Analysis
        print(f"\n{'='*80}")
        print("4. TIER1-ALPHA CONFIRMATION ANALYSIS")
        print("=" * 80)
        print("(Does the confirmation logic actually improve P&L?)\n")
        
        conf_analysis = self.analyze_tier1_alpha_confirmation()
        
        if 'error' not in conf_analysis:
            summary = conf_analysis['summary']
            by_conf = conf_analysis['by_confidence']
            by_agree = conf_analysis['by_agreement']
            sizing = conf_analysis['size_multiplier_impact']
            
            print(f"Total Trading Days: {summary['total_days']}")
            print(f"  - HIGH confidence: {summary['high_conf_days']} ({summary['high_conf_days']/summary['total_days']*100:.1f}%)")
            print(f"  - LOW confidence: {summary['low_conf_days']} ({summary['low_conf_days']/summary['total_days']*100:.1f}%)")
            print(f"  - NEUTRAL: {summary['neutral_conf_days']} ({summary['neutral_conf_days']/summary['total_days']*100:.1f}%)")
            
            print("\nPerformance by Confidence Level:")
            print("-" * 80)
            for conf_type, metrics in by_conf.items():
                print(f"\n{conf_type.upper()} Confidence:")
                print(f"  Count: {metrics['count']} days ({metrics['pct_of_total']:.1f}%)")
                print(f"  Avg Return: {metrics['avg_ret']*100:+.3f}% (Median: {metrics['median_ret']*100:+.3f}%)")
                print(f"  Win Rate: {metrics['win_rate']:.1f}%")
                print(f"  Avg Win: {metrics['avg_win']*100:+.3f}% | Avg Loss: {metrics['avg_loss']*100:.3f}%")
                print(f"  Sharpe: {metrics['sharpe']:.2f}")
            
            print("\n" + "=" * 80)
            print("SIZE MULTIPLIER IMPACT:")
            print("=" * 80)
            print(f"Baseline (no adjustment): {sizing['baseline_avg']*100:+.3f}%")
            print(f"HIGH conf (1.2x sizing): {sizing['high_conf_scaled']*100:+.3f}%")
            print(f"LOW conf (0.7x sizing): {sizing['low_conf_scaled']*100:+.3f}%")
            print(f"NEUTRAL (1.0x sizing): {sizing['neutral_scaled']*100:+.3f}%")
            
            # Calculate if sizing improves results
            weighted_avg = (
                by_conf['high']['avg_ret'] * by_conf['high']['count'] * 1.2 +
                by_conf['low']['avg_ret'] * by_conf['low']['count'] * 0.7 +
                by_conf['neutral']['avg_ret'] * by_conf['neutral']['count'] * 1.0
            ) / summary['total_days']
            
            improvement = (weighted_avg - sizing['baseline_avg']) / abs(sizing['baseline_avg']) * 100 if sizing['baseline_avg'] != 0 else 0
            
            print(f"\nWeighted Average (with sizing): {weighted_avg*100:+.3f}%")
            print(f"Improvement vs Baseline: {improvement:+.1f}%")
            
            if improvement > 5:
                print("✓ Size multiplier logic IMPROVES results")
            elif improvement < -5:
                print("⚠️  Size multiplier logic HURTS results")
            else:
                print("→ Size multiplier has MINIMAL impact")
            
            # Agreement analysis
            print("\n" + "=" * 80)
            print("AGREEMENT vs DISAGREEMENT:")
            print("=" * 80)
            agree = by_agree['agree']
            disagree = by_agree['disagree']
            
            print(f"\nBoth Strong + AGREE:")
            print(f"  Count: {agree['count']} days ({agree['pct_of_total']:.1f}%)")
            print(f"  Avg Return: {agree['avg_ret']*100:+.3f}%")
            print(f"  Win Rate: {agree['win_rate']:.1f}%")
            print(f"  Sharpe: {agree['sharpe']:.2f}")
            
            print(f"\nBoth Strong + DISAGREE:")
            print(f"  Count: {disagree['count']} days ({disagree['pct_of_total']:.1f}%)")
            print(f"  Avg Return: {disagree['avg_ret']*100:+.3f}%")
            print(f"  Win Rate: {disagree['win_rate']:.1f}%")
            print(f"  Sharpe: {disagree['sharpe']:.2f}")
            
            if agree['avg_ret'] > disagree['avg_ret'] * 1.5:
                print("\n✓ Agreement signals are SIGNIFICANTLY better")
            elif disagree['avg_ret'] > agree['avg_ret']:
                print("\n⚠️  Disagreement signals actually OUTPERFORM - logic may be inverted!")
            else:
                print("\n→ Marginal difference between agreement and disagreement")
            
            # Threshold diagnostic
            print("\n" + "=" * 80)
            print("THRESHOLD DIAGNOSTIC:")
            print("=" * 80)
            
            # Create DataFrame for analysis (if not already created)
            if 'df' not in locals():
                df = pd.DataFrame(self.features_history)
            
            # Analyze why confirmation doesn't trigger
            alpha_strong_count = df[df['tier1_confirmation_alpha_strong'] == True].shape[0]
            tier1_meaningful_count = df[df['tier1_confirmation_tier1_meaningful'] == True].shape[0]
            both_true_count = df[
                (df['tier1_confirmation_alpha_strong'] == True) & 
                (df['tier1_confirmation_tier1_meaningful'] == True)
            ].shape[0]
            
            print(f"\nCondition Trigger Rates:")
            print(f"  alpha_strong (|alpha_score| >= 0.08): {alpha_strong_count} days ({alpha_strong_count/len(df)*100:.1f}%)")
            print(f"  tier1_meaningful (|tier1_score| >= 0.08): {tier1_meaningful_count} days ({tier1_meaningful_count/len(df)*100:.1f}%)")
            print(f"  BOTH conditions true: {both_true_count} days ({both_true_count/len(df)*100:.1f}%)")
            
            # Score distribution analysis
            print(f"\nScore Distribution Analysis:")
            alpha_vals = df['alpha_score'].dropna()
            tier1_vals = df['tier1_score'].dropna()
            
            print(f"\nalpha_score:")
            print(f"  Mean: {alpha_vals.mean():+.3f}, Std: {alpha_vals.std():.3f}")
            print(f"  Range: [{alpha_vals.min():+.3f}, {alpha_vals.max():+.3f}]")
            print(f"  50th pct: {alpha_vals.quantile(0.50):+.3f}")
            print(f"  75th pct: {alpha_vals.quantile(0.75):+.3f}")
            print(f"  90th pct: {alpha_vals.quantile(0.90):+.3f}")
            print(f"  95th pct: {alpha_vals.quantile(0.95):+.3f}")
            print(f"  % above +0.08: {(alpha_vals >= 0.08).sum() / len(alpha_vals) * 100:.1f}%")
            print(f"  % below -0.08: {(alpha_vals <= -0.08).sum() / len(alpha_vals) * 100:.1f}%")
            print(f"  % with |score| >= 0.08: {(alpha_vals.abs() >= 0.08).sum() / len(alpha_vals) * 100:.1f}%")
            
            print(f"\ntier1_score:")
            print(f"  Mean: {tier1_vals.mean():+.3f}, Std: {tier1_vals.std():.3f}")
            print(f"  Range: [{tier1_vals.min():+.3f}, {tier1_vals.max():+.3f}]")
            print(f"  50th pct: {tier1_vals.quantile(0.50):+.3f}")
            print(f"  75th pct: {tier1_vals.quantile(0.75):+.3f}")
            print(f"  90th pct: {tier1_vals.quantile(0.90):+.3f}")
            print(f"  95th pct: {tier1_vals.quantile(0.95):+.3f}")
            print(f"  % above +0.08: {(tier1_vals >= 0.08).sum() / len(tier1_vals) * 100:.1f}%")
            print(f"  % below -0.08: {(tier1_vals <= -0.08).sum() / len(tier1_vals) * 100:.1f}%")
            print(f"  % with |score| >= 0.08: {(tier1_vals.abs() >= 0.08).sum() / len(tier1_vals) * 100:.1f}%")
            
            # Suggest better thresholds
            print(f"\n🔧 SUGGESTED THRESHOLDS:")
            
            # Find thresholds that would trigger 15-20% of the time
            alpha_70th = alpha_vals.abs().quantile(0.70)
            alpha_80th = alpha_vals.abs().quantile(0.80)
            tier1_60th = tier1_vals.abs().quantile(0.60)
            tier1_70th = tier1_vals.abs().quantile(0.70)
            
            print(f"\nFor ~30% alpha_strong trigger rate:")
            print(f"  Current: alpha_threshold = 0.08 (triggers {alpha_strong_count/len(df)*100:.1f}%)")
            print(f"  Suggested: alpha_threshold = {alpha_70th:.3f} (would trigger ~30%)")
            
            print(f"\nFor ~40% tier1_meaningful trigger rate:")
            print(f"  Current: tier1_threshold = 0.08 (triggers {tier1_meaningful_count/len(df)*100:.1f}%)")
            print(f"  Suggested: tier1_threshold = {tier1_60th:.3f} (would trigger ~40%)")
            
            # Estimate overlap with suggested thresholds
            would_be_both = df[
                (df['alpha_score'].abs() >= alpha_70th) & 
                (df['tier1_score'].abs() >= tier1_60th)
            ].shape[0]
            
            print(f"\nWith suggested thresholds:")
            print(f"  BOTH would trigger: {would_be_both} days ({would_be_both/len(df)*100:.1f}%)")
            print(f"  Current BOTH triggers: {both_true_count} days ({both_true_count/len(df)*100:.1f}%)")
            print(f"  Improvement: {would_be_both - both_true_count} additional days (+{(would_be_both - both_true_count)/len(df)*100:.1f}%)")
        
        # 5. Regime Distribution
        print(f"\n{'='*80}")
        print("5. REGIME DISTRIBUTION")
        print("=" * 80)
        
        df = pd.DataFrame(self.features_history)
        if 'alpha_regime' in df.columns:
            regime_counts = df['alpha_regime'].value_counts()
            regime_pcts = (regime_counts / len(df) * 100).round(1)
            print("\nAlpha Regime Distribution:")
            for regime, pct in regime_pcts.items():
                print(f"   {regime}: {pct}% ({regime_counts[regime]} days)")
        
        if 'tier1_regime' in df.columns:
            regime_counts = df['tier1_regime'].value_counts()
            regime_pcts = (regime_counts / len(df) * 100).round(1)
            print("\nTier-1 Regime Distribution:")
            for regime, pct in regime_pcts.items():
                print(f"   {regime}: {pct}% ({regime_counts[regime]} days)")
        
        # 6. Decision Distribution
        print(f"\n{'='*80}")
        print("6. SIGNAL DISTRIBUTION")
        print("=" * 80)
        
        if 'alpha_decision' in df.columns:
            decision_counts = df['alpha_decision'].value_counts()
            decision_pcts = (decision_counts / len(df) * 100).round(1)
            print("\nAlpha Decision Distribution:")
            for decision, pct in decision_pcts.items():
                print(f"   {decision}: {pct}% ({decision_counts[decision]} days)")
        
        # 7. Summary Recommendations
        print(f"\n{'='*80}")
        print("7. RECOMMENDATIONS")
        print("=" * 80)
        
        recommendations = []
        
        # Check saturation
        high_saturation = saturation[saturation['pct_saturated'] > 15.0]
        if not high_saturation.empty:
            recommendations.append(
                f"⚠️  {len(high_saturation)} features saturated >15%: Consider widening normalization bounds"
            )
        
        # Check narrow distributions
        if len(narrow_range) > 5:
            recommendations.append(
                f"⚠️  Many features have narrow ranges: Consider rescaling or using z-scores"
            )
        
        # Check score compression
        for score_col in ['alpha_score', 'tier1_score', 'mom_score', 'trend_score']:
            if score_col in df.columns:
                vals = df[score_col].dropna()
                if vals.std() < 0.15:
                    recommendations.append(
                        f"⚠️  {score_col} has low variability (std={vals.std():.3f}): May need amplification"
                    )
        
        # Check confirmation logic
        if 'error' not in conf_analysis:
            if improvement < -5:
                recommendations.append(
                    "⚠️  Size multiplier logic is hurting performance - consider revising"
                )
            
            if disagree['avg_ret'] > agree['avg_ret'] and disagree['count'] > 10:
                recommendations.append(
                    "⚠️  Disagreement signals outperform agreement - confirmation logic may be backwards"
                )
        
        if recommendations:
            print("\n" + "\n".join(recommendations))
        else:
            print("\n✓ No major issues detected")
        
        # Save report
        report_path = f'backtest_results/feature_diagnostics_{self.symbol}.txt'
        with open(report_path, 'w') as f:
            f.write(f"FEATURE DIAGNOSTICS REPORT - {self.symbol}\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Period: {self.start_date} to {self.end_date}\n\n")
            f.write(saturation.to_string())
        
        print(f"\n📄 Full report saved: {report_path}")


def main():
    if len(sys.argv) < 4:
        print("Usage: python feature_diagnostics.py SYMBOL START_DATE END_DATE [EXECUTION_MODEL]")
        print("Example: python feature_diagnostics.py AAPL 2020-01-01 2024-12-31 next_open_to_close")
        print("Execution models: close_to_close, next_open_to_close, next_open_to_next_open")
        sys.exit(1)
    
    symbol = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    execution_model = sys.argv[4] if len(sys.argv) > 4 else "next_open_to_close"
    
    # Validate execution model
    valid_models = ["close_to_close", "next_open_to_close", "next_open_to_next_open"]
    if execution_model not in valid_models:
        print(f"Invalid execution model: {execution_model}")
        print(f"Valid options: {', '.join(valid_models)}")
        sys.exit(1)
    
    # Run diagnostics
    diagnostics = FeatureDiagnostics(symbol, start_date, end_date, execution_model)
    diagnostics.collect_features_over_time()
    diagnostics.generate_report()
    diagnostics.plot_feature_distributions()
    diagnostics.plot_correlation_matrix()
    diagnostics.plot_nonlinear_diagnostics()
    diagnostics.plot_confirmation_pnl()
    
    print(f"\n{'='*80}")
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
