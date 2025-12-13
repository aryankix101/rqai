import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from liquidity_gate_filter import apply_liquidity_gate, filter_universe_by_liquidity
from regime_classifier import classify_market_regime, classify_universe_regimes
from composite_scores import calculate_all_composites
from deterministic_scoring import calculate_tier1_master_score, score_universe, rank_by_score
from technical_llm import analyze_technical_json
from signal_fusion import fuse_signals, fuse_universe_signals, filter_and_rank_signals

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineConfig:
    INDICATORS_DIR = Path("./output")
    RESULTS_DIR = Path("./signals")
    RUN_LIQUIDITY_GATE = True
    RUN_REGIME_CLASSIFICATION = True
    RUN_COMPOSITE_SCORING = True
    RUN_DETERMINISTIC_SCORING = True
    RUN_LLM_REFINEMENT = True
    RUN_SIGNAL_FUSION = True
    MIN_CONFIDENCE = 0.30
    MIN_SIGNAL_STRENGTH = 0.15
    
    # LLM parameters
    LLM_ENABLED = True
    LLM_TIMEOUT = 30
    
    # Output options
    SAVE_INTERMEDIATE_RESULTS = True
    VERBOSE_LOGGING = True


# =============================================================================
# Main Pipeline Class
# =============================================================================

class SignalGenerationPipeline:
    """
    End-to-end signal generation pipeline.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: PipelineConfig instance (uses default if None)
        """
        self.config = config or PipelineConfig()
        self.results = {}
        
        # Create results directory
        self.config.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    def load_indicators(self) -> Dict[str, Dict[str, Any]]:
        """
        Load technical indicators from JSON files.
        
        Returns:
            Dictionary mapping symbol -> indicators
        """
        logger.info(f"\n{'='*80}")
        logger.info("STAGE 0: Loading Technical Indicators")
        logger.info(f"{'='*80}")
        
        indicators_dir = self.config.INDICATORS_DIR
        
        if not indicators_dir.exists():
            raise FileNotFoundError(f"Indicators directory not found: {indicators_dir}")
        
        indicators_dict = {}
        json_files = list(indicators_dir.glob("*_technical_indicators.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No indicator JSON files found in {indicators_dir}")
        
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
                symbol = data['id']['symbol']
                indicators_dict[symbol] = data
        
        logger.info(f"Loaded {len(indicators_dict)} symbols: {', '.join(indicators_dict.keys())}")
        self.results['indicators'] = indicators_dict
        
        return indicators_dict
    
    def run_liquidity_filter(
        self,
        indicators_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Stage 1: Apply liquidity gate filter.
        
        Args:
            indicators_dict: Raw indicators for all symbols
        
        Returns:
            Filtered indicators dictionary (only passing symbols)
        """
        if not self.config.RUN_LIQUIDITY_GATE:
            logger.info("Skipping liquidity gate (disabled)")
            return indicators_dict
        
        logger.info(f"\n{'='*80}")
        logger.info("STAGE 1: Liquidity Gate Filter")
        logger.info(f"{'='*80}")
        
        filter_results = filter_universe_by_liquidity(
            indicators_dict,
            verbose=self.config.VERBOSE_LOGGING
        )
        
        # Keep only passing symbols
        passed_symbols = filter_results['passed']
        filtered_indicators = {
            symbol: indicators_dict[symbol]
            for symbol in passed_symbols
        }
        
        self.results['liquidity_filter'] = filter_results
        
        logger.info(f"\n✓ Passed liquidity gate: {len(passed_symbols)}/{len(indicators_dict)}")
        
        if self.config.SAVE_INTERMEDIATE_RESULTS:
            self._save_json('liquidity_filter_results.json', filter_results)
        
        return filtered_indicators
    
    def run_regime_classification(
        self,
        indicators_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Stage 2: Classify market regimes.
        
        Args:
            indicators_dict: Indicators for symbols
        
        Returns:
            Dictionary mapping symbol -> regime classification
        """
        if not self.config.RUN_REGIME_CLASSIFICATION:
            logger.info("Skipping regime classification (disabled)")
            return {}
        
        logger.info(f"\n{'='*80}")
        logger.info("STAGE 2: Regime Classification")
        logger.info(f"{'='*80}")
        
        regimes = classify_universe_regimes(
            indicators_dict,
            verbose=self.config.VERBOSE_LOGGING
        )
        
        self.results['regimes'] = regimes
        
        if self.config.SAVE_INTERMEDIATE_RESULTS:
            self._save_json('regime_classifications.json', regimes)
        
        return regimes
    
    def run_composite_scoring(
        self,
        indicators_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Stage 3: Calculate composite scores.
        
        Args:
            indicators_dict: Indicators for symbols
        
        Returns:
            Dictionary mapping symbol -> composite scores
        """
        if not self.config.RUN_COMPOSITE_SCORING:
            logger.info("Skipping composite scoring (disabled)")
            return {}
        
        logger.info(f"\n{'='*80}")
        logger.info("STAGE 3: Composite Scoring")
        logger.info(f"{'='*80}")
        
        composites = {}
        for symbol, indicators in indicators_dict.items():
            composites[symbol] = calculate_all_composites(indicators)
            
            if self.config.VERBOSE_LOGGING:
                mom = composites[symbol]['momentum_composite']['score']
                trend = composites[symbol]['trend_composite']['score']
                rs = composites[symbol]['rs_composite']['score']
                vol = composites[symbol]['volume_composite']['score']
                risk = composites[symbol]['risk_composite']['score']
                
                logger.info(f"{symbol}: Mom={mom:+.2f} Trend={trend:+.2f} RS={rs:+.2f} Vol={vol:.2f} Risk={risk:.2f}")
        
        self.results['composites'] = composites
        
        if self.config.SAVE_INTERMEDIATE_RESULTS:
            self._save_json('composite_scores.json', composites)
        
        return composites
    
    def run_deterministic_scoring(
        self,
        indicators_dict: Dict[str, Dict[str, Any]],
        composites: Dict[str, Dict[str, Any]],
        regimes: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Stage 4: Calculate Tier-1 deterministic scores.
        
        Args:
            indicators_dict: Raw indicators
            composites: Composite scores
            regimes: Regime classifications
        
        Returns:
            Dictionary mapping symbol -> tier1 score
        """
        if not self.config.RUN_DETERMINISTIC_SCORING:
            logger.info("Skipping deterministic scoring (disabled)")
            return {}
        
        logger.info(f"\n{'='*80}")
        logger.info("STAGE 4: Deterministic Scoring (Tier-1)")
        logger.info(f"{'='*80}")
        
        tier1_scores = score_universe(
            indicators_dict=indicators_dict,
            composites_dict=composites,
            regimes_dict=regimes,
            verbose=self.config.VERBOSE_LOGGING
        )
        
        self.results['tier1_scores'] = tier1_scores
        
        if self.config.SAVE_INTERMEDIATE_RESULTS:
            self._save_json('tier1_deterministic_scores.json', tier1_scores)
        
        return tier1_scores
    
    def run_llm_refinement(
        self,
        indicators_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Stage 5: LLM refinement of signals.
        
        Args:
            indicators_dict: Raw indicators
        
        Returns:
            Dictionary mapping symbol -> LLM verdict
        """
        if not self.config.RUN_LLM_REFINEMENT or not self.config.LLM_ENABLED:
            logger.info("Skipping LLM refinement (disabled)")
            # Return mock results
            return {
                symbol: {"score": 0.0, "confidence": 0.5, "tone": "neutral"}
                for symbol in indicators_dict.keys()
            }
        
        logger.info(f"\n{'='*80}")
        logger.info("STAGE 5: LLM Refinement")
        logger.info(f"{'='*80}")
        
        llm_results = {}
        
        for symbol, indicators in indicators_dict.items():
            try:
                logger.info(f"Analyzing {symbol} with LLM...")
                verdict = analyze_technical_json(indicators)
                llm_results[symbol] = verdict
                
                if self.config.VERBOSE_LOGGING:
                    tone = verdict.get('tone', 'unknown')
                    score = verdict.get('score', 0)
                    conf = verdict.get('confidence', 0)
                    logger.info(f"  {symbol}: {tone} (score={score:+.2f}, conf={conf:.2f})")
            
            except Exception as e:
                logger.error(f"LLM analysis failed for {symbol}: {e}")
                # Fallback to neutral
                llm_results[symbol] = {
                    "score": 0.0,
                    "confidence": 0.3,
                    "tone": "neutral",
                    "error": str(e)
                }
        
        self.results['llm_results'] = llm_results
        
        if self.config.SAVE_INTERMEDIATE_RESULTS:
            self._save_json('llm_verdicts.json', llm_results)
        
        return llm_results
    
    def run_signal_fusion(
        self,
        tier1_scores: Dict[str, Dict[str, Any]],
        llm_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Stage 6: Fuse deterministic and LLM signals.
        
        Args:
            tier1_scores: Tier-1 deterministic scores
            llm_results: LLM verdicts
        
        Returns:
            Dictionary mapping symbol -> fused signal
        """
        if not self.config.RUN_SIGNAL_FUSION:
            logger.info("Skipping signal fusion (disabled)")
            return {}
        
        logger.info(f"\n{'='*80}")
        logger.info("STAGE 6: Signal Fusion")
        logger.info(f"{'='*80}")
        
        fused_signals = fuse_universe_signals(
            tier1_scores=tier1_scores,
            llm_results=llm_results,
            verbose=self.config.VERBOSE_LOGGING
        )
        
        self.results['fused_signals'] = fused_signals
        
        if self.config.SAVE_INTERMEDIATE_RESULTS:
            self._save_json('fused_signals.json', fused_signals)
        
        return fused_signals
    
    def generate_final_signals(
        self,
        fused_signals: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        logger.info(f"\n{'='*80}")
        logger.info("STAGE 7: Final Signal Generation")
        logger.info(f"{'='*80}")
        
        # Filter and rank
        ranked = filter_and_rank_signals(
            fused_signals,
            min_confidence=self.config.MIN_CONFIDENCE,
            min_signal_strength=self.config.MIN_SIGNAL_STRENGTH
        )
        
        # Prepare final output
        final_signals = {
            "generated_at": datetime.now().isoformat(),
            "universe_size": len(fused_signals),
            "long_candidates": len(ranked['longs']),
            "short_candidates": len(ranked['shorts']),
            "neutral": len(ranked['neutral']),
            "longs": [
                {
                    "symbol": symbol,
                    "signal": signal,
                    "confidence": confidence,
                    "rank": i + 1
                }
                for i, (symbol, signal, confidence) in enumerate(ranked['longs'])
            ],
            "shorts": [
                {
                    "symbol": symbol,
                    "signal": signal,
                    "confidence": confidence,
                    "rank": i + 1
                }
                for i, (symbol, signal, confidence) in enumerate(ranked['shorts'])
            ]
        }
        
        self.results['final_signals'] = final_signals
        
        # Display results
        logger.info(f"\n{'='*80}")
        logger.info("FINAL SIGNALS")
        logger.info(f"{'='*80}")
        logger.info(f"Universe: {final_signals['universe_size']} symbols")
        logger.info(f"LONG candidates: {final_signals['long_candidates']}")
        logger.info(f"SHORT candidates: {final_signals['short_candidates']}")
        
        if ranked['longs']:
            logger.info(f"\n🟢 TOP LONGS:")
            for entry in final_signals['longs'][:5]:  # Top 5
                logger.info(f"  #{entry['rank']} {entry['symbol']}: {entry['signal']:+.3f} (conf={entry['confidence']:.2f})")
        
        if ranked['shorts']:
            logger.info(f"\n🔴 TOP SHORTS:")
            for entry in final_signals['shorts'][:5]:  # Top 5
                logger.info(f"  #{entry['rank']} {entry['symbol']}: {entry['signal']:+.3f} (conf={entry['confidence']:.2f})")
        
        # Save final signals
        self._save_json('final_trading_signals.json', final_signals)
        
        return final_signals
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Execute complete signal generation pipeline.
        
        Returns:
            Final trading signals
        """
        logger.info("\n" + "="*80)
        logger.info("SIGNAL GENERATION PIPELINE - START")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        try:
            # Stage 0: Load indicators
            indicators = self.load_indicators()
            
            # Stage 1: Liquidity filter
            filtered_indicators = self.run_liquidity_filter(indicators)
            
            if not filtered_indicators:
                logger.error("No symbols passed liquidity filter!")
                return {}
            
            # Stage 2: Regime classification
            regimes = self.run_regime_classification(filtered_indicators)
            
            # Stage 3: Composite scoring
            composites = self.run_composite_scoring(filtered_indicators)
            
            # Stage 4: Deterministic scoring
            tier1_scores = self.run_deterministic_scoring(
                filtered_indicators, composites, regimes
            )
            
            # Stage 5: LLM refinement
            llm_results = self.run_llm_refinement(filtered_indicators)
            
            # Stage 6: Signal fusion
            fused_signals = self.run_signal_fusion(tier1_scores, llm_results)
            
            # Stage 7: Final signals
            final_signals = self.generate_final_signals(fused_signals)
            
            # Calculate elapsed time
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"\n{'='*80}")
            logger.info(f"PIPELINE COMPLETE - {elapsed:.1f} seconds")
            logger.info(f"{'='*80}\n")
            
            return final_signals
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _save_json(self, filename: str, data: Any) -> None:
        """Save data to JSON file in results directory."""
        filepath = self.config.RESULTS_DIR / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug(f"Saved: {filepath}")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main entry point for CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Signal Generation Pipeline - Generate trading signals from technical indicators"
    )
    parser.add_argument(
        "--indicators-dir",
        type=str,
        default="./output",
        help="Directory containing technical indicator JSON files"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./signals",
        help="Directory to save signal results"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM refinement"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.40,
        help="Minimum confidence threshold for signals"
    )
    
    args = parser.parse_args()
    
    # Configure pipeline
    config = PipelineConfig()
    config.INDICATORS_DIR = Path(args.indicators_dir)
    config.RESULTS_DIR = Path(args.results_dir)
    config.LLM_ENABLED = not args.no_llm
    config.MIN_CONFIDENCE = args.min_confidence
    
    # Run pipeline
    pipeline = SignalGenerationPipeline(config)
    final_signals = pipeline.run_full_pipeline()
    
    # Print summary
    print("\n" + "="*80)
    print("TRADING SIGNALS GENERATED")
    print("="*80)
    print(f"Results saved to: {config.RESULTS_DIR}/final_trading_signals.json")
    print(f"LONG candidates: {final_signals.get('long_candidates', 0)}")
    print(f"SHORT candidates: {final_signals.get('short_candidates', 0)}")
    print("="*80)


if __name__ == "__main__":
    main()
