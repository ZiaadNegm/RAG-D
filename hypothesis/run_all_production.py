#!/usr/bin/env python -u
"""
Run all hypotheses on production data.

Usage:
    nohup python -u hypothesis/run_all_production.py > /mnt/hypothesis_outputs/production_run.log 2>&1 &
"""

import gc
import json
import sys
from datetime import datetime
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd

# Production run path
RUN_DIR = '/mnt/tmp/warp_measurements/production_beir_quora/runs/metrics_production_20260104_115425'
OUTPUT_DIR = Path('/mnt/hypothesis_outputs/production')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"{'='*70}")
    print(f"HYPOTHESIS PRODUCTION RUN")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Run dir: {RUN_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")
    
    # Load config
    from hypothesis.configs import load_config
    config = load_config('prod', override_run_dir=RUN_DIR)
    
    # Build or load cluster_frame
    cache_path = OUTPUT_DIR / 'cluster_frame.parquet'
    if cache_path.exists():
        print(f"Loading cached cluster_frame from {cache_path}")
        cluster_frame = pd.read_parquet(cache_path)
    else:
        print("Building cluster_frame...")
        from hypothesis.data.standardized_tables import ClusterFrameBuilder
        from hypothesis.data import MetricsLoader
        
        loader = MetricsLoader(
            index_path=config.paths.index_path,
            run_dir=config.paths.run_dir,
            chunk_size=500,
            verbose=True
        )
        builder = ClusterFrameBuilder(config, loader)
        cluster_frame = builder.build(force_rebuild=True)
        cluster_frame.to_parquet(cache_path)
        print(f"Saved cluster_frame to {cache_path}")
    
    print(f"Cluster frame shape: {cluster_frame.shape}")
    print(f"Columns: {list(cluster_frame.columns)}\n")
    
    # Import all hypothesis classes
    from hypothesis.hypotheses.template import H4_ConcentrationRedundancy, H5_DispersionMisses
    from hypothesis.hypotheses.h3_doc_diversity import H3_DocDiversityRedundancy
    from hypothesis.hypotheses.h10_hubness_redundancy import H10_HubnessRedundancy
    from hypothesis.hypotheses.h15_miss_severity import H15_MissSeverity
    from hypothesis.hypotheses.h17_borderline_clusters import H17_BorderlineClusters
    
    hypotheses = [
        ('H3', H3_DocDiversityRedundancy),
        ('H4', H4_ConcentrationRedundancy),
        ('H5', H5_DispersionMisses),
        ('H10', H10_HubnessRedundancy),
        ('H15', H15_MissSeverity),
        ('H17', H17_BorderlineClusters),
    ]
    
    results = {}
    
    for h_id, HypothesisClass in hypotheses:
        print(f"\n{'='*70}")
        print(f"Running: {h_id}")
        print(f"{'='*70}")
        
        try:
            # Create hypothesis instance
            h = HypothesisClass(config)
            h.cluster_frame = cluster_frame
            h.output_dir = OUTPUT_DIR / h_id
            h.output_dir.mkdir(parents=True, exist_ok=True)
            
            # For H15, set empty severity frame to use fallback
            if h_id == 'H15':
                h.centroid_severity = pd.DataFrame()
            
            # Analyze
            print("Analyzing...")
            result = h.analyze()
            
            # Visualize
            print("Visualizing...")
            h.visualize()
            
            # Save result
            result.save(str(h.output_dir))
            
            # Store summary
            results[h_id] = {
                'supported': result.supported,
                'effect_size': float(result.effect_size) if result.effect_size == result.effect_size else None,
                'effect_size_ci': list(result.effect_size_ci),
                'p_value': float(result.p_value) if result.p_value == result.p_value else None,
                'n_observations': result.n_observations,
                'timestamp': result.timestamp,
            }
            
            status = '✅ SUPPORTED' if result.supported else '❌ NOT SUPPORTED'
            print(f"\nResult: {status}")
            print(f"Effect size: {result.effect_size:.4f}" if result.effect_size == result.effect_size else "Effect size: N/A")
            print(f"p-value: {result.p_value:.2e}" if result.p_value == result.p_value else "p-value: N/A")
            print(f"n={result.n_observations}")
            
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
            results[h_id] = {'error': str(e)}
        
        gc.collect()
    
    # Save summary
    summary_path = OUTPUT_DIR / 'results_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved summary to {summary_path}")
    
    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for h_id, res in results.items():
        if 'error' in res:
            print(f"{h_id}: ERROR - {res['error'][:60]}")
        else:
            status = '✅ SUPPORTED' if res['supported'] else '❌ NOT SUPPORTED'
            effect = res['effect_size']
            pval = res['p_value']
            n = res['n_observations']
            if effect is None:
                print(f"{h_id}: {status} | effect=N/A | n={n}")
            else:
                print(f"{h_id}: {status} | effect={effect:.4f} | p={pval:.2e} | n={n}")
    
    print(f"\nCompleted: {datetime.now().isoformat()}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
