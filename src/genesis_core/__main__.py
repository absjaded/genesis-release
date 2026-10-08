"""Run bounded research examples. All commands print; none write research data."""
import argparse
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser(description='Genesis: analysis, controlled construction and numerical qualification')
    parser.add_argument('command', choices=('analysis', 'construction', 'records', 'qualification'))
    parser.add_argument('--json', action='store_true', help='print structured results')
    parser.add_argument('--demo', action='store_true', help='run the analysis interface on synthetic example arrays')
    parser.add_argument('--source', help='source .npy matrix; explicit input only')
    parser.add_argument('--target', help='matched target .npy matrix')
    parser.add_argument('--categories', help='integer category .npy vector in matching row order')
    parser.add_argument('--family', choices=('ridge', 'pls'), default='ridge')
    parser.add_argument('--seed', type=int, help='evaluation seed for candidate sampling, null permutations and bootstrap')
    parser.add_argument('--all-models', action='store_true', help='include diagnostic own-center entropy bounds for all seven recorded models')
    args = parser.parse_args()
    if args.command != 'analysis' and (any((args.demo, args.source, args.target, args.categories)) or args.seed is not None):
        parser.error('data inputs and --demo apply only to analysis')
    if args.all_models and args.command != 'qualification':
        parser.error('--all-models applies only to qualification')
    start = time.perf_counter()
    try:
        import numpy as np
        from threadpoolctl import threadpool_limits
        from . import experiments
        from .data import load
        with threadpool_limits(limits=1):
            if args.command == 'construction':
                result = experiments.construction()
            elif args.command == 'records':
                result = experiments.historical_records()
            elif args.command == 'qualification':
                from .qualification import saved_bounds
                result = {'entropy': experiments.entropy_bounds(args.all_models), 'saved_bounds': saved_bounds()}
            else:
                from .readout_discovery import nested_readout_result
                if args.demo:
                    if any((args.source, args.target, args.categories)):
                        parser.error('--demo cannot be mixed with matrix inputs')
                    rng = np.random.default_rng(20261005)
                    x = rng.normal(size=(72,8))
                    y = x @ rng.normal(size=(8,10)) + .01*rng.normal(size=(72,10))
                    labels = np.repeat(np.arange(3),24)
                    config = dict(outer_split_seed=20261005, outer_folds=3, inner_retrieval_candidates=13,
                                  outer_retrieval_candidates=16, ridge_alphas=[.01,.1], pls_components=[2,4],
                                  permutations_per_mode=3, bootstrap_resamples=20, bootstrap_confidence=.95)
                else:
                    if not all((args.source, args.target, args.categories)):
                        parser.error('provide --source, --target and --categories, or explicitly select --demo')
                    x, y, labels = [np.load(p, allow_pickle=False) for p in (args.source,args.target,args.categories)]
                    config = load('analysis')['discovery_config']
                seed = args.seed if args.seed is not None else (20261005 if args.demo else config['master_seed'])
                result = {'scope': 'SYNTHETIC_INTERFACE_TEST' if args.demo else 'USER_ARRAYS_DISCOVERY_PROTOCOL_NOT_AUTOMATIC_HISTORICAL_PARITY',
                          'n': len(x), 'evaluation_seed': seed, 'readout': nested_readout_result(x,y,labels,config,seed,args.family)}
        result['elapsed_seconds'] = time.perf_counter()-start
        if args.json:
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == 'analysis':
            print(result['scope'])
            print(f"{args.family}: n={result['n']}, Top-1={result['readout']['summary']['top1']['mean']:.6f}")
            print('Nested selection; same-category candidates. Null/CI condition on fitted models and fixed outcomes.')
        elif args.command == 'construction':
            print('REGENERATED / 16 historical shared components, n=512 and n=768')
            print(f"Largest historical scalar discrepancy: {result['maximum_historical_scalar_error']:.3e}")
            print('n     replica   raw cross-moment   covariance-only   cross-degree    removed energy')
            for row in result['cohorts']:
                v=row['variants']
                print(f"{row['n']:<5} {row['replica']:<9} {v['raw']['cross_moment_max_abs']:<18.3e}{v['covariance_only']['cross_moment_max_abs']:<18.3e}{v['cross_degree']['cross_moment_max_abs']:<16.3e}{row['energy_removed']:.4%}")
            print('Shared-component coupling, covariance and energy diagnostics. Full-source measurements: records command.')
        elif args.command == 'records':
            print(f"REAGGREGATED / {result['matched_readout_exact_comparisons']} exact matched-readout comparisons; {result['cell_ablation_exact_paired_deltas']} exact cell-ablation deltas")
            print('n     carrier   repair    RSA SD ratio   RSA mean change')
            for r in result['cross_degree_between_cohort_RSA']:
                print(f"{r['n']:<5} {r['carrier']:<9.3f} {r['repair']:<9.4f} {r['RSA_SD_ratio']:<14.6f} {r['RSA_mean_change']:+.6f}")
            print('Summaries reconstructed from saved measurements; n=256 uses a distinct construction branch.')
        else:
            for r in result['entropy']['bounds']:
                print(f"{r['model']} / {r['slot']}: ideal expected rank >= {r['lower']}, log margin {r['log_margin']:.9f} [{r['role']}]")
            q = result['saved_bounds']
            print(f"SAVED BOUNDS / {q['construction_interval']['cover']['cells']} gap-free leaves; {q['ridge']['fold_cell_bounds']} ridge bounds")
            print(f"SAVED SVD / 3072 values; minimum target-to-bound ratio {min(r['target_to_bound_ratio'] for r in q['captured_svd']):.1f}")
            print('Ideal-model expectation bounds and saved interval/center-execution checks. Reproduction scope: docs/TECHNICAL_NOTE.md.')
        if not args.json:
            print(f"Elapsed: {result['elapsed_seconds']:.2f}s")
        return 0
    except (ValueError, ArithmeticError, AssertionError, OSError, KeyError) as error:
        print(f'Genesis refused the calculation: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
