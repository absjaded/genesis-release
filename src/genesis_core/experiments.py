"""Bounded computations and saved-record comparisons, with explicit scopes."""
from collections import defaultdict
from fractions import Fraction as F
import math
import numpy as np
from .data import load
from .charts import shared_chart
from .intervention import orthogonal_recolor, recolor, center, diagnostics, linear_cka


def construction():
    data = load('construction')
    base = data['base']
    args = {k: base['base_coordinate'][k] for k in ('beta_low', 'beta_high', 'axial_gain', 'spectral_temperature_M', 'dominant_mass_N')}
    args.update(parity=base['mechanism']['parity_trace_mass'], order=base['execution']['gauss_hermite_order'])
    cohorts = []
    for saved in data['cohorts']:
        structural = saved['structural_receipt']
        parts = shared_chart(saved['n'], structural['base_seed'], **args)
        z, shared, target = parts['z'], parts['shared'], parts['target']
        transformed, energy = orthogonal_recolor(z, shared, target)
        old = structural['transform_receipt']
        errors = [abs(energy['projection_energy_fraction'] - old['projection_energy_fraction']),
                  abs(linear_cka(parts['shared_n'], shared) - old['before_shared_N_full_shared_M_linear_CKA'])]
        variants = {key: diagnostics(z, x, target) for key,x in
                    [('raw', center(shared)), ('covariance_only', recolor(shared, target)), ('cross_degree', transformed)]}
        post = variants['cross_degree']
        if max(errors) > 1e-9 or post['cross_moment_max_abs'] > 5e-12 or post['covariance_max_abs_error'] > 5e-11:
            raise ArithmeticError('historical shared-component replay outside declared tolerances')
        cohorts.append({'n': saved['n'], 'replica': saved['replica_index'], 'variants': variants,
                        'energy_removed': energy['projection_energy_fraction'], 'historical_scalar_error': max(errors)})
    return {'scope': 'REGENERATED_SHARED_COMPONENTS_NOT_FULL_SOURCE_PAIRS', 'cohorts': cohorts,
            'maximum_historical_scalar_error': max(r['historical_scalar_error'] for r in cohorts)}


def historical_records():
    """Reaggregate originals; never substitute saved metrics for matrix replay."""
    empirical, data = load('analysis'), load('construction')
    comparisons = 0
    for row in empirical['mp002c']['rows']:
        folds = row['folds']
        if len(folds) != 8:
            raise ValueError('eight matched readout folds required')
        for fold in folds:
            train, test = set(fold['train']), set(fold['test'])
            if len(train) != 48 or len(test) != 16 or train & test or train | test != set(range(64)):
                raise ValueError('invalid historical partition')
        for metric, saved in row['summary'].items():
            a = np.array([f['metrics'][metric] for f in folds])
            b = np.array([f['category_centroid_baseline'][metric] for f in folds])
            measured = dict(mean=float(a.mean()), median=float(np.median(a)), p05=float(np.percentile(a, 5)),
                            category_centroid_mean=float(b.mean()), mean_delta_over_category_centroid=float((a-b).mean()))
            if measured != saved:
                raise ArithmeticError('historical fold summary differs')
            comparisons += len(measured)
    groups = defaultdict(dict)
    for row in data['f151']['rows']:
        key = row['n'], row['N_carrier_mass'], row['M_linear_repair_mass']
        groups[key].setdefault(row['variant'], {})[row['replica_index']] = row
    spread = []
    for (n, carrier, repair), variants in sorted(groups.items()):
        raw, changed = variants['RAW_IID_CONTROL'], variants['CROSS_DEGREE_ORTHOGONAL_RECOLOR']
        if set(raw) != set(changed) or set(raw) != set(range(8)):
            raise ValueError('missing paired cohort')
        for i in raw:
            if any(raw[i][k] != changed[i][k] for k in ('base_seed', 'walsh_permutation_seed')):
                raise ValueError('unmatched intervention seeds')
        x = np.array([raw[i]['partial_rsa'] for i in range(8)])
        y = np.array([changed[i]['partial_rsa'] for i in range(8)])
        spread.append({'n': n, 'carrier': carrier, 'repair': repair,
                       'RSA_SD_ratio': float(y.std(ddof=1)/x.std(ddof=1)), 'RSA_mean_change': float((y-x).mean())})
    reference = {(r['replica_index'], r['M_linear_repair_mass']): r for r in data['f92']['rows']}
    paired, ablation = 0, defaultdict(list)
    for r in data['f94']['rows']:
        old = reference[(r['replica_index'], r['M_linear_repair_mass'])]
        if any(r[k] != old[k] for k in ('base_seed', 'walsh_permutation_seed')):
            raise ValueError('unmatched ablation seeds')
        values = {'partial_rsa': r['metrics']['partial_rsa']-old['metrics']['partial_rsa']}
        for direction in ('N_to_M', 'M_to_N'):
            key = direction.lower() + '_r2'
            values['reconstruction_'+direction+'_R2'] = r['metrics']['reconstruction'][key]-old['metrics']['reconstruction'][key]
        for scale in ('micro', 'mesoscopic'):
            values[scale+'_exact_local_effect'] = r['local_scale_summary'][scale]['delta_vs_category_null']-old['local_scale_summary'][scale]['delta_vs_category_null']
        if values != r['paired_delta_cell_off_minus_F92']:
            raise ArithmeticError('paired ablation delta differs')
        paired += len(values)
        ablation[r['M_linear_repair_mass']].append(values)
    scale_effects = [{'repair': k, **{name: float(np.mean([r[name] for r in rows])) for name in rows[0]}}
                     for k, rows in sorted(ablation.items())]
    validation = []
    for label in ('all_size_validation', 'n256_distinct_branch'):
        by_size = defaultdict(list)
        for row in data[label]['rows']:
            by_size[row['n']].append(row)
        for n, rows in sorted(by_size.items()):
            by_cohort = defaultdict(list)
            for r in rows:
                by_cohort[r['replica_index']].append(r['all_evaluated_finite_gate_axes_pass'])
            validation.append({'branch': 'cross_degree_construction' if label == 'all_size_validation' else 'small_cohort_original_construction', 'n': n, 'gate_pass_rows': sum(r['all_evaluated_finite_gate_axes_pass'] for r in rows),
                               'rows': len(rows), 'all_repair_pass_cohorts': sum(all(v) for v in by_cohort.values()), 'cohorts': len(by_cohort)})
    discovery = next(r for r in empirical['mp002d']['primary_prefixes'] if r['n'] == 768)
    return {'scope': 'SAVED_RECORD_REAGGREGATION_NOT_HISTORICAL_MATRIX_REPLAY', 'matched_readout_exact_comparisons': comparisons,
            'discovery_768_saved_summary': discovery, 'matrix_parity_pending': True,
            'cell_ablation_exact_paired_deltas': paired, 'cell_ablation_scale_effects': scale_effects,
            'cross_degree_between_cohort_RSA': spread, 'size_specific_validation': validation}


def entropy_bounds(all_models=False):
    from . import entropy
    data = load('entropy_laws')
    controls = entropy.moment_identity_controls()
    rows = []
    for law in data['laws']:
        if law['law_id'] != 11 and not all_models:
            continue
        center = F(2, 11) + F(law['group_energy_increment'])
        for slot in ('N', 'M'):
            result = entropy.bound(data['parent'], law, slot, center=center)
            rows.append({'model': 'selected_marginal' if law['law_id'] == 11 else 'alternative_marginal_' + str([10,12,13,14,15,16].index(law['law_id']) + 1), 'source_law_id': law['law_id'], 'slot': slot, 'center': str(center),
                         'half_width': '1/4096', 'lower': result['zero_extended_ideal_reference_rank_expectation_lower'],
                         'log_margin': result['log_comparison_positive_margin']['display'],
                         'role': 'FROZEN_SELECTED_MODEL_BOUND' if law['law_id'] == 11 else 'OWN_CENTER_DIAGNOSTIC_EXTENSION'})
    return {'scope': 'IDEAL_IID_MARGINAL_EXPECTATION_NOT_LITERAL_PIPELINE_QUALIFICATION', 'controls': controls, 'bounds': rows}
