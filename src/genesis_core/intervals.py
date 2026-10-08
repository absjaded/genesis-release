"""Check rational interval coverage and thresholds using saved enclosures."""
from fractions import Fraction as Q
import math
def require(condition, message):
    if not condition:
        raise ValueError(message)
def exact_number(value) -> Q:
    require(type(value) in (int, float) and math.isfinite(value), 'invalid numeric bound')
    return Q(value)

def check_cover(intervals: list[list[float]]) -> dict:
    require(bool(intervals), 'empty cover')
    pairs = []
    for interval in intervals:
        require(len(interval) == 2, 'interval needs two endpoints')
        lower, upper = map(exact_number, interval)
        require(lower < upper, 'empty or reversed interval')
        pairs.append((lower, upper))
    pairs.sort()
    require(pairs[0][0] <= Q(15, 1000) and pairs[-1][1] >= Q(20, 1000), 'cover misses an exact decimal domain endpoint')
    for previous, current in zip(pairs, pairs[1:]):
        require(previous[1] == current[0], 'gap, overlap or duplicate interval')
    width = sum((hi - lo for lo, hi in pairs), Q(0))
    require(width == pairs[-1][1] - pairs[0][0], 'coverage width mismatch')
    return {'cells': len(pairs), 'lower': float(pairs[0][0]), 'upper': float(pairs[-1][1]), 'zero_gaps_and_overlaps': True, 'contains_exact_decimal_closed_hull': True}

def inspect_leaves(leaves: list[dict]) -> dict:
    cover = check_cover([leaf['interval'] for leaf in leaves])
    minima = {key: [] for key in ('global', 'micro', 'mesoscopic')}
    maxima = {key: [] for key in minima}
    expected_local = {('micro', 80), ('micro', 40), ('micro', 27), ('mesoscopic', 257), ('mesoscopic', 184), ('mesoscopic', 149)}
    thresholds = {'global': Q(22, 100), 'micro': Q(12, 1000), 'mesoscopic': Q(36, 1000)}
    for leaf in leaves:
        require(leaf['resolved'] is True and leaf['micro_decision'] == 'TRUE' and (leaf['mesoscopic_decision'] == 'TRUE'), 'unresolved or nonpositive leaf')
        local = leaf['local']
        require(len(local) == 6 and {(r['role'], r['k']) for r in local} == expected_local, 'missing, duplicated or changed reference neighborhood mass')
        for role, row in [('global', leaf['global'])] + [(r['role'], r) for r in local]:
            lo, hi = (exact_number(row['effect_lower']), exact_number(row['effect_upper']))
            require(row['decision'] == 'TRUE' and thresholds[role] <= lo <= hi, f'{role} bound does not support its recorded decision')
            minima[role].append(lo)
            maxima[role].append(hi)
    return {'cover': cover, 'effects': {role: {'minimum_lower': float(min(minima[role])), 'maximum_upper': float(max(maxima[role])), 'required_lower': float(thresholds[role]), 'minimum_margin': float(min(minima[role]) - thresholds[role])} for role in minima}}

def inspect_ridge(cells: list[dict]) -> dict:
    cover = check_cover([cell['interval'] for cell in cells])
    require(sorted((c['cell_index'] for c in cells)) == list(range(16)), 'missing or duplicate ridge cell')
    upper = []
    for cell in cells:
        require(len(cell['folds']) == 6, 'ridge requires all six folds')
        values = [exact_number(fold['R2_upper']) for fold in cell['folds']]
        require(max(values) == exact_number(cell['maximum_fold_R2_upper']), 'cell maximum differs')
        upper.extend(values)
    maximum = max(upper)
    require(maximum <= Q(85, 100), 'ridge ceiling not supported')
    return {'cover': cover, 'fold_cell_bounds': len(upper), 'maximum_R2_upper': float(maximum), 'required_ceiling': 0.85, 'margin': float(Q(85, 100) - maximum)}
