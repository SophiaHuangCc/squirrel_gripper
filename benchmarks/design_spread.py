"""Separate selected-winner spread, candidate performance tails, and design diversity."""
import argparse
from collections import defaultdict
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path

import numpy as np


def average(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else None


def spread(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.std(values)) if values else None


def geometry_stats(designs, lo, hi):
    """One vector per candidate, within one scene/seed; RMS distance in range units."""
    if not designs:
        return dict(designs=0, pairwise_rms=None, near_duplicate_pair_fraction=None,
                    near_bound_fraction=None, out_of_bounds_fraction=None)
    x = np.asarray(designs, dtype=float)
    active = hi-lo > 1e-12
    x = (x[:, active]-lo[active])/(hi-lo)[active]
    distances = [float(np.sqrt(np.mean((x[i]-x[j])**2))) for i in range(len(x)) for j in range(i)]
    return dict(designs=len(x), pairwise_rms=average(distances),
        near_duplicate_pair_fraction=average([d <= .001 for d in distances]),
        near_bound_fraction=float(np.mean(np.any((x <= .01) | (x >= .99), axis=1))),
        out_of_bounds_fraction=float(np.mean(np.any((x < -1e-6) | (x > 1+1e-6), axis=1))))


def build_spread(rows, lo, hi):
    expected, grouped = defaultdict(set), defaultdict(list)
    for row in rows:
        scene = row['scenario_id'].split('@')[0]
        expected[(row['objective'], scene)].add(row['scenario_id'])
        grouped[(row['objective'], row['method'], scene, row['seed'], row['candidate_id'])].append(row)
    candidates = []
    for (objective, method, scene, seed, cid), group in sorted(grouped.items()):
        ids = [r['scenario_id'] for r in group]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate candidate-condition records: {method}/{scene}/{cid}')
        valid = [r for r in group if r.get('status', 'ok') == 'ok'
                 and all(r.get(k) is not None and math.isfinite(float(r[k]))
                         for k in ('simulator_utility','num_contacts','angular_span_deg'))]
        complete = len(valid) == len(expected[(objective, scene)])
        designs = [r['design_params'] for r in group if r.get('design_params') is not None]
        if designs and not all(np.allclose(d, designs[0], rtol=0, atol=1e-7) for d in designs):
            raise ValueError(f'Design changes across conditions: {method}/{scene}/{cid}')
        candidates.append(dict(objective=objective, method=method, scene=scene, seed=seed, candidate_id=cid,
            observed_trials=len(group), successful_trials=len(valid), expected_conditions=len(expected[(objective,scene)]),
            complete=complete, design_params=designs[0] if designs else None,
            mean_utility=average(r['simulator_utility'] for r in valid) if complete else None,
            mean_contacts=average(r['num_contacts'] for r in valid) if complete else None,
            mean_angular_span_deg=average(r['angular_span_deg'] for r in valid) if complete else None,
            within_condition_std_utility=spread(r['simulator_utility'] for r in valid) if complete else None,
            worst_trial_utility=min((r['simulator_utility'] for r in valid), default=None),
            source=group[0].get('result_path', '')))
    methods, scene_groups = defaultdict(list), defaultdict(list)
    for c in candidates:
        methods[(c['objective'],c['method'])].append(c)
        scene_groups[(c['objective'],c['method'],c['scene'],c['seed'])].append(c)
    diversity = []
    for (objective,method,scene,seed), group in sorted(scene_groups.items()):
        designs = [c['design_params'] for c in group if c['design_params'] is not None]
        diversity.append(dict(objective=objective,method=method,scene=scene,seed=seed,
            missing_designs=len(group)-len(designs), **geometry_stats(designs,lo,hi)))
    winners, summary = [], []
    for (objective,method), group in sorted(methods.items()):
        complete = [c for c in group if c['complete']]
        by_scene = defaultdict(list)
        for c in complete:
            by_scene[c['scene']].append(c)
        selected = [max(cs,key=lambda c:(c['mean_utility'],c['seed'],c['candidate_id'])) for cs in by_scene.values()]
        winners.extend(selected)
        utilities = [c['mean_utility'] for c in complete]
        div = [d for d in diversity if d['objective']==objective and d['method']==method]
        record = dict(objective=objective,method=method,observed_designs=len(group),
            complete_designs=len(complete),selected_scenes=len(selected), observed_scenes=len({c['scene'] for c in group}),
            seeds=len({c['seed'] for c in group}),observed_trials=sum(c['observed_trials'] for c in group),
            unsuccessful_or_invalid_trials=sum(c['observed_trials']-c['successful_trials'] for c in group),
            incomplete_designs=len(group)-len(complete),
            pool_mean_utility=average(utilities),pool_std_utility=spread(utilities),
            pool_p10_utility=float(np.quantile(utilities,.1)) if utilities else None,
            pool_min_utility=min(utilities,default=None),
            worst_trial_utility=min((c['worst_trial_utility'] for c in group if c['worst_trial_utility'] is not None),default=None),
            mean_within_scene_seed_std_utility=average(spread(c['mean_utility'] for c in cs if c['complete'])
                for key,cs in scene_groups.items() if key[:2]==(objective,method)),
            mean_within_condition_std_utility=average(c['within_condition_std_utility'] for c in complete),
            design_diversity_rms=average(d['pairwise_rms'] for d in div),
            near_duplicate_pair_fraction=average(d['near_duplicate_pair_fraction'] for d in div),
            near_bound_design_fraction=average(d['near_bound_fraction'] for d in div),
            out_of_bounds_design_fraction=average(d['out_of_bounds_fraction'] for d in div))
        for field in ('utility','contacts','angular_span_deg'):
            for prefix,items in (('selected',selected),('pool',complete)):
                record[f'{prefix}_mean_{field}']=average(c[f'mean_{field}'] for c in items)
                record[f'{prefix}_std_{field}']=spread(c[f'mean_{field}'] for c in items)
        summary.append(record)
    return dict(summary=summary,winners=winners,candidates=candidates,diversity_by_scene_seed=diversity)


def write_spread_report(rows, output):
    from generator.dataloader import DesignBounds, DESIGN_NAMES
    bounds = DesignBounds.defaults()
    lo,hi = bounds.lo.numpy().astype(float),bounds.hi.numpy().astype(float)
    report = build_spread(rows,lo,hi)
    report['definitions'] = {
        'selected_table':'One winner per method/base scene, maximizing mean utility over initial conditions; select across seeds as in the supplied table. SD is population SD across scene-winner means, not a confidence interval.',
        'pool_table':'One observation per candidate/base scene, averaged over conditions; incomplete designs excluded from performance averages and counted separately. Pooled SD includes scene differences.',
        'diversity':'Mean pairwise RMS parameter distance after dividing variable coordinates by default design ranges. Calculated separately per scene/seed, then equally averaged across groups. Repeated conditions do not duplicate designs. Lower indicates less parameter diversity; it is not pose diversity.',
        'near_duplicates':'Fraction of within-scene/seed candidate pairs with normalized RMS distance <=0.001.',
        'near_bounds':'Fraction of designs with any variable coordinate within 1% of a bound (or outside it). Boundary designs are not necessarily invalid or physically extreme.',
        'missing_data':'Expected conditions are inferred from observed records per objective/base scene. Never-run conditions or designs absent from all records cannot be counted; use completed, equally budgeted studies. Small seed counts and selected winners cannot establish significance.',
        'selection_bias':'These statistics describe the recorded candidate pool. For old V20 runs this pool was already surrogate-selected, not the full raw diffusion proposal pool.'}
    report['normalization_bounds'] = {name:[float(a),float(b)] for name,a,b in zip(DESIGN_NAMES,lo,hi)}
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    (output/'spread_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for key,filename in [('summary','method_spread.csv'),('winners','selected_scene_winners.csv'),
                         ('candidates','candidate_spread.csv'),('diversity_by_scene_seed','design_diversity_by_scene.csv')]:
        data = [{k:v for k,v in r.items() if k!='design_params'} for r in report[key]]
        if data:
            with (output/filename).open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    def number(v,digits=4):
        if v is None:return '—'
        rounded=Decimal(str(round(float(v),12))).quantize(Decimal(1).scaleb(-digits),rounding=ROUND_HALF_UP)
        return f'{rounded:.{digits}f}'
    def label(r):
        name={'conditional_diffusion':'Conditional diffusion','adam':'Adam optimized','cma_es':'CMA-ES'}.get(r['method'],r['method'])
        if name.startswith('pose_dgdm_gs'):name='DGDM gs='+name.removeprefix('pose_dgdm_gs').replace('p','.')
        return name if r['objective'] in ('unknown','') else f"{name} ({r['objective']})"
    def pm(r,prefix,field,digits):return number(r[f'{prefix}_mean_{field}'],digits)+' ± '+number(r[f'{prefix}_std_{field}'],digits)
    lines=['# Specialist scene-winner averages with spread','',report['definitions']['selected_table'],'',
        '| Method | Scenes | Utility mean ± SD | Contacts mean ± SD | Angular span mean ± SD (°) |',
        '|---|---:|---:|---:|---:|']
    for r in sorted(report['summary'],key=lambda r:-(r['selected_mean_utility'] or 0)):
        lines.append(f"| {label(r)} | {r['selected_scenes']} | {pm(r,'selected','utility',4)} | {pm(r,'selected','contacts',1)} | {pm(r,'selected','angular_span_deg',1)} |")
    lines+=['','## All recorded candidates: poor cases and diversity','',report['definitions']['pool_table'],'',
        '| Method | Complete designs | Utility mean ± SD | Utility P10 / minimum | Design diversity RMS | Near-bound designs | Failed/invalid trials |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in report['summary']:
        boundary=None if r['near_bound_design_fraction'] is None else 100*r['near_bound_design_fraction']
        lines.append(f"| {label(r)} | {r['complete_designs']}/{r['observed_designs']} | {pm(r,'pool','utility',4)} | {number(r['pool_p10_utility'])} / {number(r['pool_min_utility'])} | {number(r['design_diversity_rms'])} | {number(boundary,1)}% | {r['unsuccessful_or_invalid_trials']} |")
    for definition in list(report['definitions'].values())[2:]:
        lines += ['',definition]
    (output/'specialist_method_averages.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    from benchmarks.analyze_study import read_rows
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    rows=read_rows(args.study_dir.resolve(),include_failures=True)
    if not rows:raise ValueError('No recorded trials found')
    write_spread_report(rows,args.output_dir)
    print(args.output_dir/'specialist_method_averages.md')


if __name__=='__main__':main()
