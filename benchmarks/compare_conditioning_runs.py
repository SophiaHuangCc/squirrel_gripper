"""Compare conditional/context-only priors with guidance off/on on matched trials."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np


def collect(root, expected_mode):
    config = json.loads((root/'benchmark_effective_config.json').read_text())
    rows, values, manifests = [], {}, {}
    for method in ('conditional_diffusion', 'dgdm'):
        manifest = json.loads((root/'runs'/f'{method}_s0'/'manifest.json').read_text())
        meta = manifest['proposal_metadata']
        if meta['conditioning_mode'] != expected_mode or meta['selection_rule'] != 'all_generated_no_surrogate_ranking':
            raise ValueError(f'Wrong conditioning/selection protocol under {root}')
        groups, designs = {}, {}
        for job in manifest['jobs']:
            path = Path(job['run_dir'])/'benchmark_result.json'
            if not path.exists():
                raise ValueError(f'Incomplete run: {path}')
            result = json.loads(path.read_text())
            if result['status'] != 'ok':
                raise ValueError(f'Simulation failed; resume before comparing: {path}')
            groups.setdefault(job['candidate_id'], []).append(result['utility'])
            designs[job['candidate_id']] = job['design_params']
        if len(groups) != 16 or any(len(group) != 5 for group in groups.values()):
            raise ValueError('Expected 16 designs x five conditions per method')
        scores = [statistics.mean(groups[f'{method}_s0_{i:03d}']) for i in range(16)]
        values[method] = scores
        manifests[method] = manifest
        from benchmarks.design_spread import geometry_stats
        from generator.dataloader import DesignBounds
        bounds=DesignBounds.defaults()
        diversity=geometry_stats(list(designs.values()),bounds.lo.numpy(),bounds.hi.numpy())
        rows.append(dict(prior=expected_mode, guidance_scale=0 if method=='conditional_diffusion' else 1,
            designs=16, trials=80, mean_v20_utility=statistics.mean(scores),
            std_design_mean_utility=statistics.pstdev(scores), best_design_mean_utility=max(scores),
            min_design_mean_utility=min(scores),p10_design_mean_utility=float(np.quantile(scores,.1)),
            design_diversity_rms=diversity['pairwise_rms']))
    return config, manifests, rows, values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--conditional', type=Path, default=Path('outputs/direct_scene00_scale1'))
    p.add_argument('--context', type=Path, default=Path('outputs/context_only_scene00_scale1'))
    args = p.parse_args()
    old_config, old_m, old_rows, old = collect(args.conditional, 'conditional')
    new_config, new_m, new_rows, new = collect(args.context, 'context_only')
    if old_config != new_config:
        raise ValueError('Evaluation configurations differ; cannot call this a controlled comparison')
    for method in old_m:
        a,b = old_m[method],new_m[method]
        for key in ('seed','num_candidates','num_scenarios','num_rollouts'):
            if a[key] != b[key]:
                raise ValueError(f'Mismatched {key}')
        for key in ('batch_size','num_inference_steps','guidance_scale','guidance_timesteps','target_scenario_ids','dgdm_dynamics_checkpoint'):
            if a['proposal_metadata'].get(key) != b['proposal_metadata'].get(key):
                raise ValueError(f'Mismatched {key}')
        if [j['scenario'] for j in a['jobs']] != [j['scenario'] for j in b['jobs']]:
            raise ValueError('Different simulation conditions/order')
    gains = {}
    for mode, values in (('conditional',old),('context_only',new)):
        delta = [b-a for a,b in zip(values['conditional_diffusion'],values['dgdm'])]
        gains[mode] = dict(mean_guidance_gain=statistics.mean(delta),
            median_guidance_gain=statistics.median(delta), dgdm_wins=sum(d>1e-8 for d in delta),
            dgdm_losses=sum(d < -1e-8 for d in delta), ties=sum(abs(d)<=1e-8 for d in delta))
    rows = old_rows+new_rows
    report = dict(rows=rows, guidance_effect=gains,
        difference_in_guidance_gain=gains['context_only']['mean_guidance_gain']-gains['conditional']['mean_guidance_gain'],
        limitation='One scenario and one seed batch. A weaker context-only baseline alone does not prove better guidance. Compare the within-prior gains and absolute utilities; additional training/evaluation seeds are needed for general conclusions.')
    (args.context/'conditioning_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.context/'conditioning_comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines=['# Performance-conditioning comparison','',
        'SD is across 16 design means (each averages five conditions), not a confidence interval. Diversity is within-scene/seed normalized parameter RMS distance; lower means less parameter diversity.', '',
        '| Prior | Guidance scale | V20 utility mean ± SD | P10 / minimum | Design diversity RMS |',
        '|---|---:|---:|---:|---:|']
    lines += [f"| {r['prior']} | {r['guidance_scale']} | {r['mean_v20_utility']:.6f} ± {r['std_design_mean_utility']:.6f} | {r['p10_design_mean_utility']:.6f} / {r['min_design_mean_utility']:.6f} | {r['design_diversity_rms']:.6f} |" for r in rows]
    lines += ['',f"Guidance effects: `{json.dumps(gains)}`",'',report['limitation']]
    (args.context/'conditioning_comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
