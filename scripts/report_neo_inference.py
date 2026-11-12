"""Generate a focused Neo/Triton2 LaTeX section from verified raw-derived results."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/inference-freeze-audit-2026-11-11'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def generate(run):
    result = load(run / 'assessment.json')
    manifest = load(run / 'manifest.json')
    plan = load(run / 'collected/plan.json')
    assert result['status'] == 'complete'
    assert all(c['passed'] for checks in result['export_checks'].values() for c in checks)
    neo2 = 'dpa4-neo-pt2-triton2'
    neo0 = 'dpa4-neo-pt2-triton0'
    assert result['export_attempts'][neo2]['returncode'] == 0
    assert result['export_attempts'][neo2]['freeze_environment']['DP_TRITON_INFER'] == '2'
    labels = {
        'dpa4-neo-pt': r'DPA4-Neo, \texttt{.pt}',
        neo0: r'DPA4-Neo, \texttt{.pt2}, Triton0',
        neo2: r'DPA4-Neo, \texttt{.pt2}, Triton2',
        'grace-2l-fp32': 'GRACE-2L, fp32 / XLA',
        'grace-2l-fp64': 'GRACE-2L, fp64 / XLA',
    }
    sizes = plan['structures']
    times = result['timings']
    seconds = result['export_attempts'][neo2]['seconds']
    scheduler = manifest['scheduler_parent'].split('|')
    speedups = {key: {s: times[key][s]['median_ms']/times[neo2][s]['median_ms'] for s in sizes}
                for key in labels if key != neo2}
    breakeven = {s: seconds / ((times['dpa4-neo-pt'][s]['median_ms'] -
                              times[neo2][s]['median_ms']) / 1000) for s in sizes}
    prior = run.parent / plan['predecessor']['run_id']
    old = load(prior / 'assessment.json')
    drift = {}
    for key in labels:
        if key not in old['timings']:
            continue
        old_block = load(prior / 'collected/results' / (key+'-block0') / 'summary.json')
        new_block = load(run / 'collected/results' / (key+'-block0') / 'summary.json')
        assert old_block['model_sha256'] == new_block['model_sha256']
        drift[key] = {s: times[key][s]['median_ms']/old['timings'][key][s]['median_ms'] for s in sizes}
    derived = dict(triton2_speedup_relative_to=speedups, freeze_seconds=seconds,
                   freeze_only_breakeven_calls_versus_pt=breakeven,
                   breakeven_scope='Estimate using measured steady-state latencies; excludes model-load and first-call differences',
                   common_case_latency_ratio_retry_over_prior=drift,
                   prior_assessment_sha256=hashlib.sha256((prior/'assessment.json').read_bytes()).hexdigest(),
                   current_assessment_sha256=hashlib.sha256((run/'assessment.json').read_bytes()).hexdigest())
    (run / 'comparison-summary.json').write_text(json.dumps(derived, indent=2) + '\n', newline='\n')
    lines = [r'\section*{DPA4-Neo Triton2 completion}',
        r'The failed 420-second export in job 20992566 remains part of the evidence.',
        'A separate run, ' + r'\texttt{' + manifest['run_id'] + '}, job ' + manifest['job_id'] + ',',
        'increased the export limit to ' + str(plan['freeze_timeout_seconds']) + ' seconds.',
        f'The new Triton2 export completed successfully in {seconds:.2f} seconds ({seconds/60:.2f} minutes).',
        r'The verified Triton0 artifact from job 20992566 was reused after checking its hash,',
        r'GPU type and software versions. All five timing cases were measured again on the',
        r'new allocation, with the same input bytes, perturbations and timing protocol.',
        r'Each table cell is the median of 60 synchronized full ASE E/F/stress calls;',
        r'export, loading, first calls and warmup are excluded. Units are milliseconds.',
        r'\begin{center}\small', r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
        r'Model and runtime & Ti96 & TiAl216 & TiAl432 & TiAl864 \\', r'\midrule']
    for key, label in labels.items():
        lines.append(label + ' & ' + ' & '.join(f"{times[key][s]['median_ms']:.2f}" for s in sizes) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}\end{center}',
        r'\begin{center}\small', r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
        r'Triton2 speedup relative to & Ti96 & TiAl216 & TiAl432 & TiAl864 \\', r'\midrule']
    for key, label in labels.items():
        if key != neo2:
            ratios = [times[key][s]['median_ms']/times[neo2][s]['median_ms'] for s in sizes]
            lines.append(label + ' & ' + ' & '.join(f'{x:.2f}' for x in ratios) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}\end{center}',
        r'Ratios greater than one mean Neo/Triton2 is faster. GRACE precisions use',
        r'the same GRACE-2L checkpoint weights (fp32 is the controlled cast). The',
        r'comparison does not establish equal DFT accuracy between architectures.',
        r'For TiAl216/432/864, the Triton2 speedups over GRACE-2L fp32 are ' +
        ', '.join(f"{speedups['grace-2l-fp32'][s]:.3f}" for s in sizes[1:]) + '.',
        r'The corresponding speedups over the existing Triton0 export are ' +
        ', '.join(f"{speedups[neo0][s]:.3f}" for s in sizes[1:]) + '.',
        r'Using the measured steady-state saving relative to eager inference, the one-time',
        f'freeze cost is recovered after approximately {min(breakeven.values()):.0f}--{max(breakeven.values()):.0f} calls.',
        r'This estimate is freeze time divided by the per-call time saving and excludes',
        r'differences in model loading and first-call overhead.',
        r'\subsection*{Serialization checks and variability}',
        r'Both exports pass the predefined E/F/stress tolerances against the eager',
        r'checkpoint on the first perturbed frame of all four structures in both processes.',
        r'\begin{center}\small', r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
        r'Export & $\Delta E$/atom (eV) & $\Delta F$ (eV/\AA) & $\Delta\sigma$ (eV/\AA$^3$) \\', r'\midrule']
    for key in [neo0, neo2]:
        checks = result['export_checks'][key]
        maxima = [max(c['differences'][m] for c in checks) for m in result['tolerances']]
        lines.append(labels[key] + ' & ' + ' & '.join(f'{x:.3e}' for x in maxima) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}\end{center}',
        r'The following entries show the two separate process medians (ms). They',
        r'expose process variation; two processes are insufficient for a general uncertainty estimate.',
        r'\begin{center}\footnotesize', r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
        r'Model and runtime & Ti96 & TiAl216 & TiAl432 & TiAl864 \\', r'\midrule']
    for key, label in labels.items():
        values = [' / '.join(f'{x:.2f}' for x in times[key][s]['block_medians_ms']) for s in sizes]
        lines.append(label + ' & ' + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}\end{center}',
        r'This remains a single-A100 ASE benchmark including neighbours and synchronization.',
        r'Ti96 Triton2 process medians differ visibly; use the larger Ti--Al cells for',
        r'the more consistent speed comparison. Common eager Neo controls are slower',
        r'in this retry than in the initial job: the new/old median ratios are ' +
        ', '.join(f"{drift['dpa4-neo-pt'][s]:.3f}" for s in sizes) + '.',
        r'The cause of this between-allocation variation was not isolated. All retry',
        r'speedups above use controls measured in the same retry, without pooling jobs.',
        r'Ti96 and Ti--Al cells differ in composition. Native LAMMPS equivalence,',
        r'long MD stability and equal-accuracy throughput are separate validations.',
        r'\subsection*{Retry provenance and cost}',
        'Scheduler state: ' + scheduler[1] + '; exit code ' + scheduler[2] + '.',
        f'Actual allocation: {int(scheduler[3])} GPU-seconds ({int(scheduler[3])/3600:.4f} GPU-hours)',
        f'and {int(scheduler[4])} CPU-seconds ({int(scheduler[4])/3600:.4f} CPU-hours).',
        r'Executed source commit: \path{' + manifest['code_commit'] + '}.',
        r'Input manifest hash: \path{' + manifest['input_hash'] + '}.',
        r'Accepted Triton2 model SHA-256: \path{' + result['export_attempts'][neo2]['sha256'] + '}.',
        r'Raw results, input/model hashes, scheduler records, backup locations and numerical',
        r'checks are under \path{' + run.relative_to(ROOT).as_posix() + '/}.',
        r'Tables were generated directly from \texttt{assessment.json}; source and output',
        r'hashes are recorded in \texttt{report-provenance.json}.',
    ]
    output = REPORT / 'neo-results.tex'
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    provenance = {key: dict(path=path.relative_to(ROOT).as_posix(),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                  for key, path in {'assessment': run / 'assessment.json',
                                    'comparison_summary': run / 'comparison-summary.json', 'generator': Path(__file__),
                                    'output': output}.items()}
    (run / 'report-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', newline='\n')
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    generate(parser.parse_args().run.resolve())
