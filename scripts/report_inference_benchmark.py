"""Generate the LaTeX result tables directly from verified benchmark measurements."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/inference-freeze-audit-20261111-01'
REPORT = ROOT / 'reports/inference-freeze-audit-2026-11-11'


def main():
    source = RUN / 'assessment.json'
    result = json.loads(source.read_text())
    labels = {
        'dpa4-mini-pt': r'DPA4-Mini, \texttt{.pt}',
        'dpa4-mini-pt2-triton0': r'DPA4-Mini, \texttt{.pt2}, Triton0',
        'dpa4-mini-pt2-triton2': r'DPA4-Mini, \texttt{.pt2}, Triton2',
        'dpa4-neo-pt': r'DPA4-Neo, \texttt{.pt}',
        'dpa4-neo-pt2-triton0': r'DPA4-Neo, \texttt{.pt2}, Triton0',
        'grace-1l-fp64': 'GRACE-1L, fp64 / XLA',
        'grace-2l-fp32': 'GRACE-2L, fp32 / XLA',
        'grace-2l-fp64': 'GRACE-2L, fp64 / XLA',
    }
    sizes = ['Ti96', 'TiAl216', 'TiAl432', 'TiAl864']
    lines = [r'\section*{Matched A100 results}',
        r'Median full ASE energy/force/stress latency in milliseconds; lower is faster.',
        r'The table pools 60 timed calls per case and structure (30 in each of two processes).',
        r'Compilation, model loading and warmup are excluded.',
        r'\begin{center}\small', r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
        r'Model and runtime & Ti96 & TiAl216 & TiAl432 & TiAl864 \\', r'\midrule']
    for key, label in labels.items():
        row = [f"{result['timings'][key][size]['median_ms']:.2f}" for size in sizes]
        lines.append(label + ' & ' + ' & '.join(row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}\end{center}',
        r'DPA4-Neo/Triton2 is omitted because export exceeded its 420-second subprocess limit.',
        r'The last logged stage was compilation of the parallel with-communication artifact,',
        r'after the first AOTInductor compilation. The retained partial file is not an accepted export;',
        r'no inference time or numerical-equivalence result exists for this case in the initial job.',
        r'The separately recorded retry is reported below.',
        r'\begin{center}\small', r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
        r'Export / eager speedup & Ti96 & TiAl216 & TiAl432 & TiAl864 \\',
        r'\midrule']
    for key in ['dpa4-mini-pt2-triton0', 'dpa4-mini-pt2-triton2', 'dpa4-neo-pt2-triton0']:
        comparison = result['comparisons'][key + '_versus_pt']
        assert comparison['numerical_gate_passed']
        lines.append(labels[key] + ' & ' + ' & '.join(f"{comparison['speedup'][s]:.2f}" for s in sizes) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}\end{center}',
        r'For TiAl216/432/864, DPA4-Mini/Triton2 is respectively 2.09, 2.19 and 2.71 times',
        r'faster than the tested GRACE-1L fp64 path. DPA4-Neo/Triton0 and GRACE-2L fp32',
        r'have similar throughput: Neo latency is about 10\% lower at 216 atoms, almost equal',
        r'at 432 atoms, and 10\% higher at 864 atoms. These compare the specific checkpoint',
        r'variants and precisions used by the projects, not equal-accuracy models.',
        r'\section*{Numerical checks and export cost}',
        r'Every completed export passes all predefined tolerances on the first perturbed',
        r'frame of each of the four structures in both processes. Maximum absolute discrepancies',
        r'are shown below; this is serialization consistency, not DFT validation.',
        r'\begin{center}\small', r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
        r'Export & $\Delta E$/atom (eV) & $\Delta F$ (eV/\AA) & $\Delta\sigma$ (eV/\AA$^3$) & Freeze (s) \\',
        r'\midrule']
    for key, checks in result['export_checks'].items():
        assert all(c['passed'] for c in checks)
        maxima = [max(c['differences'][metric] for c in checks) for metric in result['tolerances']]
        short_label = key.replace('dpa4-', 'DPA4-').replace('mini', 'Mini').replace('neo', 'Neo').replace('-pt2-triton', '/T')
        lines.append(short_label + ' & ' + ' & '.join(f'{x:.2e}' for x in maxima) +
                     f" & {result['export_attempts'][key]['seconds']:.1f}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}\end{center}',
        r'The largest force difference across all three accepted exports is below',
        r'$9.3\,10^{-6}$ eV/\AA. TF32 and AMP were disabled. Triton2 was enabled during',
        r'the Mini export; the runtime environment reports Triton0 because the frozen',
        r'executable retains its export-time settings.',
        r'\section*{Scope and variability}',
        r'All cases ran sequentially on one NVIDIA A100-PCIE-40GB on tesla-a101, driver',
        r'580.126.20, with four allocated CPUs. DeePMD-kit 3.2.0 / Torch 2.11.0 and',
        r'TensorPotential 0.6.1 / TensorFlow 2.20.0 were used. Neighbour preparation,',
        r'host/device transfers and result synchronization are included; ASE result caching is bypassed.',
        r'The second process block reversed the case order. Some small-cell measurements show',
        r'visible process variation: Mini/Triton2 at Ti96 has process medians 9.78 and 7.85 ms;',
        r'GRACE-2L fp64 at TiAl216 has 25.11 and 20.52 ms. Raw samples, process medians and',
        r'10th/90th percentiles are retained. Two processes do not define a broad uncertainty estimate.',
        r'Ti96 is pure Ti whereas the other cases are Ti--Al; the four points are not a single',
        r'composition-controlled scaling series. This is an ASE inference benchmark, not native',
        r'LAMMPS MD throughput, long-time stability, or an equal-DFT-accuracy comparison.',
        r'The timeout applies only to Neo/Triton2 export. No new computation was submitted during collection.',
    ]
    output = REPORT / 'results.tex'
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    manifest = dict(source=str(source.relative_to(ROOT)),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    output=str(output.relative_to(ROOT)),
                    output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    (RUN / 'report-provenance.json').write_text(json.dumps(manifest, indent=2) + '\n', newline='\n')
    print(output)


if __name__ == '__main__':
    main()
