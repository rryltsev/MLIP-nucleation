"""Reproduce the historical freeze audit without new model evaluations."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
TIAL = ROOT.parent / 'TiAl'
RUN = ROOT / 'runs/inference-freeze-audit-20261111-01'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    evidence = RUN / 'historical-evidence'
    evidence.mkdir(exist_ok=True)
    paths = {
        'mlip-freeze-process': ROOT / 'runs/fm-deployment-20261017-03/dpa4/export-process.json',
        'mlip-lammps-result': ROOT / 'runs/fm-deployment-20261017-07/dpa4/result.json',
        'mlip-export-launcher': ROOT / 'scripts/hpc/fm-deployment.sbatch',
        'mlip-baseline-launcher': ROOT / 'scripts/hpc/fm-al32-baseline.sbatch',
        'mlip-baseline-worker': ROOT / 'scripts/hpc/fm-al32-baseline.py',
        'mlip-dpa4-timing': ROOT / 'runs/fm-al32-baseline-20261019-02/dpa4/predictions.json',
        'mlip-grace-timing': ROOT / 'runs/fm-al32-baseline-20261019-02/grace/predictions.json',
        'tial-pilot-worker': TIAL / 'research/tial_fm/scripts/evaluate_pilot.py',
        'tial-baseline-worker': TIAL / 'research/tial_fm/scripts/baseline_worker.py',
        'tial-baseline-export': TIAL / 'research/tial_fm/runs/dpa4-n200-pilot-20261106-02/collected/baseline/export_checks.json',
        'tial-md-launcher': TIAL / 'audit/oct26/pbe15k_melting_1775/collected/job.sbatch',
        'tial-md-input': TIAL / 'audit/oct26/pbe15k_melting_1775/collected/T1775K/melt/setup.in',
        'tial-md-preflight': TIAL / 'audit/oct26/pbe15k_melting_1775/collected/preflight.json',
        'tial-grace-gpu-summary': TIAL / 'research/tial_fm/runs/grace-precision-20261107-02/collected/comparison/gpu/summary.json',
    }
    sources = {}
    for key, path in paths.items():
        data = path.read_bytes()
        target = evidence / (key + path.suffix)
        if target.exists():
            assert target.read_bytes() == data, 'Historical snapshot changed: ' + str(path)
        else:
            target.write_bytes(data)
        sources[key] = dict(original=str(path), snapshot=str(target.relative_to(ROOT)).replace('\\', '/'),
                            sha256=hashlib.sha256(data).hexdigest())
    historical = {}
    for backend in ['dpa4', 'grace']:
        rows = load(paths['mlip-' + backend + '-timing'])['frames']
        subset = [r for r in rows[1:] if r['split'] == 'train']
        historical[backend] = dict(natoms=32, sample_count=len(subset),
             configuration_ids=[r['configuration_id'] for r in subset],
             median_ms=1000*statistics.median(r['inference_seconds'] for r in subset),
             first_call_ms=1000*rows[0]['inference_seconds'],
             raw_seconds=[r['inference_seconds'] for r in subset])
    assert historical['dpa4']['configuration_ids'] == historical['grace']['configuration_ids']
    historical.update(job_id='20952054', device='CPU apollo30, four allocated CPUs',
        protocol='Post-hoc timing of 20 train frames after excluding first call; no repeats, no dedicated warmup; heterogeneous API paths',
        variants='DPA4-Mini original .pt, compile/triton disabled; GRACE-1L fp64, main JIT disabled',
        limitation='Diagnostic historical comparison only; not an optimized throughput benchmark or A100 prediction')
    process = load(paths['mlip-freeze-process'])
    lammps = load(paths['mlip-lammps-result'])
    output = dict(date='2026-11-11', sources=sources,
        conclusion='Freeze was performed in both projects and .pt2 was actually used in native LAMMPS. Several baseline, fine-tuned metric and phonon paths continued to use .pt with inference compile and Triton disabled. Prior .pt2 exports used Triton level0; level0 still contains AOTInductor compilation.',
        mlip=dict(freeze_job='20950326', freeze_exitcode=process['exitcode'],
                  freeze_seconds=process['seconds'], command=process['command'],
                  lammps_job=lammps['job_id'], lammps_status=lammps['status'],
                  exported_sha256=lammps['consumed_source']['files']['exported.pt2']),
        tial=dict(cpu_baseline_export=load(paths['tial-baseline-export']),
                  md_preflight=load(paths['tial-md-preflight']),
                  implementation_finding='evaluate_pilot.py creates and checks export, executes del export, then train/validation and harmonic evaluation use dp=DeepPot(model), the .pt checkpoint'),
        historical_CPU=historical,
        historical_GRACE_A100=dict(job_id='20989431',model='GRACE-2L-OMAT-medium-base',
                  timings=load(paths['tial-grace-gpu-summary'])['timing'],
                  protocol='ASE TPCalculator diverse, main XLA on, TF32/UQ off; 5 warmup +30 calls per size, two fresh processes per precision, synchronized numpy results'),
        current_benchmark=dict(job_id='20992566', status='submitted; no performance results yet',
                  manifest='runs/inference-freeze-audit-20261111-01/manifest.json'),
        documentation=[
            'https://docs.deepmodeling.com/projects/deepmd/en/v3.2.0/model/dpa4.html',
            'https://huggingface.co/deepmodelingcommunity/DPA4-OMat24'],
        documentation_note='Freeze variables are captured at export; level2 fuses Triton kernels with full float32 accumulation. Export is target-specific. Performance must be measured and numerical equivalence rechecked.')
    (RUN / 'historical-audit.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(CPU_median_ms={b:historical[b]['median_ms'] for b in ['dpa4','grace']},
                         sources=len(sources), new_job=output['current_benchmark']), indent=2))


if __name__ == '__main__':
    main()
