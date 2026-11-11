"""Aggregate raw timing blocks and gate .pt2 speed claims on export equivalence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def flatten(value):
    if isinstance(value, list):
        return [x for v in value for x in flatten(v)]
    return [float(value)]


def maxdiff(a, b):
    left, right = flatten(a), flatten(b)
    assert len(left) == len(right)
    return max(abs(x-y) for x, y in zip(left, right))


def assess(folder):
    collected = folder / 'collected'
    plan = load(collected / 'plan.json')
    hashes = load(folder / 'collected-sha256.json')
    for name, digest in hashes.items():
        assert hashlib.sha256((collected / name).read_bytes()).hexdigest() == digest, name
    timings, failures, checks = {}, {}, {}
    for case in plan['cases']:
        cid = case['id']
        paths = [collected / 'results' / f'{cid}-block{i}' / 'summary.json' for i in [0, 1]]
        if not all(p.exists() for p in paths):
            failures[cid] = 'Missing at least one complete process; inspect export/process/failure logs'
            continue
        blocks = [load(p) for p in paths]
        assert all(x['status'] == 'PASS' for x in blocks)
        assert len({(x['job_id'], x['node'], x['device']) for x in blocks}) == 1
        timings[cid] = {}
        for name in plan['structures']:
            rows = [next(t for t in x['timings'] if t['name'] == name) for x in blocks]
            assert all(len(t['wall_seconds']) == plan['measured_calls'] for t in rows)
            samples = sum([t['wall_seconds'] for t in rows], [])
            med = statistics.median(samples)
            timings[cid][name] = dict(median_ms=med*1000, calls_per_second=1/med,
                                      atom_calls_per_second=rows[0]['natoms']/med,
                                      block_medians_ms=[statistics.median(t['wall_seconds'])*1000 for t in rows],
                                      cold_seconds=[t['cold_seconds'] for t in rows],
                                      calls=len(samples))
        if case.get('format') == 'pt2':
            baseline = cid.split('-pt2')[0] + '-pt'
            checks[cid] = []
            for block in [0, 1]:
                reference = collected / 'results' / f'{baseline}-block{block}' / 'predictions.json'
                if not reference.exists():
                    checks[cid].append(dict(block=block, passed=False, reason='Eager reference missing'))
                    continue
                a = load(reference)
                b = load(collected / 'results' / f'{cid}-block{block}' / 'predictions.json')
                for name in plan['structures']:
                    delta = dict(energy_eV_per_atom=maxdiff(a[name]['energy'], b[name]['energy'])/a[name]['natoms'],
                                 force_max_eV_per_A=maxdiff(a[name]['forces'], b[name]['forces']),
                                 stress_max_eV_per_A3=maxdiff(a[name]['stress'], b[name]['stress']))
                    checks[cid].append(dict(structure=name, block=block, differences=delta,
                                           passed=all(v <= plan['export_tolerances'][k] for k, v in delta.items())))
    comparisons = {}
    for model in ['dpa4-mini', 'dpa4-neo']:
        for suffix in ['pt2-triton0', 'pt2-triton2']:
            a, b = model + '-pt', model + '-' + suffix
            if a in timings and b in timings:
                comparisons[b + '_versus_pt'] = dict(
                    numerical_gate_passed=all(x['passed'] for x in checks[b]),
                    speedup={n: timings[a][n]['median_ms']/timings[b][n]['median_ms'] for n in plan['structures']})
    for dpa, grace in [('dpa4-mini', 'grace-1l-fp64'), ('dpa4-neo', 'grace-2l-fp32'), ('dpa4-neo', 'grace-2l-fp64')]:
        for suffix in ['pt', 'pt2-triton0', 'pt2-triton2']:
            cid = dpa + '-' + suffix
            if cid in timings and grace in timings:
                comparisons[cid + '_versus_' + grace] = dict(
                    meaning='DPA4 latency divided by GRACE latency; >1 means GRACE faster',
                    dpa4_export_gate=all(x['passed'] for x in checks[cid]) if cid in checks else None,
                    ratio={n: timings[cid][n]['median_ms']/timings[grace][n]['median_ms'] for n in plan['structures']})
    result = dict(status='complete' if not failures else 'partial', timings=timings, failures=failures,
                  export_checks=checks, comparisons=comparisons, tolerances=plan['export_tolerances'],
                  input_hashes=hashes, scope=plan['metric'],
                  limitations='Two process blocks, four fixed structures, single GPU; no equal-accuracy, MD or LAMMPS throughput claim')
    (folder / 'assessment.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], timings=timings, comparisons=comparisons, failures=failures), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    assess(parser.parse_args().run)
