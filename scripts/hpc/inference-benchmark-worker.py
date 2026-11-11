"""Synchronized, cache-bypassed E/F/stress latency on identical input frames."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import time
import traceback


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    parser.add_argument('block', type=int)
    args = parser.parse_args()
    root = Path.cwd()
    plan = json.loads((root / 'plan.json').read_text())
    case = next(c for c in plan['cases'] if c['id'] == args.case)
    out = root / 'results' / f'{args.case}-block{args.block}'
    out.mkdir(parents=True, exist_ok=False)
    try:
        assert os.environ.get('SLURM_JOB_ID')
        import numpy as np
        from ase.io import read
        from ase.calculators.calculator import all_changes
        for name, digest in plan['input_sha256'].items():
            assert sha(root / name) == digest, name
        for name, digest in case['source_files'].items():
            assert sha(name) == digest, name
        source = case['path']
        started = time.perf_counter()
        if case['backend'] == 'dpa4':
            import torch
            from deepmd.calculator import DP
            assert torch.cuda.is_available() and torch.cuda.device_count() == 1
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if case['format'] == 'pt2':
                source = str(root / 'exports' / (case['id'] + '.pt2'))
                export = json.loads((root / 'exports' / (case['id'] + '.json')).read_text())
                assert export['returncode'] == 0 and sha(source) == export['sha256']
            calc = DP(model=source)
            sync = torch.cuda.synchronize
            device = torch.cuda.get_device_name(0)
            versions = {p: metadata.version(p) for p in ['torch', 'deepmd-kit', 'numpy', 'ase']}
        else:
            import tensorflow as tf
            from tensorpotential import TensorPotential
            from tensorpotential.instructions.base import load_instructions
            from tensorpotential.calculator import TPCalculator
            devices = tf.config.list_physical_devices('GPU')
            assert len(devices) == 1
            tf.config.experimental.set_memory_growth(devices[0], True)
            tf.config.experimental.enable_tensor_float_32_execution(False)
            dtype = tf.float32 if case['precision'] == 'fp32' else tf.float64
            model = TensorPotential(potential=load_instructions(str(Path(source) / 'model.yaml')),
                                    param_dtype=dtype, jit_compile=False)
            model.load_checkpoint(checkpoint_name=str(Path(source) / 'checkpoint'),
                                  expect_partial=True, assert_existing_objects_matched=True, model_only=True)
            model.model.decorate_compute_function(jit_compile=True)
            assert {v.dtype.name for v in model.model.variables_to_train} == {dtype.name}
            def calculator():
                return TPCalculator(model.model, enable_uq_if_available=False, mode='diverse',
                                    adaptive_padding=False, max_number_reduction_recompilation=0)
            calc = calculator()
            # TPCalculator materializes every result as numpy; this synchronizes TF.
            sync = lambda: None
            device = tf.config.experimental.get_device_details(devices[0]).get('device_name')
            versions = {p: metadata.version(p) for p in ['tensorflow', 'tensorpotential', 'numpy', 'ase']}
        load_seconds = time.perf_counter() - started

        def evaluate(atoms):
            sync()
            begin = time.perf_counter()
            calc.calculate(atoms, properties=['energy', 'forces', 'stress'], system_changes=all_changes)
            result = {k: np.asarray(calc.results[k]).copy() for k in ['energy', 'forces', 'stress']}
            sync()
            elapsed = time.perf_counter() - begin
            assert all(np.isfinite(x).all() for x in result.values())
            return result, elapsed

        timings, predictions = [], {}
        for name in plan['structures']:
            if case['backend'] == 'grace':
                calc = calculator()
            base = read(root / 'inputs' / (name + '.extxyz'))
            # Identical frame sequence in every process, backend and precision.
            rng = np.random.default_rng(20261111)
            frames = []
            for _ in range(1 + plan['warmup_calls'] + plan['measured_calls']):
                at = base.copy()
                at.positions += rng.normal(0, 0.002, at.positions.shape)
                frames.append(at)
            value, cold = evaluate(frames[0])
            predictions[name] = {k: x.tolist() for k, x in value.items()}
            predictions[name]['natoms'] = len(base)
            for at in frames[1:1 + plan['warmup_calls']]:
                evaluate(at)
            samples = [evaluate(at)[1] for at in frames[1 + plan['warmup_calls']:]]
            timings.append(dict(name=name, natoms=len(base), cold_seconds=cold, wall_seconds=samples,
                                median_seconds=float(np.median(samples)),
                                p10_seconds=float(np.quantile(samples, .1)),
                                p90_seconds=float(np.quantile(samples, .9))))
            save(out / 'progress.json', dict(timings=timings))
        save(out / 'predictions.json', predictions)
        summary = dict(status='PASS', case=case, block=args.block, timings=timings,
                       load_seconds=load_seconds, versions=versions, device=device,
                       job_id=os.environ['SLURM_JOB_ID'], node=os.uname().nodename,
                       model_used=source, model_sha256=sha(source) if Path(source).is_file() else case['source_files'],
                       environment={k: os.environ.get(k) for k in ['DP_COMPILE_INFER', 'DP_TRITON_INFER',
                                    'DP_TF32_INFER', 'DP_AMP_INFER', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES']},
                       scope='Full ASE calculate E/F/stress including neighbors, transfers and synchronization; no result cache; no MD')
        save(out / 'summary.json', summary)
        print(json.dumps({args.case: {r['name']: r['median_seconds'] for r in timings}}), flush=True)
    except Exception:
        save(out / 'failure.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
