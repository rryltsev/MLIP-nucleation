"""Bounded GPU runtime/training probes on explicitly synthetic data, not DFT."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset(root):
    import numpy as np
    rng = np.random.default_rng(20261017)
    # Four independent atoms; harmonic pair labels with no periodic image interactions.
    origin = np.array([[3., 3., 3.], [5.6, 3., 3.], [3., 5.6, 3.], [3., 3., 5.6]])
    for split, count in [("train", 8), ("validation", 4)]:
        base = root / split
        (base / "set.000").mkdir(parents=True)
        (base / "type.raw").write_text("0\n0\n0\n0\n")
        (base / "type_map.raw").write_text("Al\n")
        coords, energies, forces, virials = [], [], [], []
        for _ in range(count):
            r = origin + rng.normal(0., 0.06, origin.shape)
            f = np.zeros_like(r)
            e = 0.
            for i in range(4):
                for j in range(i):
                    delta = r[i] - r[j]
                    distance = np.linalg.norm(delta)
                    e += 0.5 * (distance - 2.5)**2
                    force = -(distance - 2.5) * delta / distance
                    f[i] += force
                    f[j] -= force
            coords.append(r.reshape(-1)); energies.append(e); forces.append(f.reshape(-1))
            virials.append((r.T @ f).reshape(-1))
        for name, values in {"coord": coords, "energy": energies, "force": forces,
                             "virial": virials, "box": np.tile(np.eye(3).reshape(1, 9) * 12., (count, 1))}.items():
            np.save(base / "set.000" / (name + ".npy"), np.asarray(values, dtype=np.float64))
    files = {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()}
    save(root / "provenance.json", {
        "label_source": "synthetic harmonic pairs, NOT DFT or teacher predictions",
        "purpose": "software runtime check only; no scientific accuracy conclusions",
        "seed": 20261017, "energy_unit": "eV", "force_unit": "eV/angstrom",
        "virial_unit": "eV", "virial_definition": "sum_i r_i tensor F_i; row-major xx xy xz yx yy yz zx zy zz",
        "split": "independent random configurations; no held-out scientific test set used",
        "input_sha256": files,
    })


def run(command, cwd, name, seconds=100):
    start = time.monotonic()
    log = cwd / (name + ".log")
    with log.open("wb") as out:
        try:
            proc = subprocess.run(command, cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT,
                                  timeout=seconds, check=False)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 124
    return {"command": command, "returncode": code, "elapsed_seconds": time.monotonic() - start,
            "log": str(log), "log_sha256": digest(log), "tail": log.read_text(errors="replace")[-3500:]}


def gpu_probe(backend):
    if backend == "pt":
        import torch
        assert torch.cuda.is_available(), "PyTorch cannot use CUDA"
        a = torch.randn(32, 32, device="cuda", requires_grad=True)
        (a @ a.T).square().mean().backward(); torch.cuda.synchronize()
        assert a.grad.is_cuda and torch.isfinite(a.grad).all().item()
        return {"backend": "pytorch", "version": torch.__version__, "cuda_build": torch.version.cuda,
                "device": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
                "compiled_arches": torch.cuda.get_arch_list(), "gradient_device": str(a.grad.device)}
    import tensorflow as tf
    devices = tf.config.list_physical_devices("GPU")
    assert devices, "TensorFlow cannot see a GPU"
    for d in devices:
        tf.config.experimental.set_memory_growth(d, True)
    tf.config.set_soft_device_placement(False)
    with tf.device("/GPU:0"):
        a = tf.Variable(tf.random.normal([32, 32]))
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(tf.matmul(a, a, transpose_b=True)))
        gradient = tape.gradient(loss, a)
        assert bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())
        assert "GPU" in gradient.device, gradient.device
    return {"backend": "tensorflow", "version": tf.__version__, "build": tf.sysconfig.get_build_info(),
            "devices": [str(d) for d in devices], "gradient_device": gradient.device}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pt", "tf"], required=True)
    args = parser.parse_args()
    root = Path.cwd()
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "host": os.uname().nodename,
              "python": sys.version, "prefix": sys.prefix, "backend": args.backend,
              "job_id": os.environ.get("SLURM_JOB_ID"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "status": "running", "cases": []}
    result_path = root / "result.json"
    try:
        result["packages"] = {p: importlib.metadata.version(p) for p in ["deepmd-kit", "numpy"]}
        result["gpu"] = gpu_probe(args.backend)
        save(result_path, result)
        dataset(root / "data")
        variants = ["se_e2_a", "dpa4", "dpa4c"] if args.backend == "pt" else ["se_e2_a"]
        for variant in variants:
            d = root / variant; d.mkdir()
            model = {"type_map": ["Al"], "descriptor": {"type": "se_e2_a", "sel": [16],
                     "rcut": 5., "rcut_smth": 4.5, "neuron": [8, 8], "axis_neuron": 4, "seed": 42},
                     "fitting_net": {"neuron": [8, 8], "seed": 42}}
            flag = ["--pt"] if args.backend == "pt" else []
            if variant == "dpa4":
                model = {"type": "dpa4", "type_map": ["Al"],
                         "descriptor": {"sel": 16, "rcut": 5., "channels": 8, "n_radial": 8,
                                        "lmax": 2, "mmax": 1, "n_blocks": 1, "mixing_layers": 1,
                                        "use_amp": False, "precision": "float32", "seed": 42},
                         "fitting_net": {"neuron": [8], "precision": "float32", "seed": 42}}
            if variant == "dpa4c":
                flag = ["--pt-expt"]
                model = {"type_map": ["Al"], "descriptor": {"type": "dpa4c", "rcut": 5.,
                         "channels": 8, "lmax": 2, "n_radial": 8, "use_amp": False,
                         "precision": "float32", "seed": 42},
                         "fitting_net": {"neuron": [8], "precision": "float32", "seed": 42}}
            config = {"model": model, "learning_rate": {"type": "exp", "start_lr": 0.001,
                      "stop_lr": 0.0001, "decay_steps": 5}, "loss": {"type": "ener", "start_pref_e": 1.,
                      "limit_pref_e": 1., "start_pref_f": 1., "limit_pref_f": 1.,
                      "start_pref_v": 0., "limit_pref_v": 0.},
                      "training": {"training_data": {"systems": [str(root / "data/train")], "batch_size": 1},
                      "validation_data": {"systems": [str(root / "data/validation")], "batch_size": 1, "numb_btch": 1},
                      "numb_steps": 20, "seed": 42, "disp_freq": 5, "save_freq": 20,
                      "disp_file": "lcurve.out", "save_ckpt": "model.ckpt"}}
            save(d / "input.json", config)
            case = {"variant": variant, "backend_flag": flag, "input_sha256": digest(d / "input.json")}
            case["train"] = run([str(Path(sys.prefix) / "bin/dp")] + flag + ["train", "input.json"], d, "train")
            curve = d / "lcurve.out"
            if case["train"]["returncode"] == 0 and curve.exists():
                import numpy as np
                values = np.loadtxt(curve, ndmin=2)
                case["finite_learning_curve"] = bool(np.isfinite(values).all())
                case["last_step"] = int(values[-1, 0])
                case["checkpoints"] = {p.name: digest(p) for p in d.glob("model.ckpt*") if p.is_file()}
            case["status"] = "passed" if (case.get("finite_learning_curve") and case.get("last_step", 0) >= 20
                                                   and case.get("checkpoints")) else "failed"
            result["cases"].append(case); save(result_path, result)
        result["status"] = "passed" if all(c["status"] == "passed" for c in result["cases"]) else "partial_or_failed"
    except Exception:
        result["status"] = "failed"; result["error"] = traceback.format_exc()
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(result_path, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
