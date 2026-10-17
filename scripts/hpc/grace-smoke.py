"""Short GRACE 1-layer GPU fit on explicitly synthetic harmonic-pair labels."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def finite(value):
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, list): return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value, (int, float)) else True


def prepare_data(root):
    import numpy as np
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write
    from ase.stress import full_3x3_to_voigt_6_stress
    spec = importlib.util.spec_from_file_location("synthetic_mlip_data", root / "mlip-smoke.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    helper.dataset(root / "data")
    for split in ["train", "validation"]:
        base = root / "data" / split / "set.000"
        coord = np.load(base / "coord.npy").reshape(-1, 4, 3)
        energy = np.load(base / "energy.npy")
        force = np.load(base / "force.npy").reshape(-1, 4, 3)
        virial = np.load(base / "virial.npy").reshape(-1, 3, 3)
        frames = []
        for i, r in enumerate(coord):
            atoms = Atoms("Al4", positions=r, cell=[12., 12., 12.], pbc=True)
            atoms.info.update(label_source="synthetic_harmonic_pairs_NOT_DFT", parent_id=split + "-" + str(i))
            stress = full_3x3_to_voigt_6_stress(-virial[i] / atoms.get_volume())
            atoms.calc = SinglePointCalculator(atoms, energy=float(energy[i]), forces=force[i], stress=stress)
            frames.append(atoms)
        write(root / "data" / (split + ".extxyz"), frames, format="extxyz")
    p = root / "data/provenance.json"
    provenance = json.loads(p.read_text())
    provenance["extxyz_sha256"] = {p.name: digest(p) for p in (root / "data").glob("*.extxyz")}
    provenance["ase_stress"] = "-virial/volume, eV/angstrom^3; Voigt xx yy zz yz xz xy; stress not fitted"
    save(p, provenance)


def main():
    root = Path.cwd()
    result = {"status":"running", "started_utc":datetime.now(timezone.utc).isoformat(),
              "job_id":os.environ.get("SLURM_JOB_ID"), "host":os.uname().nodename,
              "environment":sys.prefix, "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
              "scope":"synthetic training from scratch only; no foundation weights or DFT labels"}
    try:
        import numpy as np
        import yaml
        import tensorpotential
        import tensorflow as tf
        result["versions"] = {p:md.version(p) for p in ["tensorpotential","tensorflow","tf-keras","numpy","ase","dpdata"]}
        devices = tf.config.list_physical_devices("GPU")
        assert len(devices) == 1, "Expected one Slurm-allocated visible GPU"
        for d in devices: tf.config.experimental.set_memory_growth(d, True)
        with tf.device("/GPU:0"):
            x = tf.Variable(tf.ones([32,32]))
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(tf.matmul(x,x)))
            gradient = tape.gradient(loss,x)
            assert "GPU:0" in gradient.device and np.isfinite(gradient.numpy()).all()
        result["gpu_probe"] = {"devices":[str(d) for d in devices], "gradient_device":gradient.device,
                               "build":tf.sysconfig.get_build_info()}
        save(root / "result.json", result)
        prepare_data(root)
        train = root / "train"; train.mkdir()
        config = {"seed":42,"cutoff":5.,
                  "data":{"filename":str(root / "data/train.extxyz"),"test_filename":str(root / "data/validation.extxyz"),
                          "reference_energy":0,"max_workers":1},
                  "potential":{"preset":"GRACE_1LAYER_latest","param_dtype":"float32","scale":False,
                    "kwargs":{"lmax":1,"n_rad_base":4,"n_rad_max":8,"prod_func_n_max":8,
                              "embedding_size":8,"n_mlp_dens":4}},
                  "fit":{"maxiter":5,"optimizer":"Adam","opt_params":{"learning_rate":0.001,"use_ema":False},
                         "loss":{"energy":{"type":"square","weight":1.},"forces":{"type":"square","weight":1.}},
                         "batch_size":2,"test_batch_size":2,"jit_compile":False,"checkpoint_freq":1,
                         "train_max_n_buckets":1,"test_max_n_buckets":1}}
        (train / "input.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        result["input_sha256"] = digest(train / "input.yaml")
        command = [str(Path(sys.prefix) / "bin/gracemaker"), "input.yaml"]
        telemetry = []
        start = time.monotonic()
        with (train / "train.log").open("wb") as out:
            proc = subprocess.Popen(command,cwd=str(train),stdout=out,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if time.monotonic() - start > 450:
                    proc.kill(); proc.wait(); raise TimeoutError("GRACE training exceeded 450 seconds")
                sample = subprocess.run(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
                                        capture_output=True,text=True,timeout=10)
                for line in sample.stdout.splitlines():
                    if line.split(",")[0].strip() == str(proc.pid):
                        telemetry.append({"elapsed_seconds":time.monotonic()-start,"process":line})
                time.sleep(2)
        result["train"] = {"command":command,"returncode":proc.returncode,"elapsed_seconds":time.monotonic()-start,
                           "pid":proc.pid,"log_sha256":digest(train / "train.log")}
        save(root / "gpu-training-process.json", telemetry)
        assert proc.returncode == 0, (train / "train.log").read_text(errors="replace")[-3500:]
        assert telemetry, "Training child process not observed using the GPU"
        output = train / "seed/42"
        metrics = {}
        for split in ["train","test"]:
            path = output / (split + "_metrics.yaml")
            values = yaml.safe_load(path.read_text())
            assert values and finite(values), "Nonfinite or absent metrics: " + split
            assert int(values[-1]["epoch"]) >= 5 and int(values[-1]["step"]) >= 20
            metrics[split] = {"epochs":len(values),"last":values[-1],"sha256":digest(path)}
        result["metrics"] = metrics
        index = output / "checkpoints/checkpoint.index"
        assert index.exists(), "No regular checkpoint"
        reader = tf.train.load_checkpoint(str(index)[:-6])
        result["checkpoint_step"] = int(reader.get_tensor("step/.ATTRIBUTES/VARIABLE_VALUE"))
        assert result["checkpoint_step"] >= 20
        for name,dtype in reader.get_variable_to_dtype_map().items():
            if dtype.is_floating:
                assert np.isfinite(reader.get_tensor(name)).all(), "Nonfinite checkpoint tensor: " + name
        result["output_sha256"] = {p.relative_to(train).as_posix():digest(p) for p in output.rglob("*")
                                  if p.is_file() and (p.name.startswith("checkpoint") or p.suffix in [".yaml",".pb"] or "final_model" in p.parts)}
        assert (output / "final_model/saved_model.pb").exists()
        result["status"] = "passed"
    except Exception:
        result["status"] = "failed"; result["error"] = traceback.format_exc()
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(root / "result.json", result)
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__": sys.exit(main())
