"""Bounded actual FM fine-tuning on synthetic labels, with initialization checks."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    p.write_text(json.dumps(obj, indent=2) + "\n")


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), Path(__file__).with_name(name + ".py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def execute(command, work, result):
    env = dict(os.environ, FM_LOADED_SNAPSHOT=str(work / "loaded-initial" / "model"))
    telemetry = []; start = time.monotonic()
    with (work / "train.log").open("wb") as out:
        proc = subprocess.Popen(command, cwd=work, stdout=out, stderr=subprocess.STDOUT, env=env)
        while proc.poll() is None:
            if time.monotonic() - start > 360:
                proc.kill(); proc.wait(); raise TimeoutError("Fine-tuning exceeded 360 seconds")
            q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            telemetry.extend({"seconds":time.monotonic()-start,"process":line} for line in q.stdout.splitlines() if line.split(",")[0].strip() == str(proc.pid))
            time.sleep(2)
    save(work / "gpu-training-process.json", telemetry)
    result["training"] = {"command":command,"exitcode":proc.returncode,"seconds":time.monotonic()-start,"pid":proc.pid,"gpu_samples":len(telemetry)}
    assert proc.returncode == 0, (work / "train.log").read_text(errors="replace")[-4000:]
    assert telemetry, "Training process not observed on GPU"


def compare(before, loaded, after, backend):
    import numpy as np
    if backend == "dpa4":
        import torch
        def tensors(p):
            d = torch.load(p, map_location="cpu", weights_only=False)
            d = d.get("model", d)
            return {k:v.numpy() for k,v in d.items() if isinstance(v, torch.Tensor) and v.is_floating_point()}
    else:
        import tensorflow as tf
        from tensorflow.core.protobuf import trackable_object_graph_pb2
        def tensors(p):
            r = tf.train.load_checkpoint(str(p))
            graph = trackable_object_graph_pb2.TrackableObjectGraph()
            graph.ParseFromString(r.get_tensor("_CHECKPOINTABLE_OBJECT_GRAPH"))
            model_id = next(c.node_id for c in graph.nodes[0].children if c.local_name == "model")
            # Optimizer aliases can change canonical on-disk tensor keys. Walk
            # the model subtree instead, then resolve each attribute's saved key.
            queue = [(model_id,"model")]; seen = set(); values = {}
            while queue:
                node_id, name = queue.pop(0)
                if node_id in seen: continue
                seen.add(node_id); node = graph.nodes[node_id]
                for a in node.attributes:
                    dtype = r.get_variable_to_dtype_map().get(a.checkpoint_key)
                    if dtype is not None and dtype.is_floating:
                        values[name+"/"+a.name] = r.get_tensor(a.checkpoint_key)
                queue.extend((c.node_id,name+"/"+c.local_name) for c in sorted(node.children,key=lambda c:c.local_name))
            save(Path('tf-inventory-'+hashlib.sha256(str(p).encode()).hexdigest()[:8]+'.json'),
                 {'checkpoint':str(p),'model_tensors':{k:{'shape':list(v.shape),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in values.items()},
                  'attributes':[{'full_name':a.full_name,'checkpoint_key':a.checkpoint_key} for n in graph.nodes for a in n.attributes]})
            return values
    source, initial, final = map(tensors, [before, loaded, after])
    assert source and source.keys() == initial.keys() == final.keys(), str({
        'source_only':sorted(source.keys()-initial.keys()),'initial_only':sorted(initial.keys()-source.keys()),
        'final_only':sorted(final.keys()-initial.keys()),'final_missing':sorted(initial.keys()-final.keys())})
    assert all(np.isfinite(a).all() for d in [source, initial, final] for a in d.values())
    equal = [k for k in source if np.array_equal(source[k], initial[k])]
    unequal = [k for k in source if k not in equal]
    # DeepMD adjusts the target energy bias from TRAIN data before fine-tuning.
    allowed_keys = {"model.Default.atomic_model.out_bias", "model.Default.atomic_model.out_std", "model.Default.atomic_model.descriptor.version_tensor"}
    allowed = backend == "dpa4" and set(unequal) <= allowed_keys
    assert not unequal or allowed, "Unexpected differences in loaded FM parameters: " + str(unequal)
    changed = [k for k in initial if not np.array_equal(initial[k], final[k])]
    learned = [k for k in changed if k in equal and initial[k].size > 1]
    assert learned, "No original pretrained parameter tensor changed during training"
    return {"source_float_tensors":len(source),"exactly_restored_tensors":len(equal),"initial_changed_keys":unequal,
            "initial_adjustments":{k:{"max_abs_difference":float(np.max(np.abs(initial[k]-source[k]))),
                                       "before":source[k].tolist() if source[k].size==1 else "array",
                                       "after":initial[k].tolist() if initial[k].size==1 else "array"} for k in unequal},
            "updated_tensors":len(changed),"updated_original_tensors":len(learned),
            "max_abs_update":max(float(np.max(np.abs(final[k]-initial[k]))) for k in changed)}


def main():
    backend = sys.argv[1]; root = Path.cwd(); src = Path(__file__).parent
    model = json.loads((src / (backend + "-checkpoint.json")).read_text())
    result = {"status":"running","backend":backend,"job_id":os.environ.get("SLURM_JOB_ID"),
              "host":os.uname().nodename,"environment":sys.prefix,"started_utc":datetime.now(timezone.utc).isoformat(),
              "model_id":model["model_id"],"source_sha256":model["sha256"],
              "scope":"synthetic Al4 diagnostic, NOT DFT, not a scientific accuracy benchmark"}
    try:
        for item in model["files"].values():
            assert digest(Path(item["remote_path"])) == item["sha256"]
        if backend == "dpa4":
            helper = load_helper("mlip-smoke"); helper.dataset(root / "data")
            source = Path(model["remote_root"]) / model["filename"]
            config = json.loads((Path(model["remote_root"]) / model["configuration"]).read_text())
            # Same architecture, type map, precision and AMP; skip graph compilation for this bounded check.
            config["model"]["use_compile"] = False
            result["runtime_overrides"] = {"model.use_compile":False}
            config["optimizer"] = {"type":"Adam"}
            config["learning_rate"] = {"type":"exp","start_lr":0.0001,"stop_lr":0.00001,"decay_steps":5}
            config["loss"] = {"type":"ener","start_pref_e":1.,"limit_pref_e":1.,"start_pref_f":1.,"limit_pref_f":1.,"start_pref_v":0.,"limit_pref_v":0.}
            config["training"] = {"training_data":{"systems":[str(root/"data/train")],"batch_size":1},
                "validation_data":{"systems":[str(root/"data/validation")],"batch_size":1,"numb_btch":4},
                "numb_steps":20,"seed":42,"disp_freq":5,"save_freq":20,"disp_file":"lcurve.out","save_ckpt":"model.ckpt"}
            save(root / "input.json", config)
            execute([sys.executable,str(src/"fm-child.py"),backend,"--pt","train","input.json","--finetune",str(source)],root,result)
            import numpy as np
            values = np.loadtxt(root / "lcurve.out", ndmin=2)
            assert np.isfinite(values).all() and int(values[-1,0]) == 20
            result["optimizer_updates"] = 20
            outputs = sorted(root.glob("model.ckpt-*.pt"))
            assert len(outputs) == 1, str(outputs)
            result["weight_checks"] = compare(source,root/"loaded-initial/model",outputs[0],backend)
        else:
            import shutil
            import yaml
            shutil.copyfile(src / "mlip-smoke.py", root / "mlip-smoke.py")
            helper = load_helper("grace-smoke"); helper.prepare_data(root)
            source = Path(next(v["remote_path"] for k,v in model["files"].items() if k.endswith("model.yaml"))).parent
            config = {"seed":42,"cutoff":6.,"data":{"filename":str(root/"data/train.extxyz"),
                "test_filename":str(root/"data/validation.extxyz"),"reference_energy":0,"max_workers":1},
                "potential":{"filename":str(source/"model.yaml"),"checkpoint_name":str(source/"checkpoint"),"shift":False,"reduce_elements":False},
                "fit":{"maxiter":5,"optimizer":"Adam","opt_params":{"learning_rate":0.0001,"use_ema":False},
                "reset_epoch_and_step":True,"reset_optimizer":True,"eval_init_stats":True,
                "scheduler":"cosine_decay","scheduler_params":{"minimum_learning_rate":0.00001},
                "loss":{"energy":{"type":"square","weight":1.},"forces":{"type":"square","weight":1.}},
                "batch_size":2,"test_batch_size":2,"jit_compile":False,"checkpoint_freq":1,
                "train_max_n_buckets":1,"test_max_n_buckets":1}}
            (root/"input.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
            execute([sys.executable,str(src/"fm-child.py"),backend,"input.yaml"],root,result)
            metrics = {}
            for split in ["train","test"]:
                values = yaml.safe_load((root / f"seed/42/{split}_metrics.yaml").read_text())
                assert helper.finite(values) and int(values[-1]["step"]) >= 20 and int(values[-1]["epoch"]) >= 5
                metrics[split] = values[-1]
            result["metrics"] = metrics; result["optimizer_updates"] = int(metrics["train"]["step"])
            result["weight_checks"] = compare(source/"checkpoint",root/"loaded-initial/model",root/"seed/42/checkpoints/checkpoint",backend)
        result["output_sha256"] = {p.relative_to(root).as_posix():digest(p) for p in root.rglob("*") if p.is_file() and "cache" not in p.parts}
        result["status"] = "passed"
    except Exception:
        result["status"] = "failed"; result["error"] = traceback.format_exc()
    result["finished_utc"] = datetime.now(timezone.utc).isoformat(); save(root/"result.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__": sys.exit(main())
