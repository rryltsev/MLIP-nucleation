"""Prepare 40 Al geometries and reviewable, unvalidated VASP inputs locally.

This inexpensive EMT proposal generation is not a production MD calculation.
No VASP, scheduler submission or model training is performed by this command.
Requires ASE 3.29, NumPy and SciPy. Never overwrites an existing dataset.
"""
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("al_pilot_hpc",ROOT/"scripts/hpc/al-pilot.py")
pilot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def main():
    root=ROOT/"datasets/al32-pilot-v1"
    root.mkdir(parents=True,exist_ok=False)
    data=(ROOT/"configs/systems/al32-pilot-v1.json").read_bytes().replace(b"\r\n",b"\n")
    (root/"system.json").write_bytes(data)
    cfg=json.loads(data)
    rows=pilot.generate(root,cfg)
    inputs={}
    for row in rows:
        p=root/row["poscar"]
        inc=p.with_name("INCAR")
        inc.write_bytes(pilot.incar(450,0.05,row["configuration_id"]).encode())
        kp=p.with_name("KPOINTS")
        kp.write_bytes(b"UNVALIDATED candidate Gamma mesh\n0\nGamma\n4 4 4\n0 0 0\n")
        for source in (p,inc,kp): inputs[source.relative_to(root).as_posix()]=pilot.digest(source)
    for cid in cfg["calibration_configuration_ids"]:
        row=next(r for r in rows if r["configuration_id"]==cid)
        assert row["split"]=="train"
        for name,encut,k,sigma in cfg["calibration_cases"]:
            dest=root/"calibration-inputs"/cid/name
            dest.mkdir(parents=True)
            (dest/"POSCAR").write_bytes((root/row["poscar"]).read_bytes())
            (dest/"INCAR").write_bytes(pilot.incar(encut,sigma,cid).encode())
            (dest/"KPOINTS").write_bytes(f"Calibration Gamma mesh\n0\nGamma\n{k} {k} {k}\n0 0 0\n".encode())
            for source in dest.iterdir(): inputs[source.relative_to(root).as_posix()]=pilot.digest(source)
    pilot.save(root/"preparation.json",dict(
        code_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        input_sha256=inputs, accepted_DFT_labels=0,
        status="geometries and draft inputs prepared; protocol unvalidated; no DFT executed",
        potcar_policy="Assemble privately on cluster from pinned PBE Al; never store licensed content in Git",
        calibration_configuration_ids=cfg["calibration_configuration_ids"]))
    print(json.dumps(dict(dataset=str(root),configurations=len(rows),accepted_DFT_labels=0)))


if __name__=="__main__": main()
