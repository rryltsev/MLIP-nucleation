"""Generate independent Al proposals, label with VASP, validate and export DFT only.

Runs only inside a Slurm allocation. Original XML/OUTCAR remain private artifacts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def proposal(a, root, config_id, parent, split, kind, extra):
    a = a.copy()  # strips all proposal-calculator labels
    a.wrap()
    d = a.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    if len(a) != 32 or d.min() < 2.0:
        raise ValueError("Invalid size or near collision: " + config_id)
    p = root / "structures" / config_id / "POSCAR"
    p.parent.mkdir(parents=True, exist_ok=False)
    write(p, a, format="vasp", direct=True, sort=False)
    p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
    return dict(configuration_id=config_id, parent_id=parent, split=split,
                kind=kind, poscar=str(p.relative_to(root)), poscar_sha256=digest(p),
                minimum_distance_A=float(d.min()), atoms=32, **extra)


def generate(root, cfg):
    g = cfg["generator"]
    rows = []
    for vi, volume in enumerate(g["fcc_volumes_A3_per_atom"]):
        for ri, rattle in enumerate(g["fcc_rattle_A"]):
            i = vi * 3 + ri
            a = bulk("Al", "fcc", a=(4 * volume)**(1/3), cubic=True).repeat((2, 2, 2))
            # Different distortions of one prototype belong to one training group.
            strain = np.eye(3)
            strain[0, 1] = [-0.02, 0.0, 0.02][ri]
            a.set_cell(a.cell.array @ strain, scale_atoms=True)
            a.rattle(rattle, seed=1500+i)
            rows.append(proposal(a, root, "fcc-%02d" % i, g["fcc_parent"],
                                 "train", "fcc_strain_rattle", dict(volume_A3_per_atom=volume, seed=1500+i)))
    for i, (split, temperature, volume) in enumerate(g["md_parents"]):
        seed = g["md_seed_base"] + i
        rng = np.random.RandomState(seed)
        a = bulk("Al", "fcc", a=(4*volume)**(1/3), cubic=True).repeat((2, 2, 2))
        a.calc = EMT()
        MaxwellBoltzmannDistribution(a, temperature_K=g["melt_temperature_K"], rng=rng)
        Stationary(a)
        dyn = Langevin(a, g["timestep_fs"]*units.fs,
                       temperature_K=g["melt_temperature_K"],
                       friction=g["friction_per_fs"]/units.fs, rng=rng, logfile=None)
        dyn.run(g["melt_steps"])
        dyn.set_temperature(temperature_K=temperature)
        dyn.run(g["target_steps"])
        for frame in range(g["frames_per_parent"]):
            dyn.run(g["frame_interval_steps"])
            fractional = a.get_scaled_positions()
            # FCC (200)/(020)/(002) Bragg amplitude; small-cell diagnostic only.
            bragg = float(np.mean([abs(np.exp(8j*np.pi*fractional[:, j]).mean()) for j in range(3)]))
            rows.append(proposal(a, root, "md-%02d-f%d" % (i, frame),
                "emt-md-%02d" % i, split, "emt_disordered_proposal",
                dict(seed=seed, target_ionic_temperature_K=temperature,
                     instantaneous_temperature_K=float(a.get_temperature()),
                     volume_A3_per_atom=volume, fcc_bragg_amplitude=bragg,
                     proposal_step=g["melt_steps"]+g["target_steps"]+(frame+1)*g["frame_interval_steps"])))
    for i, volume in enumerate(g["hcp_volumes_A3_per_atom"]):
        a = bulk("Al", "hcp", a=2.86, c=2.86*np.sqrt(8/3)).repeat((2, 2, 4))
        a.set_cell(a.cell.array*(32*volume/a.get_volume())**(1/3), scale_atoms=True)
        rows.append(proposal(a, root, "hcp-%02d" % i, g["hcp_parent"], "test",
                             "hcp_extrapolation", dict(volume_A3_per_atom=volume)))
    groups = {}
    for r in rows:
        old = groups.setdefault(r["parent_id"], r["split"])
        assert old == r["split"], "Parent leakage"
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")}
    assert counts == cfg["split_policy"]["counts"]
    assert len({r["poscar_sha256"] for r in rows}) == len(rows)
    save(root/"dataset-manifest.json", dict(dataset_id=cfg["dataset_id"],
         generator_sha256=digest(__file__), config_sha256=digest(root/"system.json"),
         ase_version=__import__("ase").__version__, numpy_version=np.__version__,
         labels="none: geometry proposals only", configurations=rows, counts=counts,
         split_policy=cfg["split_policy"]))
    return rows


def incar(encut, sigma, cid):
    return f"""SYSTEM = Al32 {cid}
GGA = PE
ENCUT = {encut}
PREC = Accurate
LREAL = .FALSE.
LASPH = .TRUE.
ADDGRID = .TRUE.
ISPIN = 1
ISYM = 0
ISMEAR = 0
SIGMA = {sigma}
ALGO = Normal
EDIFF = 1E-7
NELM = 160
NELMIN = 4
IBRION = -1
NSW = 0
ISIF = 2
LWAVE = .FALSE.
LCHARG = .FALSE.
NCORE = 4
"""


def parse_label(path, expected_atoms=32):
    """Require one complete, electronically converged static calculation.

    ASE XML reader supplies tensile-positive stress; energy is explicitly free_energy.
    """
    tree = ET.parse(path/"vasprun.xml")  # rejects truncated output
    calcs = tree.findall("calculation")
    params = {e.get("name"): (e.text or "").strip() for e in tree.findall("parameters//i")}
    if len(calcs) != 1 or int(params["NSW"]) != 0:
        raise ValueError("Expected exactly one static calculation")
    nscf = len(calcs[0].findall("scstep"))
    out = (path/"OUTCAR").read_text(errors="replace")
    if not (0 < nscf < int(params["NELM"])) or "aborting loop because EDIFF is reached" not in out:
        raise ValueError("Electronic convergence not established")
    if "General timing and accounting informations for this job" not in out:
        raise ValueError("VASP normal termination missing")
    a = read(path/"vasprun.xml", index=-1, format="vasp-xml")
    energy = float(a.calc.results["free_energy"])
    sigma0 = float(a.calc.results["energy"])
    force = np.asarray(a.get_forces())
    stress = np.asarray(a.get_stress())
    if len(a) != expected_atoms or force.shape != (expected_atoms, 3) or stress.shape != (6,):
        raise ValueError("Invalid label dimensions")
    if not np.isfinite(np.r_[energy, sigma0, force.ravel(), stress]).all():
        raise ValueError("Nonfinite label")
    # Independently verify VASP pressure sign/order against ASE normalization.
    raw = np.array([[float(v) for v in e.text.split()] for e in calcs[0].findall("varray[@name='stress']/v")])
    expected = (-raw*0.1*units.GPa).flat[[0, 4, 8, 5, 2, 1]]
    if not np.allclose(stress, expected, rtol=1e-10, atol=1e-12):
        raise ValueError("Stress conversion disagreement")
    return dict(energy_free_eV=energy, energy_sigma0_eV=sigma0,
                forces_eV_per_A=force.tolist(), stress_eV_per_A3=stress.tolist(),
                volume_A3=float(a.get_volume()), electronic_steps=nscf,
                electronic_converged=True, normal_termination=True,
                source_hashes={n: digest(path/n) for n in ("POSCAR", "INCAR", "KPOINTS", "POTCAR", "OUTCAR", "vasprun.xml")})


def calculate(root, row, name, encut, k, sigma):
    p = root/"calculations"/name
    p.mkdir(parents=True, exist_ok=False)
    (p/"POSCAR").write_bytes((root/row["poscar"]).read_bytes())
    (p/"INCAR").write_text(incar(encut, sigma, row["configuration_id"]))
    (p/"KPOINTS").write_text(f"Gamma mesh\n0\nGamma\n{k} {k} {k}\n0 0 0\n")
    subprocess.run([sys.executable, os.environ["VASP_PAW_ROOT"]+"/vasp_potentials.py", "assemble",
                    os.environ["VASP_PAW_REGISTRY"], "--family", "PBE", "--variants", "Al", "--output", str(p/"POTCAR")], check=True)
    if digest(p/"POTCAR") != "1b07010218b572e3404ce1eea8fed39e24b52859ebef1517bffa2ef69b8d1315":
        raise ValueError("Unexpected POTCAR")
    save(p/"input-manifest.json", dict(configuration_id=row["configuration_id"],
         parent_id=row["parent_id"], split=row["split"],
         input_sha256={n:digest(p/n) for n in ("INCAR", "POSCAR", "KPOINTS", "POTCAR")},
         code_commit=os.environ["PILOT_CODE_COMMIT"], job_id=os.environ["SLURM_JOB_ID"]))
    started = time.time()
    with (p/"vasp.stdout").open("w") as out:
        r = subprocess.run(["mpirun", "-np", os.environ["SLURM_NTASKS"], "vasp_std"],
                           cwd=p, stdout=out, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError("VASP failed: " + name)
    label = parse_label(p)
    label.update(configuration_id=row["configuration_id"], parent_id=row["parent_id"],
                 split=row["split"], wall_seconds=time.time()-started, path=str(p),
                 code_commit=os.environ["PILOT_CODE_COMMIT"], job_id=os.environ["SLURM_JOB_ID"])
    save(p/"label.json", label)
    return label


def differences(a, b):
    return dict(energy_eV_per_atom=abs(a["energy_free_eV"]-b["energy_free_eV"])/len(a["forces_eV_per_A"]),
                max_force_component_eV_per_A=float(np.max(np.abs(np.array(a["forces_eV_per_A"])-b["forces_eV_per_A"]))),
                max_stress_component_eV_per_A3=float(np.max(np.abs(np.array(a["stress_eV_per_A3"])-b["stress_eV_per_A3"]))))


def calibrate(root, cfg, rows):
    checks = []
    for cid in cfg["calibration_configuration_ids"]:
        row = next(r for r in rows if r["configuration_id"] == cid)
        assert row["split"] == "train"
        labels = {name: calculate(root, row, cid+"-"+name, encut, k, sigma)
                  for name, encut, k, sigma in cfg["calibration_cases"]}
        for name in cfg["acceptance"]["required_against_reference"]:
            diff = differences(labels[name], labels["reference"])
            checks.append(dict(configuration_id=cid, case=name, differences=diff,
                               passed=all(v <= cfg["acceptance"][k] for k, v in diff.items())))
    passed = all(c["passed"] for c in checks)
    save(root/"calibration.json", dict(passed=passed, checks=checks,
         protocol=dict(encut_eV=450, gamma_mesh=4, sigma_eV=0.05) if passed else None,
         criterion="All predefined force, stress and absolute energy tolerances; training parents only",
         dataset_manifest_sha256=digest(root/"dataset-manifest.json")))
    if not passed:
        raise RuntimeError("Convergence criteria failed; production blocked")


def label_all(root, cfg, source):
    calibration = json.loads((source/"calibration.json").read_text())
    if not calibration["passed"] or calibration["dataset_manifest_sha256"] != digest(source/"dataset-manifest.json"):
        raise ValueError("Unvalidated protocol or changed structures")
    manifest = json.loads((source/"dataset-manifest.json").read_text())
    rows = manifest["configurations"]
    for r in rows:
        data = (source/r["poscar"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == r["poscar_sha256"]
        dest = root/r["poscar"]
        dest.parent.mkdir(parents=True, exist_ok=False)
        dest.write_bytes(data)
    save(root/"dataset-manifest.json", manifest)
    save(root/"calibration-source.json", dict(path=str(source), sha256=digest(source/"calibration.json"), **calibration))
    labels = []
    for row in rows:
        labels.append(calculate(root, row, row["configuration_id"], 450, 4, 0.05))
        save(root/"labels.json", dict(source="DFT/VASP", contract=cfg["label_contract"], labels=labels,
             complete=len(labels)==len(rows)))
    export(root, cfg, rows, labels)


def export(root, cfg, rows, labels):
    from ase.stress import voigt_6_to_full_3x3_stress
    by_id = {r["configuration_id"]: r for r in rows}
    for split in ("train", "validation", "test"):
        subset = [l for l in labels if l["split"] == split]
        structures = []
        dp = root/"export"/"deepmd"/split
        st = dp/"set.000"
        st.mkdir(parents=True, exist_ok=False)
        coord, box, energy, forces, virial = [], [], [], [], []
        for label in subset:
            row = by_id[label["configuration_id"]]
            a = read(root/row["poscar"], format="vasp")
            a.info.update(configuration_id=row["configuration_id"], parent_id=row["parent_id"], label_source="DFT_VASP")
            e, f, s = label["energy_free_eV"], np.array(label["forces_eV_per_A"]), np.array(label["stress_eV_per_A3"])
            a.calc = SinglePointCalculator(a, energy=e, free_energy=e, forces=f, stress=s)
            structures.append(a)
            coord.append(a.positions.ravel()); box.append(a.cell.array.ravel()); energy.append(e)
            forces.append(f.ravel()); virial.append((-a.get_volume()*voigt_6_to_full_3x3_stress(s)).ravel())
        write(root/"export"/(split+".extxyz"), structures, format="extxyz")
        (dp/"type.raw").write_text(" ".join(["0"]*32)+"\n")
        (dp/"type_map.raw").write_text("Al\n")
        for name, array in (("coord",coord),("box",box),("energy",energy),("force",forces),("virial",virial)):
            np.save(st/(name+".npy"), np.array(array, dtype=np.float64))
        save(dp/"frame-ids.json", [l["configuration_id"] for l in subset])
        # Round-trip checks cover column order, energy type and virial sign.
        reread = read(root/"export"/(split+".extxyz"), index=":")
        for a, b in zip(structures, reread):
            assert np.allclose(a.get_forces(), b.get_forces(), atol=1e-7)
            assert np.allclose(a.get_stress(), b.get_stress(), atol=1e-8)
            assert abs(a.get_potential_energy()-b.get_potential_energy()) < 1e-7
        assert np.allclose(np.load(st/"virial.npy")[0], virial[0])
    files = {str(p.relative_to(root)):digest(p) for p in (root/"export").rglob("*") if p.is_file()}
    save(root/"export-manifest.json", dict(files=files, contract=cfg["label_contract"],
         labels_sha256=digest(root/"labels.json"), dataset_manifest_sha256=digest(root/"dataset-manifest.json"),
         energy_reference="Raw PAW64 TOTEN; no model-dependent shift applied"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["calibrate", "label"])
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Compute execution requires Slurm")
    root = Path.cwd()
    cfg = json.loads((root/"system.json").read_text())
    status = dict(phase=args.phase, job_id=os.environ["SLURM_JOB_ID"], passed=False)
    try:
        if args.phase == "calibrate":
            manifest = json.loads((root/"dataset-manifest.json").read_text())
            if manifest["config_sha256"] != digest(root/"system.json"):
                raise ValueError("Prepared system configuration changed")
            rows = manifest["configurations"]
            groups = {}
            for row in rows:
                if digest(root/row["poscar"]) != row["poscar_sha256"]:
                    raise ValueError("Prepared geometry changed")
                if groups.setdefault(row["parent_id"],row["split"]) != row["split"]:
                    raise ValueError("Parent split leakage")
            calibrate(root, cfg, rows)
        else:
            label_all(root, cfg, args.source)
        status["passed"] = True
    except Exception as e:
        status["error"] = repr(e)
        raise
    finally:
        save(root/"result.json", status)


if __name__ == "__main__":
    main()
