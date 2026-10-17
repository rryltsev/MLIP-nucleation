# MLIP-nucleation

Research infrastructure for **Programming Crystal Nucleation through Interfacial Liquid Order in Metallic Alloys**.

The source research brief is preserved in `project/brief.txt`. Scientific manuscripts and reports will be written in LaTeX. This repository currently contains configuration and operational skills, not a working simulation pipeline or scientific results.

## Infrastructure

- Remote repository: https://github.com/rryltsev/MLIP-nucleation
- Cluster: `umt.imm.uran.ru:22`, user `u3129`, scheduler Slurm.
- Observed Slurm configuration (2026-10-17): 74 unique nodes, 1850 Slurm CPUs, 2 A100, 18 K40m and 62 M2090 GPUs, including configured devices on unavailable nodes. Partition totals overlap. GPU runtime evidence below applies only to the tested allocations.
- DFT engine: VASP 6.6.1 CPU/MPI installed in `/home/u3129/.local/opt/vasp/6.6.1-oneapi-2025`; `vasp_std`, `vasp_gam`, `vasp_ncl` passed four selected upstream cases on `apollo31`. See `reports/vasp-installation-2026-10-17.tex`.
- A100 DPA4/DPA4c training and K40m legacy DeepMD training passed small synthetic Slurm checks. Production-size memory/performance and actual FM fine-tuning remain untested. See `reports/mlip-environment-2026-10-17.tex`.

## Repository layout

Implemented: `project/` for the research brief and durable state; `configs/` for infrastructure profiles; `.agents/skills/` for reusable workflows; `scripts/` for local checks.

Create these directories when their first real content exists:

```text
src/mlip_research/    structure generation, code adapters, analysis
workflows/           persistent execution and dependencies
configs/systems/     compositions, surfaces, thermodynamic conditions
configs/dft/         converged VASP protocols
configs/training/    teacher and student experiments
datasets/            manifests and group-based splits
models/              model cards, hashes and artifact references
runs/                immutable run manifests and collected summaries
results/             reproducible tables and figures
manuscripts/         LaTeX, BibTeX and figures
reports/             LaTeX research reports
environments/        locked software environments
tests/               scientifically meaningful verification
```

## Next milestones

1. SSH, inventory, VASP/PAW setup and small DeepMD GPU training checks completed on 2026-10-17. Install and validate the separate GRACE environment, then select and test actual trainable foundation checkpoints. See the reports below for evidence and limitations.
2. On another computer clone the repository, configure its own credentials and read `project/state.json`.
3. Compare candidate alloy/nucleant systems using literature novelty, experimental access, DFT feasibility and model chemical coverage. Choose one small pilot.
4. Implement one executable path: structures → converged VASP labels → teacher fine-tuning → independent DFT validation → distillation → student validation → one observable → LaTeX report.
5. Add active learning and rare-event methods after the pilot is validated.

## Storage

Commit source, configuration, skills, manifests, small curated results, LaTeX and bibliography. Choose LFS or durable external storage before producing large datasets, checkpoints and trajectories. Keep immutable SHA-256 manifests and verify retrieval; cluster scratch is not an archive. VASP POTCAR files and credentials must not be uploaded.

## SSH

`configs/ssh_config` is a portable configuration, not an installed user setting. Example from the repository root:

```powershell
python scripts/cluster_ssh.py hostname
```

The helper resolves the profile and `configs/known_hosts` relative to its own location. The public server key was independently confirmed by the user; its ED25519 fingerprint is `SHA256:sAMo6jV6KxAx0ntkLR5Gjm5wVWtb4QuRsDvGBhRB01k`. Public-key-only noninteractive login is verified on DESKTOP-J6E8U22. Authentication is machine-local; do not put credentials in this repository.

Run the offline configuration check with `python scripts/check_project.py`.

### Machine-local SSH key

One-time setup in an interactive terminal (enter the cluster password only at the SSH prompt):

```powershell
python scripts/cluster_ssh.py --setup-key
```

The script creates an ED25519 key without a passphrase for unattended jobs at `%USERPROFILE%/.ssh/id_ed25519_mlip_umt` on Windows, or `~/.ssh/id_ed25519_mlip_umt` otherwise. It preserves existing keys and authorized-key entries, installs the public key through password authentication, then verifies a separate public-key-only connection. The private key stays outside Git. Use a separate key on each computer.

After setup, commands can run from any working directory:

```powershell
python scripts/cluster_ssh.py hostname
python scripts/cluster_ssh.py 'sinfo -o "%P %a %l %D %G"'
```

The helper selects the local key explicitly and pins the verified server key. Setup status is stored in ignored `artifacts/ssh-setup.local.json` without passwords or private key content.

On this computer the public key was installed through an existing authenticated cluster session after password attempts in the setup window were rejected. A separate noninteractive connection verified the installed key. The local key fingerprint is `SHA256:jqLNBsyLJeE6thwpPCNI/6cPoCsmdypfllz/OFuCnC0`.

### Read-only cluster inventory

```powershell
python scripts/inspect_cluster.py --output artifacts/latest-cluster-inventory.json
```

Each probe records its own command, UTC timestamp, output and errors and has remote/local time limits. No jobs are submitted. The committed snapshot is under `configs/observations/`; scheduler availability changes over time.

The default account is `imet-2026-2` with QoS `normal`. Candidate partitions are `apollo` for CPU DFT and `v100` for A100 training. Initial activation observations are documented in the cluster inventory report; subsequent compute-node training checks are described below. No existing environment was modified.

### MLIP training environments

The preferred existing A100 environment for initial DPA4/DPA4c work is `/home/u3129/miniconda3/envs/lammps_deepmd_dpa4` (DeePMD 3.2.0, Torch 2.11.0+cu130, vesin-torch 0.6.1). Despite its name it includes the training CLI. Inside a Slurm allocation, source `/home/u3129/miniconda3/etc/profile.d/conda.sh` and run `conda activate lammps_deepmd_dpa4`. DPA4 uses `dp --pt train`; DPA4c uses `dp --pt-expt train`. The earlier `deepmd_320_a100` passes DPA4c but its DPA4 training fails without the Torch neighbor-list dependency. That failure is preserved.

K40m `deepmd_2210_k40` (DeepMD 2.2.10, TF 2.9, CUDA 10.2) passes a small `se_e2_a` GPU training check. This does not establish DPA4/DPA4c/modern GRACE compatibility on K40m.

Jobs `20950192`, `20950193`, `20950195` consumed a combined 241 single-GPU seconds. Successful cases completed 20 optimizer steps, with finite curves, checkpoint hashes and training-device confirmation. Inputs are explicitly synthetic harmonic pairs, not DFT. Run records are in `runs/mlip-smoke-20261017-*/`; the full interpretation is in `reports/mlip-environment-2026-10-17.tex`. Tiny tests do not provide a production performance comparison.

GRACE/TensorPotential is absent from inspected environments. Its published TensorFlow <=2.20 requirement conflicts with TF 2.21 in the existing DeepMD A100 environment. `environments/grace-a100.proposed.yml` describes a separate proposed installation; it is not installed or locked. Select actual trainable FM checkpoints separately, record revisions/hashes and chemical coverage, then validate loading, fine-tuning, restart and export. No checkpoint was downloaded by these checks.

`configs/model_backends.json` keeps the candidate list extensible: DPA4, DPA4c, GRACE and a legacy DeepMD student path. It defines a future adapter contract; a common execution adapter is not yet implemented.

`scripts/run_mlip_smoke.py` records input hashes and submission intent before `sbatch`, captures the job ID, and collects small records with checksum verification. Recorded run IDs cannot be submitted again silently. To refresh a run, for example:

```powershell
python scripts/run_mlip_smoke.py collect runs/mlip-smoke-20261017-a100-vesin-01/manifest.json
```

### VASP installation

Within an appropriate Slurm allocation, activate the installed CPU/MPI build with:

```bash
source /home/u3129/.local/opt/vasp/6.6.1-oneapi-2025/env.sh
# From a compute-writable calculation directory with valid VASP inputs:
mpirun -np "$SLURM_NTASKS" vasp_std
```

The same environment provides `vasp_gam` and `vasp_ncl`. Use a writable calculation directory such as an explicitly prepared directory under `/home6/u3129`; main home was read-only on the tested compute node. The environment sources pinned shared Intel Compiler/MPI/MKL scripts directly because compute Lmod does not expose the login Intel module name.

Installation provenance and hashes are in `runs/vasp661-oneapi-20261017-05/`. Job `20950123` completed on `apollo31` in 9 min 27 s with 8 allocated CPUs. Four upstream cases passed, including gamma-only, noncollinear SOC and nonlocal vdW; the full upstream suite was not run. No GPU, HDF5 or optional external interfaces were enabled. Use the separately installed PAW library below for production input preparation; bundled test potentials remain private and are for validation.

The licensed source archive, source tree, binaries, test potentials and raw build logs remain on the cluster, outside Git. `scripts/install_vasp.py` records input hashes before submission and guards deployment on successful scheduler state, test coverage, resolved libraries and matching binary checksums. Earlier failed launch attempts and their diagnosed filesystem/module causes are retained in `runs/`.

### PAW pseudopotentials

The user-provided `Pseudopotentials_v6.4.zip` is stored privately, together with its extracted files, under:

```text
/home/u3129/.local/share/vasp/pseudopotentials/64-b3b00158ad69/
    Pseudopotentials_v6.4.zip
    potpaw_PBE/       # 343 potentials
    potpaw_LDA/       # 297 potentials
    registry.json    # identification and checksums, no potential contents
    env.sh
    vasp_potentials.py
```

`configs/vasp/paw-64.json` records all POTCAR identifiers and SHA-256 hashes, the two nested archive hashes and aggregate hashes of all extracted files. Installation evidence is in `runs/paw64-20261017-01/`. The archive and extracted files remain outside Git. Copying this library does not establish DFT convergence or select a functional for the study.

To prepare a calculation on the cluster, explicitly choose one family and exact variants in **POSCAR species order**. For example, the following is only an input-assembly example, not a selected scientific protocol:

```bash
source /home/u3129/.local/share/vasp/pseudopotentials/64-b3b00158ad69/env.sh
# In a new calculation directory, for a POSCAR whose species order is Al Ti:
vasp_potcar --family PBE --variants Al Ti_pv --output POTCAR
```

The helper checks each selected source hash, concatenates in the supplied order and writes `potcar-provenance.json` with family, variants, individual hashes and assembled POTCAR hash. It refuses existing outputs and unknown variants. Preserve that provenance in the run manifest; do not commit POTCAR itself. Keep family and variants consistent across a DFT dataset, and validate cutoffs and other settings for the chosen system. Source `env.sh` above separately from the VASP executable environment; VASP itself reads the calculation's local POTCAR.

To recheck the whole installed library without running VASP:

```bash
python3 "$VASP_PAW_ROOT/vasp_potentials.py" verify "$VASP_PAW_REGISTRY"
```
