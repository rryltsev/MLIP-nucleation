# MLIP-nucleation

Research infrastructure for **Programming Crystal Nucleation through Interfacial Liquid Order in Metallic Alloys**.

The source research brief is preserved in `project/brief.txt`. Scientific manuscripts and reports use LaTeX. This repository contains infrastructure, operational skills and synthetic workflow checks; a production research pipeline and scientific results remain to be developed.

## Infrastructure

- Remote repository: https://github.com/rryltsev/MLIP-nucleation
- Cluster: `umt.imm.uran.ru:22`, user `u3129`, scheduler Slurm.
- Observed Slurm configuration (2026-10-17): 74 unique nodes, 1850 Slurm CPUs, 2 A100, 18 K40m and 62 M2090 GPUs, including configured devices on unavailable nodes. Partition totals overlap. GPU runtime evidence below applies only to the tested allocations.
- DFT engine: VASP 6.6.1 CPU/MPI installed in `/home/u3129/.local/opt/vasp/6.6.1-oneapi-2025`; `vasp_std`, `vasp_gam`, `vasp_ncl` passed four selected upstream cases on `apollo31`. See `reports/vasp-installation-2026-10-17.tex`.
- A100 DPA4/DPA4c/GRACE-1L training and K40m legacy DeepMD training passed small synthetic Slurm checks. Actual DPA4-Mini and GRACE-1L-OMAT weights passed fine-tuning, Adam restart to update24 and ASE/native-LAMMPS export comparisons on A100. Production performance and scientific accuracy remain untested. See `reports/restart-export-validation-2026-10-17.tex` and the earlier reports.

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

1. SSH, inventory, VASP/PAW setup, small GPU training and actual foundation checkpoint fine-tuning checks completed on 2026-10-17. Next audit reference settings and choose the scientific pilot. See the reports below for evidence and limitations.
2. On another computer clone the repository, configure its own credentials and read `project/state.json`.
3. First bounded DFT pilot selected: Al32 FCC/disordered/HCP, 40 geometries, grouped train/validation/test 21/6/13. Calibration finished: all 16 static calculations converged electronically, but the Gamma4 mesh failed scientific convergence; dependent labeling was cancelled. Accepted production labels: 0/40. See `reports/al32-dft-pilot-2026-10-18.tex`. This does not select the final alloy/nucleant or establish novelty.
4. Implement one executable path: structures → converged VASP labels → teacher fine-tuning → independent DFT validation → distillation → student validation → one observable → LaTeX report.
5. Add active learning and rare-event methods after the pilot is validated.

## Al32 DFT pilot

`configs/systems/al32-pilot-v1.json` defines the chemistry, seeds, grouped split,
label units, convergence thresholds and proposed allocation. Prepared geometries
and unvalidated INCAR/KPOINTS live in `datasets/al32-pilot-v1/`; no POTCAR content
belongs in this tree. Geometry generation uses ASE EMT only as a proposal method,
without exporting its energies/forces. These are not equilibrium DFT liquid samples.

With an isolated Python environment matching `environments/al32-preparation.txt`,
`python scripts/prepare_al_pilot.py` creates a new dataset and refuses to overwrite
an existing one. A clone containing the frozen dataset should reuse its exact bytes.
Run `python -m unittest discover -s tests -v` for energy/stress/export contract tests.

`scripts/al_pilot.py stage <unique-run-id> calibrate` transfers the frozen proposals
and checked source to the private cluster work root, without submitting a job.
Run IDs use `al32-calibration-YYYYMMDD-NN`. Submission requires recorded
user authorization for the 128 CPU-hour overall maximum and an explicit `submit` command.
One attempt per phase is allowed for each explicit authorization revision;
all revisions share terminal actual CPUTimeRAW plus full reservations of live jobs.
Failed attempts do not silently regain the budget. A lost submission response
requires checking remote `job.id`, `squeue` and `sacct` before further action.

Collect terminal results with `scripts/al_pilot.py collect <manifest.json>`.
Only an accepted calibration can be passed to `stage <al32-labeling-YYYYMMDD-NN>
label --source <calibration-manifest.json>`. Alternatively, while the calibration
is submitted, add `--after-calibration` to stage a dependent labeling job:
Slurm uses `afterok` and `--kill-on-invalid-dep=yes`, and the compute script also
requires the accepted scientific convergence result, identical system settings
and the exact frozen dataset manifest before starting any labeling calculation.
Phase limits are defined in the immutable configuration and authorization;
previous costs are included before every new submission.
Labeling exports raw force-consistent
DFT energies, forces, stress and virials for ASE/GRACE and DeepMD. Use `backup`
after collection to retain and verify actual private raw-output archives, including
POTCAR, in main home and `artifacts/dft-run-backups/`; these archives never go to Git.

Do not start model fitting with the unvalidated draft inputs or synthetic checkpoints.
Original foundation weights remain the initializations; PAW54/PAW64 energy-reference
compatibility and possible pretraining overlap are recorded in both model cards.

Collected jobs: calibration `20950353` failed the predefined k-mesh criteria;
labeling `20950356` was cancelled before allocation. All 16 calibration labels
passed independent raw-XML, checksum, settings and unit checks, but they use
different protocols and are diagnostics, not a training set. The comparison
Gamma4 versus Gamma5 failed forces on both parents; Gamma5 itself still needs
testing against denser meshes. Thresholds were not relaxed and no new jobs were
submitted during collection. Total allocation cost: 10.4578 CPU-hours.

`runs/al32-calibration-20261018-01/quality-assessment.json` records the decision.
Run `python scripts/assess_al_pilot.py` to repeat the independent audit with the
private archives available. Both terminal run archives have actual private main-home
and workstation copies with every member verified; locations/full hashes are in
each run's `artifact-backup.json`. The failed scientific gate is preserved.

The user authorized a revised dense-mesh study under the same overall cap.
`configs/dft/al32-kmesh-v2.json` freezes Gamma6/7/8 checks on the same two train
parents, fixed Gaussian sigma0.05 and unchanged tolerances. Gamma7 versus Gamma8
must pass first. Then choose Gamma6 only if its 450eV labels agree with the 550eV
Gamma6/7/8 calculations on both parents; otherwise choose Gamma7 only if its own
450/550eV and Gamma8 comparisons pass. No candidate passing means no labeling.
Ten new static calculations are planned. Geometry bytes and splits stay frozen.

Revised limits: 16 ranks for 2h calibration (32 CPUh) and 5h labeling (80 CPUh).
Including 37648 previously allocated CPU-seconds, the worst case is 122.4578 CPUh,
below128. The submitter reconciles all recorded parent jobs with Slurm, refuses
unknown Al32 scheduler jobs or ambiguous submissions, and counts each live job's
full reservation. `--system configs/dft/al32-kmesh-v2.json` selects the revision
when staging; dependent labeling inherits the source configuration.

## Storage

Commit source, configuration, skills, manifests, small curated results, LaTeX and bibliography. Choose LFS or durable external storage before producing large datasets, checkpoints and trajectories. Keep immutable SHA-256 manifests and verify retrieval; cluster scratch is not an archive. VASP POTCAR files and credentials must not be uploaded.

Diagnostic foundation and fine-tuned weights currently have actual private copies in cluster main home and on this workstation, with verified archive retrieval. Git tracks their hashes and locations. This provisional backend is recorded in `models/foundation/` and `runs/fm-finetune-20261017-01/*backup.json`; retention and a shared institutional archive remain unresolved.

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

GRACE is installed separately in `/home6/u3129/mlip-research/envs/grace_ft_a100`: Python 3.11.17, TensorPotential 0.6.1, TensorFlow 2.20.0, tf-keras 2.20.1. Its own CUDA dependencies avoid changing DeepMD environments. Inside an appropriate Slurm allocation, use the activation helper, which also configures TensorFlow's packaged CUDA compiler/libdevice path:

```bash
source /home6/u3129/mlip-research/envs/grace_ft_a100/activate-grace.sh
gracemaker input.yaml
```

GRACE job `20950274` completed on `tesla-a101` in 38 seconds: a reduced-size one-layer model trained five epochs / 20 updates on synthetic labels, with finite metrics and checkpoint weights. The training child process was observed on GPU. All three GRACE GPU attempts consumed 87 GPU-seconds. Earlier failures (CUDA search path and missing scheduler parameters in the test input) are retained under `runs/grace-smoke-20261017-01/` and `-02/`. Installation cache/read-only-home and time-limit failures are also preserved. See `reports/grace-installation-2026-10-17.tex`.

`environments/grace-a100.lock.json` records the accepted environment: explicit Conda package URLs/hashes and 82 exact pip wheels with SHA-256. To rebuild on Linux x86_64, create a new prefix from `grace-a100-conda-linux-64.lock.txt`, then install `grace-a100-pip-linux-64.lock.txt` with `python -m pip install --no-deps --require-hashes -r ...`. The activation helper is prefix-specific. The original `grace-a100.proposed.yml` is retained as proposal provenance. `scripts/continue_grace_setup.py` resumes recorded setup jobs and the latest recorded GPU check; it never blindly retries a training failure.

The small accepted from-scratch test checkpoint (about 100 kB) is backed up in `runs/grace-smoke-20261017-03/checkpoint/`, with the model definition and hashes alongside it. It is a software-test artifact. Subsequent foundation fine-tuning, restart and export checks below used separately downloaded weights.

### Pinned foundation checkpoints and actual fine-tuning

Diagnostic initializations are `DPA4-Mini-OMat24-v20260805` (655504 parameters, 118 types, CC-BY-NC-4.0) and `GRACE-1L-OMAT` (3447148 parameters, 89 types, Academic Software License). Both use a 6-Angstrom cutoff and OMat24 pretraining; type-map coverage is not an accuracy guarantee. Model metadata in `models/foundation/` pins source revisions, SHA-256, licences and private locations. Neither is yet a selected scientific teacher. Before real DFT use, audit pretraining electronic settings, energy references and overlap.

`scripts/foundation_checkpoints.py` retrieves exact source revisions, checks upstream artifact hashes, retains actual workstation copies and transfers verified cluster copies. Weights are read from `/home/u3129/.local/share/mlip-research/foundation-checkpoints/`; this avoids compute-read-only-home cache writes and moving model aliases. No environment was modified for these checks.

GPU job `20950277` ran both models sequentially on one A100 in 73 seconds. Each training child exited 0 and completed 20 optimizer updates using the same explicitly synthetic Al4 diagnostic labels. Original weights were observed immediately after loading; training PIDs were observed on GPU. The Slurm job itself is `FAILED` because its original verification code did not handle checkpoint serialization changes. That failure is preserved. Read-only CPU verification `20950306` subsequently passed both models without repeating training: DPA4 restored 99/102 float state tensors exactly (only output statistics and version buffer changed initially), then updated 83 original tensors; GRACE restored all 19 model tensors exactly, then updated 18. Earlier CPU checks and diagnoses are retained.

Read `reports/foundation-finetuning-2026-10-17.tex`, `runs/fm-finetune-20261017-01/assessment.json` and `runs/fm-verify-20261017-03/result.json` for the original fine-tuning evidence. Archives of final checkpoints/data and initial loaded snapshots are stored privately in cluster main home and have verified workstation copies. The later restart/export checks below do not establish DFT accuracy, MD stability or performance comparisons. Do not use the synthetic fine-tuned checkpoints as scientific teachers.

### Restart and ASE/LAMMPS deployment checks

Both models restored weights and Adam state and continued from update20 to update24. DPA4 restored 413 state tensors (267 optimizer tensors); GRACE restored 62 (38 optimizer tensors). DeepMD 3.2.0's native restart reuses completed loop index19 despite Adam count20. The tested `scripts/hpc/restart-child.py` guards this exact discrepancy, advances the next loop start to20 and rebuilds the extended LR schedule. It does not change the installed package or source checkpoint. Use this checked path for the tested Adam setup; bitwise equivalence to uninterrupted training was not tested.

Export comparisons used the same checkpoint24 in memory, through ASE and through actual native LAMMPS on one A100. Four synthetic geometries cover orthogonal/triclinic cells and periodic images. Energies, forces and tensile stresses passed tolerances fixed before submission. Maximum force differences were 5.63e-6 eV/Angstrom for DPA4 ASE, 1.13e-5 for DPA4 LAMMPS, 7.41e-15 for GRACE ASE and 1.10e-14 for GRACE LAMMPS. These are serialization/runtime comparisons, not errors against DFT.

The accepted mapping is `runs/fm-deployment-20261017-07/assessment.json`; read `reports/restart-export-validation-2026-10-17.tex` for units, tolerances, failed attempts and exact source hashes. Accepted jobs are `20950321` for GRACE, `20950322` for corrected DPA4 restart, `20950326` for DPA4 export/ASE, and `20950337` for DPA4 LAMMPS. Failed aggregate statuses are preserved. All seven GPU attempts consumed 606 GPU-seconds, including compilation and GPU-observation holds; no throughput claim is made.

Use the private native runtimes inside suitable Slurm allocations:

```bash
# DPA4: official LAMMPS 22Jul2025 Update2 + PLUGIN + Intel MPI 2021.17
source /home6/u3129/mlip-research/software/lammps-deepmd-mpi-a33449868448/activate-lammps.sh
lmp -in in.lammps

# GRACE: LAMMPS grace branch + ML-PACE + TensorFlow; serial host, GPU evaluation
source /home6/u3129/mlip-research/software/lammps-grace-b1f4a6c8cc4e/activate-lammps.sh
lmp -in in.lammps
```

For DPA4 the input must explicitly load the plugin before `pair_style deepmd`; see the retained `dpa4/lammps-*/in.lammps` inputs. The tested helper supplies `libtorch`, Torch's CUDA13 libraries and the real CUDA12 runtime required by the native DeepMD wheel. The old bundled pip LAMMPS binary crashes before input parsing and is not the validated executable. The PT `edge_vec` .pt2 schema is deprecated by the installed backend; validate a graph-schema migration before adopting it. Multi-rank/MPI scaling and long MD remain untested.

Source archives, compiled native libraries, checkpoint24, exports and restoration snapshots have actual private cluster-main-home and workstation backups. Their SHA-256 and member manifests are in the corresponding `artifact-backup.json` and `initialization-backup.json` records. Compiler/preview startup for the LaTeX report remains blocked by Windows access error5; the source is preserved and the existing VASP editor was not changed.

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
