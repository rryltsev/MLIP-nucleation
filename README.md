# MLIP-nucleation

Research infrastructure for **Programming Crystal Nucleation through Interfacial Liquid Order in Metallic Alloys**.

The source research brief is preserved in `project/brief.txt`. Scientific manuscripts and reports will be written in LaTeX. This repository currently contains configuration and operational skills, not a working simulation pipeline or scientific results.

## Infrastructure

- Remote repository: https://github.com/rryltsev/MLIP-nucleation
- Cluster: `umt.imm.uran.ru:22`, user `u3129`, scheduler Slurm.
- Observed Slurm configuration (2026-10-17): 74 unique nodes, 1850 Slurm CPUs, 2 A100, 18 K40m and 62 M2090 GPUs, including configured devices on unavailable nodes. Partition totals overlap. GPU memory and runtime compatibility remain unverified.
- DFT engine: VASP 6.6.1 CPU/MPI installed in `/home/u3129/.local/opt/vasp/6.6.1-oneapi-2025`; `vasp_std`, `vasp_gam`, `vasp_ncl` passed four selected upstream cases on `apollo31`. See `reports/vasp-installation-2026-10-17.tex`.
- A100 is the initial candidate for training. K40m execution requires a separate compatibility and performance check; no modern CUDA/PyTorch compatibility is assumed.

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

1. SSH, inventory and VASP CPU/MPI installation completed on 2026-10-17. Configure production PAW potentials and agree a bounded Slurm runtime check for existing MLIP environments. See `reports/cluster-inventory-2026-10-17.tex` and `reports/vasp-installation-2026-10-17.tex`.
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

The default account is `imet-2026-2` with QoS `normal`. Candidate partitions are `apollo` for CPU DFT and `v100` for A100 training. Existing `deepmd_320_a100`, `deepmd_229_cpu` and `deepmd_2210_k40` environments activate and resolve `python`, `dp` and `lmp`; package imports and compute-node execution have not been tested. Their metadata and two additional DPA-labelled environments are documented in the LaTeX report. No shared environment was modified.

### VASP installation

Within an appropriate Slurm allocation, activate the installed CPU/MPI build with:

```bash
source /home/u3129/.local/opt/vasp/6.6.1-oneapi-2025/env.sh
# From a compute-writable calculation directory with valid VASP inputs:
mpirun -np "$SLURM_NTASKS" vasp_std
```

The same environment provides `vasp_gam` and `vasp_ncl`. Use a writable calculation directory such as an explicitly prepared directory under `/home6/u3129`; main home was read-only on the tested compute node. The environment sources pinned shared Intel Compiler/MPI/MKL scripts directly because compute Lmod does not expose the login Intel module name.

Installation provenance and hashes are in `runs/vasp661-oneapi-20261017-05/`. Job `20950123` completed on `apollo31` in 9 min 27 s with 8 allocated CPUs. Four upstream cases passed, including gamma-only, noncollinear SOC and nonlocal vdW; the full upstream suite was not run. No GPU, HDF5 or optional external interfaces were enabled. A full licensed production PAW dataset still needs configuration; bundled test potentials remain private and are for validation.

The licensed source archive, source tree, binaries, test potentials and raw build logs remain on the cluster, outside Git. `scripts/install_vasp.py` records input hashes before submission and guards deployment on successful scheduler state, test coverage, resolved libraries and matching binary checksums. Earlier failed launch attempts and their diagnosed filesystem/module causes are retained in `runs/`.
