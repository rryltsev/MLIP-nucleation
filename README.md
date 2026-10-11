# MLIP-nucleation

Research infrastructure for **Programming Crystal Nucleation through Interfacial Liquid Order in Metallic Alloys**.

The source research brief is preserved in `project/brief.txt`. Scientific manuscripts and reports will be written in LaTeX. This repository currently contains configuration and operational skills, not a working simulation pipeline or scientific results.

## Infrastructure

- Remote repository: https://github.com/rryltsev/MLIP-nucleation
- Cluster: `umt.imm.uran.ru:22`, user `u3129`, scheduler Slurm.
- Reported hardware: 2 A100 GPUs, several dozen K40m GPUs, approximately 1000 CPUs. Partitions, quotas, GPU memory and software compatibility remain unverified.
- DFT engine: VASP. Executable, licensed pseudopotentials, MPI environment and convergence settings must be resolved on the cluster.
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

1. Complete interactive SSH authentication. The user confirmed the server fingerprint on 2026-10-11. Inspect Slurm partitions, quotas and VASP installation without launching calculations.
2. On another computer clone the repository, configure its own credentials and read `project/state.json`.
3. Compare candidate alloy/nucleant systems using literature novelty, experimental access, DFT feasibility and model chemical coverage. Choose one small pilot.
4. Implement one executable path: structures → converged VASP labels → teacher fine-tuning → independent DFT validation → distillation → student validation → one observable → LaTeX report.
5. Add active learning and rare-event methods after the pilot is validated.

## Storage

Commit source, configuration, skills, manifests, small curated results, LaTeX and bibliography. Choose LFS or durable external storage before producing large datasets, checkpoints and trajectories. Keep immutable SHA-256 manifests and verify retrieval; cluster scratch is not an archive. VASP POTCAR files and credentials must not be uploaded.

## SSH

`configs/ssh_config` is a portable configuration, not an installed user setting. Example from the repository root:

```powershell
ssh -F configs/ssh_config mlip-umt hostname
```

Run this command from the repository root: the profile uses the relative path `configs/known_hosts` containing the public server key independently confirmed by the user. Its ED25519 fingerprint is `SHA256:sAMo6jV6KxAx0ntkLR5Gjm5wVWtb4QuRsDvGBhRB01k`. Authentication is machine-local; the current user logs in interactively by password. Do not put credentials in this repository. For unattended execution, set up a machine-local SSH key through an authenticated session, according to cluster policy.

Run the offline configuration check with `python scripts/check_project.py`.
